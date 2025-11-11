import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ========== 配置 Qwen3-Max via DashScope ==========

api_key = (
    os.getenv("DASHSCOPE_API_KEY")
    or (st.secrets.get("DASHSCOPE_API_KEY") if hasattr(st, "secrets") else None)
)

if not api_key:
    st.error("请配置 DASHSCOPE_API_KEY（环境变量或 .streamlit/secrets.toml）")
    st.stop()

# 按实际区域选 base_url，中国区：
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)

MODEL_NAME = "qwen3-max"


# ========== 通用：带重试的 Qwen 调用 ==========

def call_qwen(messages, max_tokens=800, temperature=0):
    last_err = None
    for _ in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if "timed out" in str(e).lower():
                continue
            break
    raise last_err


# ========== Streamlit 页面基础配置 ==========

st.set_page_config(
    page_title="智能数据查询 & 数据校对助手",
    page_icon="📊",
    layout="centered",
)

st.title("📊 智能数据查询 & 数据校对助手")
st.caption("上传 1~2 个 Excel/CSV → 智能查询 或 数据校对 → 返回结果预览 + Excel 公式示例（免登录，对外可用）")

# ========== 状态初始化 ==========

if "df1" not in st.session_state:
    st.session_state.df1 = None
if "df2" not in st.session_state:
    st.session_state.df2 = None
if "schema1" not in st.session_state:
    st.session_state.schema1 = ""
if "schema2" not in st.session_state:
    st.session_state.schema2 = ""


# ========== 上传数据：支持 1 或 2 个文件 ==========

uploaded_files = st.file_uploader(
    "上传数据文件（支持 .xlsx / .xls / .csv，可上传 1 或 2 个）",
    type=["xlsx", "xls", "csv"],
    accept_multiple_files=True,
)

df1 = df2 = None

def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    else:
        return pd.read_excel(f)

if uploaded_files:
    # 最多只取前两个，简单直接
    if len(uploaded_files) >= 1:
        try:
            df1 = load_file(uploaded_files[0])
        except Exception as e:
            st.error(f"第一个文件读取失败：{e}")
            st.stop()
        if df1.empty:
            st.error("第一个文件为空或无法解析为有效表格")
            st.stop()

    if len(uploaded_files) >= 2:
        try:
            df2 = load_file(uploaded_files[1])
        except Exception as e:
            st.error(f"第二个文件读取失败：{e}")
            st.stop()
        if df2.empty:
            st.error("第二个文件为空或无法解析为有效表格")
            st.stop()

    st.session_state.df1 = df1
    st.session_state.df2 = df2

    if df1 is not None:
        schema_lines1 = [f"{c} ({str(df1[c].dtype)})" for c in df1.columns]
        st.session_state.schema1 = "\n".join(schema_lines1)
    if df2 is not None:
        schema_lines2 = [f"{c} ({str(df2[c].dtype)})" for c in df2.columns]
        st.session_state.schema2 = "\n".join(schema_lines2)

df1 = st.session_state.df1
df2 = st.session_state.df2

if df1 is None:
    st.info("请至少上传一个文件。")
    st.stop()

# 展示预览
with st.expander("📄 文件预览", expanded=True):
    st.markdown("**表1（第一个文件）字段与示例：**")
    st.write(", ".join(df1.columns))
    st.dataframe(df1.head(10), width="stretch")
    if df2 is not None:
        st.markdown("---")
        st.markdown("**表2（第二个文件）字段与示例：**")
        st.write(", ".join(df2.columns))
        st.dataframe(df2.head(10), width="stretch")


# ========== Tab 布局：智能查询 / 数据校对 ==========

tab_query, tab_check = st.tabs(["🔍 智能查询", "✅ 数据校对"])


# ========== Tab 1：智能查询（仅用表1，延用你现有逻辑） ==========

with tab_query:
    st.markdown("### 🔍 智能查询（基于第一个表）")
    default_example = "例如：按渠道汇总本月 GMV 排名前 5 的渠道；或：给我每行的毛利率 Excel 公式"
    question = st.text_input("请输入你的问题：", placeholder=default_example, key="q_query")
    go_query = st.button("生成结果和 Excel 公式", key="btn_query")

    if go_query:
        q = (question or "").strip()
        if not q:
            st.warning("问题不能为空。")
        else:
            schema_text = st.session_state.schema1
            sample_rows = df1.head(5).to_dict(orient="records")

            # 1) 生成 pandas 代码
            system_prompt_code = """
你是一个数据分析助手。
现在有一个 pandas.DataFrame，变量名为 df。
我会给你 df 的字段信息、示例数据和用户的问题。

请只输出可以直接执行的 Python 代码，不要加任何解释或 Markdown，不要加 ```。
要求：
1. 不要导入任何库（pandas 已经导入）。
2. 不要读写文件，不要进行网络请求。
3. 直接使用已有的 df。
4. 如果答案是一个表，把最终结果放在 result_df 变量中。
5. 如果答案是单个数字或少量文本，把结果放在 result 变量中。
6. 不要打印，不要 input。
7. 不要修改原始 df，如需处理请使用 df 的拷贝。
8. 代码尽量简短。
"""
            messages_code = [
                {"role": "system", "content": system_prompt_code},
                {
                    "role": "user",
                    "content": (
                        f"字段信息：\n{schema_text}\n\n"
                        f"前 5 行示例数据：\n{json.dumps(sample_rows, ensure_ascii=False)}\n\n"
                        f"用户问题：{q}"
                    ),
                },
            ]

            with st.spinner("正在用 Qwen3-Max 分析数据..."):
                try:
                    code = call_qwen(messages_code, max_tokens=600, temperature=0)
                except Exception as e:
                    st.error(f"生成分析逻辑失败：{e}")
                    st.stop()

            # 2) 执行代码
            local_vars = {"df": df1.copy()}
            try:
                exec(code, {}, local_vars)
            except Exception as e:
                st.error(f"执行分析逻辑出错：{e}")
                # 如需排查，可临时打印生成代码：
                # st.code(code, language="python")
                st.stop()

            preview = None
            is_table = False
            if "result_df" in local_vars and isinstance(local_vars["result_df"], pd.DataFrame):
                result_df = local_vars["result_df"]
                preview = result_df.head(20)
                is_table = True
            elif "result" in local_vars:
                preview = local_vars["result"]
            else:
                st.error("未获取到可用结果（缺少 result_df 或 result）。")
                st.stop()

            # 3) 生成 Excel 公式示例
            system_prompt_excel = """
你是一个精简的 Excel 公式助手。
现在给你字段信息和用户的问题，请输出严格 JSON：
{
  "explanation": "用中文解释大致的统计/计算逻辑，1-3 句话",
  "excel_formulas": ["公式1", "公式2", "公式3"]
}
要求：
1. 使用给定字段名，假设数据在一张表中，第一行是表头。
2. 公式使用通用形式，如 =SUMIFS(...), =IF(...), =XLOOKUP(...).
3. 使用列绝对引用形式，如 $B:$B。
4. 不要返回多余文本。
"""
            messages_excel = [
                {"role": "system", "content": system_prompt_excel},
                {
                    "role": "user",
                    "content": (
                        f"字段信息：\n{schema_text}\n\n"
                        f"用户问题：{q}"
                    ),
                },
            ]

            explanation = ""
            excel_formulas = []
            with st.spinner("正在生成 Excel 公式示例..."):
                try:
                    raw = call_qwen(messages_excel, max_tokens=400, temperature=0)
                    try:
                        j = json.loads(raw)
                        explanation = j.get("explanation", "") or ""
                        excel_formulas = j.get("excel_formulas", []) or []
                    except json.JSONDecodeError:
                        explanation = raw
                        excel_formulas = []
                except Exception as e:
                    explanation = f"（生成 Excel 公式说明失败：{e}）"

            # 展示结果
            st.markdown("### ✅ 分析结果")
            if explanation:
                st.markdown(f"**说明：** {explanation}")
            if is_table:
                st.markdown("**结果预览（最多 20 行）：**")
                st.dataframe(preview, width="stretch")
            else:
                st.markdown("**结果：**")
                st.write(preview)

            if excel_formulas:
                st.markdown("### 📎 Excel 公式示例（可直接复制）")
                for f in excel_formulas:
                    st.code(f, language="excel")
            else:
                st.caption("（本次未生成可用的 Excel 公式示例。）")


# ========== Tab 2：数据校对（两个表对比） ==========

with tab_check:
    st.markdown("### ✅ 数据校对（两个表间差异对比）")

    if df2 is None:
        st.info("要使用数据校对功能，请上传两个文件（表1 和 表2）。")
    else:
        st.markdown(
            "说明：输入两个表共有的索引列，用来匹配行，系统会："
            "- 找出只在表1/表2存在的记录；"
            "- 对共有记录的共有字段逐列比对，给出不一致明细；"
            "- 自动生成 Excel 校对公式示例。"
        )

        # 提示共同列，方便用户选
        common_cols = [c for c in df1.columns if c in df2.columns]
        if common_cols:
            st.caption(f"两个表共有字段示例：{', '.join(common_cols[:10])}"
                       + (" ..." if len(common_cols) > 10 else ""))
        else:
            st.error("两个表没有任何同名列，无法根据列名做索引匹配。")
            st.stop()

        key_text = st.text_input(
            "请输入作为索引的字段名（逗号分隔，例如：订单号, 日期）：",
            key="q_keys",
        )

        go_check = st.button("执行校对", key="btn_check")

        if go_check:
            raw = (key_text or "").strip()
            if not raw:
                st.warning("请先输入至少一个索引列名。")
            else:
                # 解析 key 列名
                keys = [k.strip() for k in raw.replace("，", ",").split(",") if k.strip()]
                # 校验 key 是否存在于两表
                not_in_1 = [k for k in keys if k not in df1.columns]
                not_in_2 = [k for k in keys if k not in df2.columns]
                if not keys or not_in_1 or not_in_2:
                    msg = []
                    if not keys:
                        msg.append("没有解析出有效的索引列名。")
                    if not_in_1:
                        msg.append(f"以下索引列不在表1中：{', '.join(not_in_1)}")
                    if not_in_2:
                        msg.append(f"以下索引列不在表2中：{', '.join(not_in_2)}")
                    st.error("；".join(msg))
                else:
                    # ====== 开始校对逻辑 ======
                    left = df1.copy()
                    right = df2.copy()

                    # 外连接对齐
                    merged = left.merge(
                        right,
                        on=keys,
                        how="outer",
                        indicator=True,
                        suffixes=("_表1", "_表2"),
                    )

                    only_in_1 = merged[merged["_merge"] == "left_only"]
                    only_in_2 = merged[merged["_merge"] == "right_only"]
                    both = merged[merged["_merge"] == "both"].copy()

                    # 要比对的列：在两表都存在、且不是key的列
                    compare_cols = [
                        c for c in common_cols
                        if c not in keys
                    ]

                    mismatch_details = []
                    for col in compare_cols:
                        col_l = f"{col}_表1"
                        col_r = f"{col}_表2"
                        if col_l in both.columns and col_r in both.columns:
                            # 相等逻辑：值相等或同时为 NaN 视为一致
                            l_vals = both[col_l]
                            r_vals = both[col_r]
                            equal = (l_vals == r_vals) | (l_vals.isna() & r_vals.isna())
                            diff_rows = both[~equal][keys + [col_l, col_r]]
                            if not diff_rows.empty:
                                mismatch_details.append((col, diff_rows.head(50)))

                    # ====== 展示结果 ======
                    st.markdown("### 🔎 校对结果总览")

                    st.write(f"- 使用索引列：{', '.join(keys)}")
                    st.write(f"- 仅在表1中的记录数：{len(only_in_1)}")
                    st.write(f"- 仅在表2中的记录数：{len(only_in_2)}")
                    st.write(f"- 索引匹配成功（两表均存在）的记录数：{len(both)}")

                    if only_in_1 is not None and len(only_in_1) > 0:
                        st.markdown("**仅在表1中的样例记录（最多 20 行）：**")
                        st.dataframe(only_in_1[keys].head(20), width="stretch")

                    if only_in_2 is not None and len(only_in_2) > 0:
                        st.markdown("**仅在表2中的样例记录（最多 20 行）：**")
                        st.dataframe(only_in_2[keys].head(20), width="stretch")

                    if mismatch_details:
                        st.markdown("### ❌ 字段不一致明细（示例）")
                        for col, diff in mismatch_details:
                            st.markdown(f"**字段：{col}**（显示前 50 条差异）")
                            st.dataframe(diff, width="stretch")
                    else:
                        st.markdown("### ✅ 公共字段在匹配行中未发现明显差异（基于选择的索引列）")

                    # ====== 生成 Excel 校对公式示例 ======
                    # 用 Qwen 生成几条通用的 VLOOKUP/XLOOKUP 对账公式
                    schema_text_1 = st.session_state.schema1
                    schema_text_2 = st.session_state.schema2
                    compare_cols_sample = compare_cols[:5]

                    prompt_excel = f"""
你是一个 Excel 对账公式助手。
现在有两个表：表1 和 表2，用户用索引列 {', '.join(keys)} 做匹配。
两个表共有字段示例（可用于比对）：{', '.join(compare_cols_sample)}。
请给出 2-4 条通用的 Excel 公式示例，帮助用户在 Excel 里做同样的校对，包括：
1. 检查某个索引在表2中是否存在；
2. 比较某个字段在表1和表2中的值是否一致。

要求：
- 使用中文解释简短说明。
- 公式使用 XLOOKUP 或 VLOOKUP + IF 等常见写法。
- 假设表1在工作表 Sheet1，表2在 Sheet2，首行是表头。
- 使用绝对引用列范围，如 Sheet2!$A:$A。
输出格式为严格 JSON：
{{
  "explanation": "一句或几句说明整体校对思路",
  "excel_formulas": ["公式1", "公式2", "公式3", ...]
}}
只返回 JSON，不要多余文字。
"""
                    messages_excel2 = [
                        {"role": "system", "content": "你擅长生成用于两表对账的 Excel 公式。"},
                        {"role": "user", "content": prompt_excel},
                    ]

                    explanation2 = ""
                    excel_formulas2 = []
                    try:
                        raw2 = call_qwen(messages_excel2, max_tokens=500, temperature=0)
                        try:
                            j2 = json.loads(raw2)
                            explanation2 = j2.get("explanation", "") or ""
                            excel_formulas2 = j2.get("excel_formulas", []) or []
                        except json.JSONDecodeError:
                            explanation2 = raw2
                            excel_formulas2 = []
                    except Exception as e:
                        explanation2 = f"（生成 Excel 校对公式示例失败：{e}）"

                    st.markdown("### 📎 Excel 对账公式示例")
                    if explanation2:
                        st.markdown(f"**说明：** {explanation2}")
                    if excel_formulas2:
                        for f in excel_formulas2:
                            st.code(f, language="excel")
                    else:
                        st.caption("（暂未生成 Excel 校对公式示例，可手动使用 XLOOKUP / VLOOKUP 参考上述差异。）")
