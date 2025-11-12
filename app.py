import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ================== Qwen3-Max 配置 ==================

api_key = (
    os.getenv("DASHSCOPE_API_KEY")
    or (st.secrets.get("DASHSCOPE_API_KEY") if hasattr(st, "secrets") else None)
)

if not api_key:
    st.error("请配置 DASHSCOPE_API_KEY（环境变量或 .streamlit/secrets.toml）")
    st.stop()

# 中国区 DashScope OpenAI-兼容接口
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)

MODEL_NAME = "qwen3-max"


# ================== 通用：带重试的 Qwen 调用 ==================

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


# ================== 自动列映射：用前3行示例猜两表对应列 ==================

def build_column_samples(df, n=3):
    head = df.head(n)
    samples = {}
    for col in df.columns:
        vals = ["" if pd.isna(v) else str(v) for v in head[col].tolist()]
        samples[col] = vals
    return samples


def infer_column_mapping(df1, df2):
    """
    使用 Qwen 根据列名 + 前几行样例，推断表1列 到 表2列 的语义映射。
    返回 dict: {left_col: right_col}
    """
    left_samples = build_column_samples(df1)
    right_samples = build_column_samples(df2)

    system_prompt = """
你是一个表头匹配助手。
有两个表：表1(主表) 和 表2(对照表)。
给你每个表的列名及前几行示例，请判断哪些列表示相同含义。

只返回 JSON：
{
  "mappings": [
    {"left": "表1列名", "right": "表2列名"},
    ...
  ]
}

要求：
1. 列名相同且示例值类型相似时优先匹配。
2. 可以根据中文含义、英文缩写、示例值（如都是金额/数量/编码）推断。
3. 没把握就不要匹配，宁缺毋滥。
4. 每个表1列最多对应表2中的一个列；不要重复。
5. 不要输出任何非 JSON 内容。
"""
    user_content = (
        "表1列与样例值：\n"
        + json.dumps(left_samples, ensure_ascii=False)
        + "\n\n表2列与样例值：\n"
        + json.dumps(right_samples, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = call_qwen(messages, max_tokens=800, temperature=0)
        data = json.loads(raw)
        mappings = data.get("mappings", [])
        result = {}
        used_right = set()
        for m in mappings:
            l = m.get("left")
            r = m.get("right")
            if (
                isinstance(l, str)
                and isinstance(r, str)
                and l in df1.columns
                and r in df2.columns
                and r not in used_right
            ):
                result[l] = r
                used_right.add(r)
        return result
    except Exception:
        return {}


# ================== Streamlit 基本设置 ==================

st.set_page_config(
    page_title="智能数据查询 & 数据对照助手",
    page_icon="📊",
    layout="centered",
)

st.title("📊 智能数据查询 & 数据对照助手")
st.caption(
    "上传 1~2 个 Excel/CSV：Tab1 智能查询，Tab2 自动对齐相关列并并排展示，方便人工核对。"
)

# ================== 状态初始化 ==================

if "df1" not in st.session_state:
    st.session_state.df1 = None
if "df2" not in st.session_state:
    st.session_state.df2 = None
if "schema1" not in st.session_state:
    st.session_state.schema1 = ""
if "schema2" not in st.session_state:
    st.session_state.schema2 = ""
if "auto_mapping" not in st.session_state:
    st.session_state.auto_mapping = {}


# ================== 上传数据：支持 1 或 2 个文件 ==================

uploaded_files = st.file_uploader(
    "上传数据文件（支持 .xlsx / .xls / .csv，可上传 1 或 2 个文件）",
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
    # 取前两个文件
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

# 自动映射（如果有两个表且还没算过）
if df2 is not None and not st.session_state.auto_mapping:
    with st.spinner("正在自动分析两表字段对应关系（读取前 3 行示例）..."):
        st.session_state.auto_mapping = infer_column_mapping(df1, df2)

auto_mapping = st.session_state.auto_mapping

# ================== 预览上传的表 & 自动映射 ==================

with st.expander("📄 文件预览 & 自动字段映射", expanded=True):
    st.markdown("**表1 字段 & 示例（前10行）：**")
    st.write(", ".join(df1.columns))
    st.dataframe(df1.head(10), width="stretch")

    if df2 is not None:
        st.markdown("---")
        st.markdown("**表2 字段 & 示例（前10行）：**")
        st.write(", ".join(df2.columns))
        st.dataframe(df2.head(10), width="stretch")

        if auto_mapping:
            st.markdown("---")
            st.markdown("**自动推断的字段映射（表1列 → 表2列）：**")
            rows = [{"表1列": l, "表2列": r} for l, r in auto_mapping.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.caption("（暂未推断出可靠映射，如需对照请确保列名尽量一致。）")


# ================== Tabs ==================

tab_query, tab_check = st.tabs(["🔍 智能查询", "✅ 数据对照"])


# ================== Tab 1：智能查询（跟之前一样） ==================

with tab_query:
    st.markdown("### 🔍 智能查询（基于表1）")
    default_example = "例如：按 Item Number 汇总未税金额；或：给出每行未税金额的 Excel 公式"
    question = st.text_input("请输入你的问题：", placeholder=default_example, key="q_query")
    go_query = st.button("生成结果和 Excel 公式", key="btn_query")

    if go_query:
        q = (question or "").strip()
        if not q:
            st.warning("问题不能为空。")
        else:
            schema_text = st.session_state.schema1
            sample_rows = df1.head(5).astype(str).to_dict(orient="records")

            system_prompt_code = """
你是一个数据分析助手。
现在有 pandas.DataFrame df。
请根据字段信息和示例数据生成可执行的 Python 代码来回答问题。

只输出代码本身，不要 Markdown 或解释。
要求：
1. 不导入库，不读写文件，不联网。
2. 使用已有 df。
3. 表格结果放在 result_df；简单结果放在 result。
4. 不要 print/input，不要修改原始 df。
"""
            messages_code = [
                {"role": "system", "content": system_prompt_code},
                {
                    "role": "user",
                    "content": (
                        f"字段信息：\n{schema_text}\n\n"
                        f"前 5 行示例数据（字符串，仅供参考）：\n"
                        f"{json.dumps(sample_rows, ensure_ascii=False)}\n\n"
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

            local_vars = {"df": df1.copy()}
            try:
                exec(code, {}, local_vars)
            except Exception as e:
                st.error(f"执行分析逻辑出错：{e}")
                # st.code(code, language="python")
                st.stop()

            preview = None
            is_table = False
            if "result_df" in local_vars and isinstance(local_vars["result_df"], pd.DataFrame):
                preview = local_vars["result_df"].head(20)
                is_table = True
            elif "result" in local_vars:
                preview = local_vars["result"]
            else:
                st.error("未获取到可用结果（缺少 result_df 或 result）。")
                st.stop()

            # Excel 公式示例
            system_prompt_excel = """
你是一个 Excel 公式助手。
根据字段信息和问题，输出 JSON：
{
  "explanation": "中文解释 1-3 句",
  "excel_formulas": ["公式1", "公式2", "公式3"]
}
要求：
1. 假设数据在一张表，首行是表头。
2. 用常见函数，如 SUMIFS, IF, XLOOKUP 等。
3. 使用 $列:$列 绝对列引用。
4. 不要输出 JSON 外的内容。
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
                st.markdown("### 📎 Excel 公式示例（可复制）")
                for f in excel_formulas:
                    st.code(f, language="excel")
            else:
                st.caption("（本次未生成可用的 Excel 公式示例。）")


# ================== Tab 2：数据对照（只输出对照视图，不判对错） ==================

with tab_check:
    st.markdown("### ✅ 数据对照（根据索引 & 自动映射并排展示）")

    if df2 is None:
        st.info("要使用数据对照功能，请上传两个文件（表1 和 表2）。")
    else:
        st.markdown(
            "使用方式：\n"
            "1. 在下方输入索引列（只用表1列名，例如：Item Number, 规格型号）；\n"
            "2. 输入你关心的字段（只用表1列名，如：开票型号, 开票名称, 未税金额 等）；\n"
            "3. 系统会根据自动映射找到表2对应列，做一次合并，"
            "   把这些字段以“字段_表1 / 字段_表2”的形式并排展示，方便你自己判断是否一致。"
        )

        key_text = st.text_input(
            "索引列（用来匹配两表记录，只写表1的列名，逗号分隔）：",
            placeholder="例如：Item Number, 规格型号",
            key="q_keys",
        )

        field_text = st.text_input(
            "要对照展示的字段（只写表1的列名，逗号分隔，例如：开票型号, 开票名称, 开票数量, 未税单价, 未税金额）：",
            key="q_fields",
        )

        go_view = st.button("生成对照视图", key="btn_view")

        if go_view:
            # 解析索引列
            raw_keys = (key_text or "").replace("，", ",")
            keys = [k.strip() for k in raw_keys.split(",") if k.strip()]

            if not keys:
                st.error("请至少填写一个索引列（使用表1的列名）。")
            else:
                miss1 = [k for k in keys if k not in df1.columns]
                mapped_keys = {}
                for k in keys:
                    if k in df1.columns:
                        # 优先用自动映射，否则尝试同名
                        r = auto_mapping.get(k, k if k in df2.columns else None)
                        if r:
                            mapped_keys[k] = r
                miss2 = [k for k in keys if k not in mapped_keys]

                if miss1 or miss2:
                    msg = []
                    if miss1:
                        msg.append("这些索引列不在表1中：" + ", ".join(miss1))
                    if miss2:
                        msg.append(
                            "这些索引列在表2中找不到对应列（自动映射和同名都失败）："
                            + ", ".join(miss2)
                        )
                    st.error("；".join(msg))
                else:
                    # 解析字段列表（只用表1列名）
                    raw_fields = (field_text or "").replace("，", ",")
                    fields_left = [f.strip() for f in raw_fields.split(",") if f.strip()]

                    if not fields_left:
                        # 没填就用所有自动映射字段里，排除索引列
                        fields_left = [
                            l for l in auto_mapping.keys() if l not in keys
                        ]

                    if not fields_left:
                        st.error("没有可用于对照展示的字段，请在上方填写字段名。")
                    else:
                        # 为每个字段找表2对应列
                        compare_pairs = []
                        skipped = []
                        for l in fields_left:
                            if l not in df1.columns:
                                skipped.append(f"{l}（不在表1中）")
                                continue
                            r = auto_mapping.get(l, l if l in df2.columns else None)
                            if r and r in df2.columns:
                                compare_pairs.append((l, r))
                            else:
                                skipped.append(f"{l}（在表2中未找到对应列）")

                        if not compare_pairs:
                            st.error(
                                "没有找到可用的对照字段。"
                                "请检查字段名是否正确，或确保两表列名足够接近以便自动映射。"
                            )
                        else:
                            if skipped:
                                st.caption("以下字段未被对照展示：" + "， ".join(skipped))

                            # 准备重命名并合并（核心：只展示，不判是否一致）
                            left = df1.copy()
                            right = df2.copy()

                            # 索引列重命名（让表2键列与表1同名）
                            index_rename = {
                                mapped_keys[k]: k for k in keys if mapped_keys[k] != k
                            }
                            right = right.rename(columns=index_rename)

                            # 值字段重命名（让表2列名先对齐表1，再通过 suffix 区分）
                            value_rename = {r: l for (l, r) in compare_pairs if l != r}
                            right = right.rename(columns=value_rename)

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

                            # 构造对照视图列顺序
                            compare_cols = [l for (l, _) in compare_pairs]
                            view_cols = list(keys)
                            for col in compare_cols:
                                col_l = f"{col}_表1"
                                col_r = f"{col}_表2"
                                if col_l in both.columns:
                                    view_cols.append(col_l)
                                if col_r in both.columns:
                                    view_cols.append(col_r)

                            st.markdown("### 📊 对照结果概览")
                            st.write(f"- 使用索引列：{', '.join(keys)}")
                            st.write(f"- 仅在表1中的记录数：{len(only_in_1)}")
                            st.write(f"- 仅在表2中的记录数：{len(only_in_2)}")
                            st.write(f"- 两表都存在（可对照）的记录数：{len(both)}")

                            if len(only_in_1) > 0:
                                st.markdown("**仅在表1中的样例（最多20行，仅展示索引列）：**")
                                st.dataframe(only_in_1[keys].head(20), width="stretch")

                            if len(only_in_2) > 0:
                                st.markdown("**仅在表2中的样例（最多20行，仅展示索引列）：**")
                                st.dataframe(only_in_2[keys].head(20), width="stretch")

                            st.markdown("### 🔍 匹配记录的字段对照（前 200 行）")
                            if len(both) == 0:
                                st.write("没有匹配成功的记录。")
                            else:
                                st.dataframe(
                                    both[view_cols].head(200),
                                    width="stretch"
                                )
                                st.caption(
                                    "说明：每个字段会以“字段_表1 / 字段_表2”并排展示，"
                                    "你可以直接肉眼或导出后用 Excel 比较是否一致。"
                                )

