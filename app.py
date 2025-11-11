import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ================== Qwen3-Max 基础配置 ==================

api_key = (
    os.getenv("DASHSCOPE_API_KEY")
    or (st.secrets.get("DASHSCOPE_API_KEY") if hasattr(st, "secrets") else None)
)

if not api_key:
    st.error("请配置 DASHSCOPE_API_KEY（环境变量或 .streamlit/secrets.toml）")
    st.stop()

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,
)

MODEL_NAME = "qwen3-max"


# ================== 通用：带重试的 LLM 调用 ==================

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


# ================== 自动列映射（两表字段含义匹配） ==================

def build_column_samples(df, n=3):
    head = df.head(n)
    samples = {}
    for col in df.columns:
        vals = ["" if pd.isna(v) else str(v) for v in head[col].tolist()]
        samples[col] = vals
    return samples


def infer_column_mapping(df1, df2):
    """
    使用模型，根据列名+前几行示例，推断“表1列 -> 表2列”的对应关系。
    返回 dict: {表1列: 表2列}，宁缺毋滥。
    """
    left_samples = build_column_samples(df1)
    right_samples = build_column_samples(df2)

    system_prompt = """
你是一个通用的字段匹配助手。

现在有两个数据表：
- 表1：主表
- 表2：对照表

我会给你：
- 每个表的列名
- 每个列前几行的示例值

你的任务：
- 判断哪些列在两个表中表示“相同含义”的数据（例如：商品编码 vs 物料编码，数量 vs 开票数量）。

输出要求（只允许输出 JSON，对象结构如下）：
{
  "mappings": [
    {"left": "表1列名", "right": "表2列名"},
    ...
  ]
}

规则：
1. 可以利用列名相似度（中英文、缩写）、示例值模式（金额、数量、编码）等进行判断。
2. 有较高把握时再建立映射；不确定就不要映射。
3. 一个表1列至多对应一个表2列；避免重复和冲突。
4. 只输出 JSON，不要解释文字。
"""
    user_content = (
        "表1列及样例值：\n"
        + json.dumps(left_samples, ensure_ascii=False)
        + "\n\n表2列及样例值：\n"
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
    except Exception:
        return {}

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


# ================== 解析自然语言对照请求（通用版） ==================

def parse_compare_request(question: str, df1_columns):
    """
    从自然语言中抽取：
    - keys: 用作索引/匹配的列（表1列名）
    - fields: 想要对照展示的列（表1列名）

    要求（由模型执行，我们只在本地做兜底过滤）：
    - 列名必须来自 df1_columns。
    - 可以根据语义短语推断：如“作为索引/主键/匹配条件”等。
    - 对照字段通常是被点名的列，或与金额、数量、名称等相关的列。

    返回: (keys: list[str], fields: list[str])
    """
    cols_str = ", ".join(df1_columns)

    system_prompt = f"""
你是一个通用数据任务解析助手。

你会收到：
1. 一组列名（来自表1）：[{cols_str}]
2. 用户用自然语言描述的需求（可能涉及“根据某些列做索引/匹配”、“输出/对比某些字段”等）。

你的任务：
- 从用户描述中，提取：
  - "keys": 用于匹配两表记录的列名列表（索引列、主键、匹配条件等）。
  - "fields": 需要在结果中展示/对照的列名列表。

约束：
1. "keys" 和 "fields" 中的每个列名必须是给定列名列表中的一个。
2. 如果用户出现“作为索引/匹配/根据……关联”等字样，优先从这些词附近选出 keys。
3. 如果用户点名了一些字段（如“开票数量”、“未税金额”），把这些放入 fields。
4. 如果用户没说清，可以根据常识做合理猜测（例如包含 "编号"、"ID" 的列适合作为 keys）。
5. 无法判断的部分留空，不要乱造列名。
6. 只输出一个 JSON 对象，格式如下：
{{
  "keys": ["列名1", "列名2", ...],
  "fields": ["列名A", "列名B", ...]
}}
不要输出 JSON 以外的任何内容。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        raw = call_qwen(messages, max_tokens=400, temperature=0)
        data = json.loads(raw)
    except Exception:
        return [], []

    keys = [k for k in data.get("keys", []) if k in df1_columns]
    fields = [f for f in data.get("fields", []) if f in df1_columns]
    return keys, fields


# ================== Streamlit 初始化 ==================

st.set_page_config(
    page_title="智能数据查询 & 对照助手",
    page_icon="📊",
    layout="centered",
)

st.title("📊 智能数据查询 & 对照助手")
st.caption("上传 1~2 个 Excel/CSV：左边智能查询，右边自然语言/半自动字段映射，输出并排对照表。")

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

# ================== 上传文件 ==================

uploaded_files = st.file_uploader(
    "上传数据文件（支持 .xlsx / .xls / .csv，可 1~2 个）",
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
        st.session_state.schema1 = "\n".join(
            f"{c} ({str(df1[c].dtype)})" for c in df1.columns
        )
    if df2 is not None:
        st.session_state.schema2 = "\n".join(
            f"{c} ({str(df2[c].dtype)})" for c in df2.columns
        )

df1 = st.session_state.df1
df2 = st.session_state.df2

if df1 is None:
    st.info("请至少上传一个文件。")
    st.stop()

# 自动字段映射（有两个表时）
if df2 is not None and not st.session_state.auto_mapping:
    with st.spinner("正在自动分析两表字段对应关系（前 3 行示例）..."):
        st.session_state.auto_mapping = infer_column_mapping(df1, df2)

auto_mapping = st.session_state.auto_mapping

# ================== 预览 & 映射 ==================

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
            st.markdown("**自动推断的字段映射（表1 → 表2）：**")
            rows = [{"表1列": l, "表2列": r} for l, r in auto_mapping.items()]
            st.dataframe(pd.DataFrame(rows), width="stretch")
        else:
            st.caption("（未推断出可靠映射，后续对照时会尽量使用同名列。）")

# ================== Tabs ==================

tab_query, tab_compare = st.tabs(["🔍 智能查询", "✅ 数据对照"])


# ================== Tab 1：智能查询（表1） ==================

with tab_query:
    st.markdown("### 🔍 智能查询（基于表1）")
    default_example = "例如：按渠道汇总未税金额，或生成每行未税金额的 Excel 公式"
    q = st.text_input("请输入你的问题：", placeholder=default_example, key="q_query")
    go = st.button("生成结果和 Excel 公式", key="btn_query")

    if go:
        question = (q or "").strip()
        if not question:
            st.warning("问题不能为空。")
        else:
            schema_text = st.session_state.schema1
            sample_rows = df1.head(5).astype(str).to_dict(orient="records")

            # 生成分析代码（通用，不绑定具体业务）
            system_prompt_code = """
你是一个数据分析助手。
现在有 pandas.DataFrame df。
我会提供列信息和少量示例数据，以及用户问题。
请生成可直接执行的 Python 代码来回答问题。

约束：
1. 只输出代码，不要任何解释或 Markdown。
2. 不导入新库，不读写文件，不访问网络。
3. 使用已有的 df 变量。
4. 若结果为表格，存入 result_df；
   若结果为标量或简单结构，存入 result。
5. 不要 print()，不要 input()，不要修改原始 df（如需处理请基于副本）。
"""
            messages_code = [
                {"role": "system", "content": system_prompt_code},
                {
                    "role": "user",
                    "content": (
                        f"列信息：\n{schema_text}\n\n"
                        f"前5行示例（字符串，仅供参考）：\n"
                        f"{json.dumps(sample_rows, ensure_ascii=False)}\n\n"
                        f"用户问题：{question}"
                    ),
                },
            ]

            with st.spinner("正在分析数据..."):
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
                # 如要排查，可临时打印 code：
                # st.code(code, language="python")
                st.stop()

            if "result_df" in local_vars and isinstance(local_vars["result_df"], pd.DataFrame):
                result = local_vars["result_df"].head(20)
                is_table = True
            elif "result" in local_vars:
                result = local_vars["result"]
                is_table = False
            else:
                st.error("未获取到可用结果（缺少 result_df 或 result）。")
                st.stop()

            # 生成通用 Excel 公式示例
            system_prompt_excel = """
你是一个通用 Excel 公式助手。
根据列信息和问题，生成一些可以帮助用户在 Excel 中完成类似分析的公式示例。

输出 JSON：
{
  "explanation": "中文说明 1-3 句",
  "excel_formulas": ["公式1", "公式2", "公式3"]
}

要求：
1. 只基于列名本身，不依赖本地文件路径。
2. 使用常见函数（SUMIFS, AVERAGEIFS, IF, XLOOKUP 等）。
3. 列引用可使用整列绝对引用（如 $B:$B）或结构化引用。
4. 只输出 JSON，不要其它内容。
"""
            messages_excel = [
                {"role": "system", "content": system_prompt_excel},
                {
                    "role": "user",
                    "content": (
                        f"列信息：\n{schema_text}\n\n"
                        f"用户问题：{question}"
                    ),
                },
            ]

            explanation = ""
            excel_formulas = []
            with st.spinner("正在生成 Excel 公式示例..."):
                try:
                    raw = call_qwen(messages_excel, max_tokens=400, temperature=0)
                    try:
                        data = json.loads(raw)
                        explanation = data.get("explanation", "") or ""
                        excel_formulas = data.get("excel_formulas", []) or []
                    except json.JSONDecodeError:
                        explanation = raw
                except Exception as e:
                    explanation = f"（生成 Excel 公式说明失败：{e}）"

            st.markdown("### ✅ 分析结果")
            if explanation:
                st.markdown(f"**说明：** {explanation}")
            if is_table:
                st.dataframe(result, width="stretch")
            else:
                st.write(result)

            if excel_formulas:
                st.markdown("### 📎 Excel 公式示例")
                for f in excel_formulas:
                    st.code(f, language="excel")
            else:
                st.caption("（本次未生成可用的 Excel 公式示例。）")


# ================== Tab 2：数据对照 ==================

with tab_compare:
    st.markdown("### ✅ 数据对照（自然语言 / 手动配置 → 并排展示，不自动判定）")

    if df2 is None:
        st.info("要使用数据对照，请上传两个文件（表1 和 表2）。")
    else:
        st.markdown(
            "你可以直接用自然语言描述需求，例如：\n"
            "“根据 Item Number 和 规格型号 作为索引，输出两个表中的 开票型号 开票名称 开票数量 未税单价 未税金额 的对照信息。”\n"
            "系统会尝试解析索引列和字段，并基于自动映射找到表2对应列，输出并排表。"
        )

        nl = st.text_input(
            "自然语言描述（推荐）：",
            key="compare_nl",
        )

        st.markdown(
            "<div style='font-size:12px;color:#666;margin-top:4px;'>"
            "如解析不准确，可在下方手动指定索引列和字段（都用表1的列名）。"
            "</div>",
            unsafe_allow_html=True,
        )

        manual_keys = st.text_input(
            "手动索引列（表1列名，逗号分隔，可留空）：",
            key="compare_keys_manual",
        )
        manual_fields = st.text_input(
            "手动对照字段（表1列名，逗号分隔，可留空）：",
            key="compare_fields_manual",
        )

        go_view = st.button("生成对照视图", key="compare_go")

        if go_view:
            cols1 = list(df1.columns)

            # 1️⃣ 优先用自然语言解析
            keys, fields = [], []
            if nl.strip():
                keys, fields = parse_compare_request(nl, cols1)

            # 2️⃣ 自然语言解析为空时，用手动输入兜底
            if not keys and manual_keys.strip():
                raw = manual_keys.replace("，", ",")
                keys = [k.strip() for k in raw.split(",") if k.strip() and k.strip() in cols1]
            if not fields and manual_fields.strip():
                raw = manual_fields.replace("，", ",")
                fields = [f.strip() for f in raw.split(",") if f.strip() and f.strip() in cols1]

            # 3️⃣ 再兜一层：如果 fields 还是空且有自动映射，用所有映射字段（除索引）
            if not fields and auto_mapping:
                fields = [c for c in auto_mapping.keys() if c in cols1 and c not in keys]

            # 校验
            if not keys:
                st.error("没有识别到索引列。请在自然语言中说明，或在手动索引列中填写（使用表1列名）。")
            elif not fields:
                st.error("没有识别到需要对照的字段。请在自然语言或手动字段中列出（使用表1列名）。")
            else:
                # 去重
                keys = list(dict.fromkeys([k for k in keys if k in cols1]))
                fields = list(dict.fromkeys([f for f in fields if f in cols1]))

                if not keys:
                    st.error("索引列经过校验后为空，请检查列名是否存在于表1。")
                elif not fields:
                    st.error("对照字段经过校验后为空，请检查列名是否存在于表1。")
                else:
                    st.caption(f"索引列（表1）：{', '.join(keys)}")
                    st.caption(f"对照字段（表1）：{', '.join(fields)}")

                    # 4️⃣ 为索引列和对照字段寻找表2对应列（自动映射优先，其次同名）
                    if df2 is None:
                        st.error("未检测到第二个表。")
                    else:
                        mapped_keys = {}
                        key_missing = []
                        for k in keys:
                            cand = auto_mapping.get(k, k if k in df2.columns else None)
                            if cand and cand in df2.columns:
                                mapped_keys[k] = cand
                            else:
                                key_missing.append(k)

                        if key_missing:
                            st.error(
                                "以下索引列在表2中找不到对应列（自动映射和同名都失败）："
                                + ", ".join(key_missing)
                            )
                        else:
                            compare_pairs = []
                            skipped = []
                            for l in fields:
                                cand = auto_mapping.get(l, l if l in df2.columns else None)
                                if cand and cand in df2.columns:
                                    compare_pairs.append((l, cand))
                                else:
                                    skipped.append(l)

                            if not compare_pairs:
                                st.error(
                                    "选中的对照字段在表2中都找不到对应列，"
                                    "请检查列名或自动映射结果。"
                                )
                            else:
                                if skipped:
                                    st.caption(
                                        "以下字段未展示（在表2未找到对应列）："
                                        + ", ".join(skipped)
                                    )

                                left = df1.copy()
                                right = df2.copy()

                                # 索引列重命名（让表2 join 键与表1同名）
                                index_rename = {
                                    mapped_keys[k]: k
                                    for k in keys
                                    if mapped_keys[k] != k
                                }
                                right = right.rename(columns=index_rename)

                                # 值字段重命名（先统一为表1列名，merge 时自动加后缀）
                                value_rename = {
                                    r: l for (l, r) in compare_pairs if l != r
                                }
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

                                # 构造对照视图列顺序：索引 + 每个字段的 _表1/_表2
                                view_cols = list(keys)
                                for l, _r in compare_pairs:
                                    col1 = f"{l}_表1"
                                    col2 = f"{l}_表2"
                                    if col1 in both.columns:
                                        view_cols.append(col1)
                                    if col2 in both.columns:
                                        view_cols.append(col2)

                                st.markdown("### 📊 对照结果概览")
                                st.write(f"- 仅在表1中的键数量：{len(only_in_1)}")
                                st.write(f"- 仅在表2中的键数量：{len(only_in_2)}")
                                st.write(f"- 两表均存在的键数量：{len(both)}")

                                if len(only_in_1) > 0:
                                    st.markdown("**仅在表1中的样例键（前20行）：**")
                                    st.dataframe(only_in_1[keys].head(20), width="stretch")

                                if len(only_in_2) > 0:
                                    st.markdown("**仅在表2中的样例键（前20行）：**")
                                    st.dataframe(only_in_2[keys].head(20), width="stretch")

                                st.markdown("### 🔍 匹配键的字段并排视图（前200行）")
                                if len(both) == 0:
                                    st.write("没有匹配成功的记录。")
                                else:
                                    st.dataframe(both[view_cols].head(200), width="stretch")
                                    st.caption(
                                        "说明：系统不自动判断是否一致，只把两个表中对应字段并排展示，"
                                        "由你自行查看或导出后用 Excel 做更复杂比对。"
                                    )
