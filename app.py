import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

# ========== 配置 Qwen3-Max via DashScope ==========

# 推荐使用阿里云 DashScope 的 OpenAI 兼容接口：
# 中国区: https://dashscope.aliyuncs.com/compatible-mode/v1
# 国际区: https://dashscope-intl.aliyuncs.com/compatible-mode/v1

api_key = (
    os.getenv("DASHSCOPE_API_KEY")
    or (st.secrets.get("DASHSCOPE_API_KEY") if hasattr(st, "secrets") else None)
)

if not api_key:
    st.error("请配置 DASHSCOPE_API_KEY（环境变量或 .streamlit/secrets.toml）")
    st.stop()

# 按你实际所在区域选择 base_url
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60.0,  # 提高超时时间，避免默认太短
)

MODEL_NAME = "qwen3-max"

# ========== 一个小工具：带重试的 Qwen 调用 ==========

def call_qwen(messages, max_tokens=8000, temperature=0.1):
    """
    通用的 Qwen chat.completions 封装：
    - 限制 max_tokens，避免输出过长导致超时
    - 简单重试一次
    - 只返回 content 字符串，出错抛异常
    """
    last_err = None
    for _ in range(2):  # 最多尝试2次
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            last_err = e
            # 如果是超时，再重试一次；否则直接抛
            if "timed out" in err.lower():
                continue
            else:
                break
    # 重试后仍失败，抛出最后一次的错误
    raise last_err


# ========== Streamlit 页面配置 ==========

st.set_page_config(
    page_title="智能数据查询助手",
    page_icon="📊",
    layout="centered",
)

st.title("📊 智能数据查询助手（Qwen3-Max）")
st.caption("上传 Excel/CSV → 用自然语言提问 → 返回结果预览 + Excel 公式示例（免登录，对外可用）")

# ========== 状态初始化 ==========

if "df" not in st.session_state:
    st.session_state.df = None
if "schema_text" not in st.session_state:
    st.session_state.schema_text = ""
if "columns" not in st.session_state:
    st.session_state.columns = []


# ========== 上传数据 ==========

uploaded_file = st.file_uploader(
    "上传数据文件（支持 .xlsx / .xls / .csv）", type=["xlsx", "xls", "csv"]
)

if uploaded_file is not None:
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    if df.empty:
        st.error("文件为空或无法解析为有效表格")
        st.stop()

    st.session_state.df = df
    st.session_state.columns = list(df.columns)
    schema_lines = [f"{col} ({str(df[col].dtype)})" for col in df.columns]
    st.session_state.schema_text = "\n".join(schema_lines)

if st.session_state.df is not None:
    with st.expander("当前数据集预览（前 10 行）", expanded=True):
        st.write(f"字段：{', '.join(st.session_state.columns)}")
        st.dataframe(st.session_state.df.head(10), width="stretch")
else:
    st.info("请先上传一个 Excel / CSV 文件。")
    st.stop()


# ========== 提问区域 ==========

st.markdown("### ✏️ 提问")
default_example = "例如：按渠道汇总本月 GMV 排名前 5 的渠道；或：给我每行的毛利率 Excel 公式"
question = st.text_input("请输入你的问题：", placeholder=default_example)
go = st.button("生成结果和 Excel 公式")

if not go:
    st.stop()

q = (question or "").strip()
if not q:
    st.warning("问题不能为空。")
    st.stop()

df = st.session_state.df
schema_text = st.session_state.schema_text
sample_rows = df.head(5).to_dict(orient="records")

# ========== 1. 用 Qwen3-Max 生成 pandas 分析逻辑 ==========

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
8. 代码尽量简短，避免无用注释。
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

# ========== 2. 执行生成的代码（后端本地执行） ==========

local_vars = {"df": df.copy()}
try:
    exec(code, {}, local_vars)
except Exception as e:
    st.error(f"执行分析逻辑出错：{e}")
    # 如果要排查，可以临时打开下一行查看生成代码
    # st.code(code, language="python")
    st.stop()

preview = None
is_table = False

if "result_df" in local_vars and isinstance(local_vars["result_df"], pd.DataFrame):
    result_df = local_vars["result_df"]
    preview = result_df.head(20)
    is_table = True
elif "result" in local_vars:
    val = local_vars["result"]
    preview = val
else:
    st.error("未获取到可用结果（缺少 result_df 或 result）。")
    # st.code(code, language="python")
    st.stop()

# ========== 3. 用 Qwen3-Max 生成说明 + Excel 公式示例 ==========

system_prompt_excel = """
你是一个精简的 Excel 公式助手。
现在给你字段信息和用户的问题，请输出严格 JSON：
{
  "explanation": "用中文解释大致的统计/计算逻辑，1-3 句话",
  "excel_formulas": ["公式1", "公式2", "公式3"]
}

要求：
1. 使用给定字段名，假设数据在一张表中，第一行是表头。
2. 公式使用通用形式，例如：=SUMIFS(...), =AVERAGEIFS(...), =IF(...), =VLOOKUP(...), =XLOOKUP(...), =SUMPRODUCT(...)。
3. 使用列绝对引用形式，如 $B:$B。
4. 不要返回注释、不返回 Markdown、不返回多余文本。
只返回一个合法的 JSON 对象。
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

with st.spinner("正在用 Qwen3-Max 生成 Excel 公式示例..."):
    try:
        raw = call_qwen(messages_excel, max_tokens=400, temperature=0)
        try:
            j = json.loads(raw)
            explanation = j.get("explanation", "") or ""
            excel_formulas = j.get("excel_formulas", []) or []
        except json.JSONDecodeError:
            # 模型没严格按 JSON 来，就当成说明文本用
            explanation = raw
            excel_formulas = []
    except Exception as e:
        explanation = f"（生成 Excel 公式说明失败：{e}）"
        excel_formulas = []

# ========== 4. 展示结果（不展示 Python 代码） ==========

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
    st.caption("（本次未生成可用的 Excel 公式示例，可能问题更偏解释类。）")
