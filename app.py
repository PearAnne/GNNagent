"""图机器学习问题优化与求解智能体 — Streamlit 公网入口。

运行:
    cd homework3/gnn_agent
    export DEEPSEEK_API_KEY=sk-...
    streamlit run app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from agent import GNNAgent

st.set_page_config(
    page_title="图机器学习智能体",
    page_icon="🕸️",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #5a6a7a;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .step-box {
        background: #f0f4f8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #3b82f6;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🕸️ 图机器学习问题优化与求解智能体</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">基于 DeepSeek API · 提示词工程 + 轻量工具调用 · 自动输出 PDF 报告</p>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("配置")
    import os as _os
    default_key = _os.environ.get("DEEPSEEK_API_KEY", "")
    try:
        default_key = st.secrets.get("DEEPSEEK_API_KEY", default_key)
    except Exception:
        pass
    api_key = st.text_input(
        "DeepSeek API Key",
        value=default_key,
        type="password",
        help="也可设置环境变量 DEEPSEEK_API_KEY",
    )
    model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"], index=0)
    use_tool_calling = st.checkbox("使用 Function Calling 模式", value=False)
    st.divider()
    st.markdown(
        """
**作业说明**
1. 智能体可公开部署（Streamlit Cloud）
2. 解决图机器学习节点分类规划问题
3. 求解过程自动生成 PDF

**示例输入**
- 我有论文引用网络，约 3000 节点，想做节点分类
- 社交网络用户关系图，检测异常账号，同质图、有节点特征
        """
    )

example = st.selectbox(
    "快速示例",
    [
        "（自定义）",
        "我有一篇论文的引用网络，大概 2700 个节点，节点有词袋特征，想预测论文类别（节点分类）。",
        "社交网络同质图，用户之间有关注关系，每个用户有发帖统计特征，想识别异常机器人账号。",
        "电商用户-商品二部图，想做链接预测，推荐用户可能购买的商品。",
    ],
)

default_text = ""
if example != "（自定义）":
    default_text = example

user_input = st.text_area(
    "描述你的图机器学习问题（可模糊）",
    value=default_text,
    height=140,
    placeholder="例如：我有引用网络数据，想做节点分类……",
)

run_btn = st.button("🚀 启动智能体求解", type="primary", use_container_width=True)

if run_btn:
    if not user_input.strip():
        st.error("请输入问题描述。")
        st.stop()

    key = api_key.strip() or __import__("os").environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        st.error("请配置 DeepSeek API Key（侧边栏或环境变量 DEEPSEEK_API_KEY）。")
        st.stop()

    output_dir = _ROOT / "outputs"
    steps: list[str] = []

    def on_step(phase: str, data: dict):
        steps.append(f"{phase}: {json.dumps(data, ensure_ascii=False)[:200]}")

    with st.status("智能体求解中…", expanded=True) as status:
        try:
            agent = GNNAgent(
                api_key=key,
                model=model,
                output_dir=output_dir,
                on_step=lambda p, d: steps.append(f"**{p}**"),
            )

            progress = st.empty()
            log_area = st.empty()

            def tracked_step(phase: str, data: dict):
                on_step(phase, data)
                log_area.markdown(
                    "\n".join(f'<div class="step-box">{s}</div>' for s in steps[-8:])
                    ,
                    unsafe_allow_html=True,
                )

            agent.on_step = tracked_step

            if use_tool_calling:
                state = agent.run_with_tool_calling(user_input.strip())
            else:
                state = agent.run(user_input.strip())

            status.update(label="求解完成", state="complete", expanded=False)
        except Exception as e:
            status.update(label="求解失败", state="error")
            st.exception(e)
            st.stop()

    st.success("智能体已完成求解！")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("优化后问题")
        st.info(state.task_spec.get("optimized_question", "—"))

        st.subheader("任务规格")
        st.json(state.task_spec)

        st.subheader("模型推荐")
        st.json(state.recommendation)

    with col2:
        st.subheader("实验结果")
        if state.experiment:
            exp = state.experiment
            st.metric("测试集准确率", f"{exp.get('test_accuracy', 0):.2%}")
            st.metric("验证集准确率", f"{exp.get('val_accuracy', 0):.2%}")
            st.metric("数据集", exp.get("dataset", "—"))
            st.json(state.experiment)
        else:
            st.warning("未运行实验")

    st.subheader("工具调用轨迹")
    for item in state.tool_trace:
        st.markdown(f"- **{item.get('tool')}**: {item.get('summary')}")

    if state.pdf_path and state.pdf_path.exists():
        st.subheader("PDF 报告")
        pdf_bytes = state.pdf_path.read_bytes()
        st.download_button(
            label="📄 下载 PDF 报告",
            data=pdf_bytes,
            file_name=state.pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
        with open(state.pdf_path, "rb") as f:
            st.pdf(f)

    with st.expander("LLM 对话日志（提示词工程材料）"):
        st.json(state.messages_log)

else:
    st.info("输入问题后点击「启动智能体求解」。首次运行 GNN baseline 会下载 Cora 数据集（约数 MB）。")

st.caption("Homework3 · 图机器学习领域通用问题优化智能体 · DeepSeek API")
