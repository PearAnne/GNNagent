"""图机器学习问题优化与求解智能体 — Streamlit 公网入口。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import streamlit as st

from agent import GNNAgent
from config import DEFAULT_MODEL, DEEPSEEK_API_KEY

st.set_page_config(
    page_title="图机器学习智能体",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        max-width: 1100px;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #2563eb 100%);
        border-radius: 16px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
        box-shadow: 0 10px 40px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.6;
    }
    .hero-tags {
        margin-top: 1rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .tag {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 999px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        color: #e2e8f0;
    }

    .step-box {
        background: #f8fafc;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        border-left: 3px solid #2563eb;
        font-size: 0.88rem;
        color: #334155;
    }

    .pipeline {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0 0 1.25rem 0;
    }
    .pipe-step {
        flex: 1;
        min-width: 130px;
        background: #f8fafc;
        border-radius: 10px;
        padding: 0.7rem 0.75rem;
        text-align: center;
        font-size: 0.78rem;
        color: #64748b;
        border: 1px solid #e2e8f0;
    }
    .pipe-step strong {
        display: block;
        color: #1e293b;
        font-size: 0.85rem;
        margin-bottom: 0.2rem;
    }

    .sidebar-brand {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.25rem;
    }
    .sidebar-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .feat-item {
        font-size: 0.88rem;
        color: #475569;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f5f9;
        line-height: 1.5;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        border: none !important;
        font-weight: 600;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


def _resolve_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    try:
        key = st.secrets.get("DEEPSEEK_API_KEY", key)
    except Exception:
        pass
    return key.strip()


with st.sidebar:
    st.markdown('<p class="sidebar-brand">GNN Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">图机器学习任务规划智能体</p>', unsafe_allow_html=True)

    with st.expander("高级选项", expanded=False):
        model = st.selectbox("模型", ["deepseek-chat", "deepseek-reasoner"], index=0)
        use_tool_calling = st.checkbox("Function Calling 模式", value=False)

    st.divider()
    st.markdown("**功能说明**")
    st.markdown(
        """
<div class="feat-item">① 将模糊需求优化为结构化图学习任务</div>
<div class="feat-item">② 自动推荐 GNN 模型并运行 baseline</div>
<div class="feat-item">③ 求解过程自动生成 PDF 报告</div>
""",
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("示例：引用网络节点分类、社交网络异常检测、链接预测")

st.markdown(
    """
<div class="hero">
  <h1>图机器学习问题优化与求解智能体</h1>
  <p>输入自然语言描述的图学习任务，智能体将完成问题优化、模型推荐、baseline 实验，并输出完整 PDF 过程报告。</p>
  <div class="hero-tags">
    <span class="tag">DeepSeek</span>
    <span class="tag">提示词工程</span>
    <span class="tag">工具调用</span>
    <span class="tag">PDF 报告</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="pipeline">
  <div class="pipe-step"><strong>1. 问题优化</strong>结构化任务规格</div>
  <div class="pipe-step"><strong>2. 模型推荐</strong>GNN 选型建议</div>
  <div class="pipe-step"><strong>3. Baseline</strong>节点分类实验</div>
  <div class="pipe-step"><strong>4. PDF</strong>完整求解文档</div>
</div>
""",
    unsafe_allow_html=True,
)

example = st.selectbox(
    "快速示例",
    [
        "（自定义）",
        "我有一篇论文的引用网络，大概 2700 个节点，节点有词袋特征，想预测论文类别（节点分类）。",
        "社交网络同质图，用户之间有关注关系，每个用户有发帖统计特征，想识别异常机器人账号。",
        "电商用户-商品二部图，想做链接预测，推荐用户可能购买的商品。",
    ],
    label_visibility="collapsed",
)

st.markdown("**描述你的图机器学习问题**")
default_text = example if example != "（自定义）" else ""
user_input = st.text_area(
    "问题描述",
    value=default_text,
    height=130,
    placeholder="例如：我有引用网络数据，节点约 3000 个，想做节点分类……",
    label_visibility="collapsed",
)

run_btn = st.button("启动智能体求解", type="primary", use_container_width=True)

if run_btn:
    if not user_input.strip():
        st.error("请输入问题描述。")
        st.stop()

    key = _resolve_api_key()
    if not key:
        st.error("服务配置异常，请联系管理员。")
        st.stop()

    output_dir = _ROOT / "outputs"
    steps: list[str] = []

    with st.status("智能体求解中…", expanded=True) as status:
        try:
            agent = GNNAgent(
                api_key=key,
                model=model,
                output_dir=output_dir,
            )

            log_area = st.empty()

            def tracked_step(phase: str, data: dict):
                steps.append(f"**{phase}**")
                log_area.markdown(
                    "\n".join(f'<div class="step-box">{s}</div>' for s in steps[-8:]),
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

    st.success("求解完成，以下为结构化结果与实验报告。")

    if state.experiment:
        exp = state.experiment
        m1, m2, m3 = st.columns(3)
        m1.metric("测试集准确率", f"{exp.get('test_accuracy', 0):.2%}")
        m2.metric("验证集准确率", f"{exp.get('val_accuracy', 0):.2%}")
        m3.metric("数据集", str(exp.get("dataset", "—")))

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 优化后问题")
        st.info(state.task_spec.get("optimized_question", "—"))
        with st.expander("任务规格详情", expanded=False):
            st.json(state.task_spec)
        with st.expander("模型推荐详情", expanded=False):
            st.json(state.recommendation)

    with col2:
        st.markdown("#### 实验与轨迹")
        if state.experiment:
            st.json(state.experiment)
        else:
            st.warning("未运行实验")
        with st.expander("工具调用轨迹", expanded=True):
            for item in state.tool_trace:
                st.markdown(f"- **{item.get('tool')}**: {item.get('summary')}")

    if state.pdf_path and state.pdf_path.exists():
        st.markdown("---")
        st.markdown("#### PDF 过程报告")
        pdf_bytes = state.pdf_path.read_bytes()
        st.download_button(
            label="下载 PDF 报告",
            data=pdf_bytes,
            file_name=state.pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
        )

    with st.expander("LLM 对话日志", expanded=False):
        st.json(state.messages_log)

else:
    st.info("选择示例或输入问题后，点击「启动智能体求解」。首次运行可能下载数据集，约需 1–2 分钟。")

st.caption("图机器学习领域通用问题优化智能体 · Powered by DeepSeek")
