"""轻量工具定义与执行。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from gnn_runner import run_gnn_baseline
from pdf_report import generate_pdf_report


def _parse_json_from_llm(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


@dataclass
class AgentState:
    user_input: str
    task_spec: dict[str, Any] = field(default_factory=dict)
    recommendation: dict[str, Any] = field(default_factory=dict)
    experiment: dict[str, Any] | None = None
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    executive_summary: str = ""
    conclusions: str = ""
    pdf_path: Path | None = None
    messages_log: list[dict[str, Any]] = field(default_factory=list)


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "recommend_gnn",
            "description": "根据已优化的任务规格，推荐图神经网络模型与训练策略。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_spec_json": {
                        "type": "string",
                        "description": "结构化任务规格的 JSON 字符串",
                    },
                },
                "required": ["task_spec_json"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_gnn_baseline",
            "description": "在公开图数据集（Cora/Citeseer）上运行 GCN 节点分类 baseline 实验。",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "enum": ["Cora", "Citeseer", "Pubmed"],
                        "description": "Planetoid 数据集名称",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "模型名称（当前实现为 GCN）",
                    },
                    "epochs": {
                        "type": "integer",
                        "description": "训练轮数，默认 100",
                    },
                },
                "required": ["dataset_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pdf_report",
            "description": "将问题优化、推荐、实验与工具轨迹写入 PDF 报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "executive_summary": {
                        "type": "string",
                        "description": "执行摘要（2-4 段中文）",
                    },
                    "conclusions": {
                        "type": "string",
                        "description": "结论与建议（2-4 段中文）",
                    },
                },
                "required": ["executive_summary", "conclusions"],
            },
        },
    },
]


def execute_recommend_gnn(state: AgentState, task_spec_json: str) -> str:
    if not state.task_spec:
        try:
            state.task_spec = json.loads(task_spec_json)
        except json.JSONDecodeError:
            state.task_spec = _parse_json_from_llm(task_spec_json)
    result = {
        "status": "pending_llm",
        "task_spec": state.task_spec,
        "message": "任务规格已就绪，模型推荐由 LLM 在对话中完成。",
    }
    state.tool_trace.append({
        "tool": "recommend_gnn",
        "summary": "解析任务规格，准备模型推荐",
    })
    return json.dumps(result, ensure_ascii=False)


def execute_run_gnn_baseline(
    state: AgentState,
    dataset_name: str = "Cora",
    model_name: str = "GCN",
    epochs: int = 100,
) -> str:
    exp = run_gnn_baseline(
        dataset_name=dataset_name,
        model_name=model_name,
        epochs=epochs,
    )
    state.experiment = exp.to_dict()
    state.tool_trace.append({
        "tool": "run_gnn_baseline",
        "summary": (
            f"{exp.dataset} test_acc={exp.test_accuracy}, "
            f"val_acc={exp.val_accuracy}, {exp.duration_seconds}s"
        ),
        "args": {"dataset_name": dataset_name, "epochs": epochs},
    })
    return json.dumps(state.experiment, ensure_ascii=False)


def execute_generate_pdf_report(
    state: AgentState,
    executive_summary: str,
    conclusions: str,
    output_dir: Path,
) -> str:
    state.executive_summary = executive_summary
    state.conclusions = conclusions
    out = output_dir / f"gnn_agent_report_{Path(output_dir).name or 'run'}.pdf"
    # unique name with timestamp
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = output_dir / f"gnn_agent_report_{stamp}.pdf"

    state.pdf_path = generate_pdf_report(
        output_path=out,
        user_input=state.user_input,
        task_spec=state.task_spec,
        recommendation=state.recommendation,
        experiment=state.experiment,
        tool_trace=state.tool_trace,
        executive_summary=executive_summary,
        conclusions=conclusions,
    )
    state.tool_trace.append({
        "tool": "generate_pdf_report",
        "summary": f"PDF 已生成: {state.pdf_path.name}",
    })
    return json.dumps({"pdf_path": str(state.pdf_path)}, ensure_ascii=False)


TOOL_EXECUTORS: dict[str, Callable[..., str]] = {
    "recommend_gnn": execute_recommend_gnn,
    "run_gnn_baseline": execute_run_gnn_baseline,
}
