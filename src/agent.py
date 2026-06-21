"""DeepSeek API 驱动的图机器学习智能体主循环。"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from prompts import (
    OPTIMIZE_QUESTION_PROMPT,
    RECOMMEND_GNN_PROMPT,
    REPORT_SUMMARY_PROMPT,
    SYSTEM_PROMPT,
)
from tools import (
    AgentState,
    TOOL_DEFINITIONS,
    _parse_json_from_llm,
    execute_generate_pdf_report,
    execute_recommend_gnn,
    execute_run_gnn_baseline,
)


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_BASE_URL = "https://api.deepseek.com"


class GNNAgent:
    """问题优化 + 工具调用 + PDF 报告的智能体。"""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        output_dir: Path | None = None,
        on_step: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "未设置 DEEPSEEK_API_KEY。请在环境变量或 Streamlit Secrets 中配置。"
            )
        self.client = OpenAI(api_key=key, base_url=base_url or DEFAULT_BASE_URL)
        self.model = model
        self.output_dir = Path(output_dir or "outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.on_step = on_step

    def _emit(self, phase: str, data: dict[str, Any]):
        if self.on_step:
            self.on_step(phase, data)

    def _chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    def optimize_question(self, state: AgentState) -> dict[str, Any]:
        prompt = OPTIMIZE_QUESTION_PROMPT.format(user_input=state.user_input)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._chat(messages)
        state.messages_log.append({"phase": "optimize_question", "response": text})
        spec = _parse_json_from_llm(text)
        state.task_spec = spec
        self._emit("optimize_question", {"task_spec": spec, "raw": text})
        return spec

    def recommend_gnn(self, state: AgentState) -> dict[str, Any]:
        prompt = RECOMMEND_GNN_PROMPT.format(
            task_spec=json.dumps(state.task_spec, ensure_ascii=False, indent=2),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self._chat(messages)
        state.messages_log.append({"phase": "recommend_gnn", "response": text})
        rec = _parse_json_from_llm(text)
        state.recommendation = rec
        state.tool_trace.append({
            "tool": "recommend_gnn",
            "summary": f"推荐主模型: {rec.get('primary_model', '?')}",
        })
        self._emit("recommend_gnn", {"recommendation": rec, "raw": text})
        return rec

    def run_baseline(self, state: AgentState) -> dict[str, Any]:
        dataset = "Cora"
        baseline_ds = state.recommendation.get("baseline_dataset", "")
        if isinstance(baseline_ds, str):
            for name in ("Cora", "Citeseer", "Pubmed"):
                if name.lower() in baseline_ds.lower():
                    dataset = name
                    break
        model = state.recommendation.get("primary_model", "GCN")
        if not isinstance(model, str):
            model = "GCN"
        result_json = execute_run_gnn_baseline(
            state,
            dataset_name=dataset,
            model_name=model.split()[0] if model else "GCN",
            epochs=80,
        )
        exp = json.loads(result_json)
        self._emit("run_baseline", {"experiment": exp})
        return exp

    def write_report_sections(self, state: AgentState) -> tuple[str, str]:
        prompt = REPORT_SUMMARY_PROMPT.format(
            user_input=state.user_input,
            optimized_question=state.task_spec.get("optimized_question", ""),
            task_spec=json.dumps(state.task_spec, ensure_ascii=False),
            recommendation=json.dumps(state.recommendation, ensure_ascii=False),
            experiment_result=json.dumps(state.experiment or {}, ensure_ascii=False),
            tool_trace=json.dumps(state.tool_trace, ensure_ascii=False),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt + "\n\n请用以下 JSON 格式回复：{\"executive_summary\": \"...\", \"conclusions\": \"...\"}"},
        ]
        text = self._chat(messages)
        state.messages_log.append({"phase": "report_summary", "response": text})
        parsed = _parse_json_from_llm(text)
        summary = parsed.get("executive_summary", text[:800])
        conclusions = parsed.get("conclusions", "")
        return summary, conclusions

    def generate_pdf(self, state: AgentState, summary: str, conclusions: str) -> Path:
        result_json = execute_generate_pdf_report(
            state,
            executive_summary=summary,
            conclusions=conclusions,
            output_dir=self.output_dir,
        )
        path = Path(json.loads(result_json)["pdf_path"])
        self._emit("generate_pdf", {"pdf_path": str(path)})
        return path

    def run(self, user_input: str) -> AgentState:
        """完整求解流水线。"""
        state = AgentState(user_input=user_input)

        self._emit("start", {"user_input": user_input})

        self.optimize_question(state)
        self.recommend_gnn(state)
        self.run_baseline(state)
        summary, conclusions = self.write_report_sections(state)
        self.generate_pdf(state, summary, conclusions)

        self._emit("done", {
            "pdf_path": str(state.pdf_path),
            "test_accuracy": (state.experiment or {}).get("test_accuracy"),
        })
        return state

    def run_with_tool_calling(self, user_input: str) -> AgentState:
        """可选：使用 DeepSeek function calling 的交互式求解。"""
        state = AgentState(user_input=user_input)
        self.optimize_question(state)
        self.recommend_gnn(state)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"用户问题：{user_input}\n\n"
                    f"已优化任务规格：{json.dumps(state.task_spec, ensure_ascii=False)}\n"
                    f"已推荐模型：{json.dumps(state.recommendation, ensure_ascii=False)}\n"
                    "请调用 run_gnn_baseline 运行实验，完成后调用 generate_pdf_report 生成报告。"
                ),
            },
        ]

        for _ in range(6):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break

            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")

                if name == "run_gnn_baseline":
                    out = execute_run_gnn_baseline(state, **args)
                elif name == "recommend_gnn":
                    out = execute_recommend_gnn(
                        state, args.get("task_spec_json", "{}"),
                    )
                elif name == "generate_pdf_report":
                    out = execute_generate_pdf_report(
                        state,
                        executive_summary=args.get("executive_summary", ""),
                        conclusions=args.get("conclusions", ""),
                        output_dir=self.output_dir,
                    )
                else:
                    out = json.dumps({"error": f"unknown tool {name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": out,
                })
                self._emit("tool_call", {"tool": name, "result": out[:500]})

            if state.pdf_path:
                break

        if not state.experiment:
            self.run_baseline(state)
        if not state.pdf_path:
            summary, conclusions = self.write_report_sections(state)
            self.generate_pdf(state, summary, conclusions)

        return state
