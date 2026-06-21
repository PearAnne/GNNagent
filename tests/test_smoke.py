"""冒烟测试：不调用 DeepSeek API。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from gnn_runner import run_gnn_baseline
from pdf_report import generate_pdf_report
from tools import AgentState, execute_run_gnn_baseline


class TestSmoke(unittest.TestCase):
    def test_gnn_baseline(self):
        result = run_gnn_baseline(dataset_name="Cora", epochs=50)
        self.assertGreaterEqual(result.test_accuracy, 0.0)
        self.assertLessEqual(result.test_accuracy, 1.0)
        self.assertIn(result.dataset, ("Cora", "Citeseer", "Pubmed", "Synthetic(Cora)"))
        self.assertIn("node_classification", result.task_type)

    def test_pdf_generation(self):
        out = _ROOT / "outputs" / "test_report.pdf"
        state = AgentState(user_input="测试：Cora 节点分类")
        state.task_spec = {
            "optimized_question": "在 Cora 引用网络上进行节点分类",
            "task_type": "node_classification",
        }
        state.recommendation = {"primary_model": "GCN", "rationale": "同质小图"}
        execute_run_gnn_baseline(state, dataset_name="Cora", epochs=15)
        path = generate_pdf_report(
            output_path=out,
            user_input=state.user_input,
            task_spec=state.task_spec,
            recommendation=state.recommendation,
            experiment=state.experiment,
            tool_trace=state.tool_trace,
            executive_summary="本报告为冒烟测试生成的摘要。",
            conclusions="GCN 在 Cora 上可达到合理 baseline 准确率。",
        )
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
