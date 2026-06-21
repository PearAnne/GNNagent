#!/usr/bin/env python3
"""命令行运行图机器学习智能体。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "src"))

from agent import GNNAgent


def main():
    parser = argparse.ArgumentParser(description="图机器学习问题优化智能体")
    parser.add_argument("question", help="图机器学习问题描述（自然语言）")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--tool-calling", action="store_true")
    args = parser.parse_args()

    output_dir = _ROOT / "outputs"
    agent = GNNAgent(model=args.model, output_dir=output_dir)

    if args.tool_calling:
        state = agent.run_with_tool_calling(args.question)
    else:
        state = agent.run(args.question)

    print(json.dumps({
        "optimized_question": state.task_spec.get("optimized_question"),
        "test_accuracy": (state.experiment or {}).get("test_accuracy"),
        "pdf_path": str(state.pdf_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
