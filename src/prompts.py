"""提示词工程：问题优化、规划与报告撰写。"""

from __future__ import annotations

SYSTEM_PROMPT = """你是「图机器学习任务规划智能体」，专长于将模糊的自然语言描述转化为可执行的图学习任务规格，并协调工具完成 baseline 实验与报告生成。

工作原则：
1. 先优化/结构化用户问题，再调用工具求解，不要跳过问题优化步骤。
2. 推荐模型时说明理由（同质/异质、规模、任务类型、是否有边特征等）。
3. 数值结果必须来自工具返回值，禁止编造准确率等指标。
4. 全程使用中文，术语可保留英文（如 GCN、node classification）。
5. 完成实验后必须调用 generate_pdf_report 生成 PDF 报告。"""

OPTIMIZE_QUESTION_PROMPT = """用户用自然语言描述了一个图机器学习相关需求（可能模糊或不完整）。请将其优化为结构化任务规格。

用户原始描述：
{user_input}

请输出 JSON（不要 markdown 代码块），字段如下：
{{
  "task_type": "node_classification | link_prediction | graph_classification | anomaly_detection | recommendation | other",
  "graph_type": "homogeneous | heterogeneous | dynamic | unknown",
  "estimated_scale": "节点/边规模估计或 unknown",
  "features": "节点/边特征描述",
  "labels_or_target": "标签或预测目标",
  "constraints": "算力、延迟、可解释性等约束",
  "evaluation_metrics": ["accuracy", "f1", ...],
  "optimized_question": "一句精炼、可执行的问题陈述（中文）",
  "clarifying_assumptions": ["为补全信息所做的假设"],
  "missing_info": ["仍缺失但重要的信息"]
}}"""

RECOMMEND_GNN_PROMPT = """根据以下结构化任务规格，推荐 1–3 个图神经网络模型及训练策略。

任务规格（JSON）：
{task_spec}

请输出 JSON（不要 markdown 代码块）：
{{
  "primary_model": "模型名",
  "alternatives": ["备选模型"],
  "rationale": "推荐理由（结合图类型、任务、规模）",
  "training_hints": ["学习率、层数、采样等建议"],
  "baseline_dataset": "若用户未提供数据，建议用哪个公开数据集做演示（如 Cora、Citeseer）"
}}"""

PLANNER_PROMPT = """当前任务规格：
{task_spec}

模型推荐：
{recommendation}

请决定下一步应调用的工具。可用工具：
- recommend_gnn：根据任务规格生成模型推荐（若尚未推荐则调用）
- run_gnn_baseline：在公开数据集上运行 GNN baseline 实验
- generate_pdf_report：生成完整 PDF 过程报告（实验完成后必须调用）

若实验尚未运行，应先 run_gnn_baseline；若实验已完成且尚未生成 PDF，应 generate_pdf_report。
回复一个词：recommend_gnn | run_gnn_baseline | generate_pdf_report | done"""

REPORT_SUMMARY_PROMPT = """请根据以下智能体求解全过程，撰写报告「执行摘要」与「结论建议」（各 2–4 段，中文）。

用户原始输入：{user_input}

优化后问题：{optimized_question}

任务规格：{task_spec}

模型推荐：{recommendation}

实验结果：{experiment_result}

工具调用轨迹：{tool_trace}
"""
