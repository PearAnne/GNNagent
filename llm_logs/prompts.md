# 提示词工程清单

与 `src/prompts.py` 同步。正式材料请附 DeepSeek 原始回复。

## 1. 系统提示词（SYSTEM_PROMPT）

```
你是「图机器学习任务规划智能体」……
（见 src/prompts.py）
```

**设计要点**：强调先优化问题再工具求解；禁止编造实验数值；必须生成 PDF。

## 2. 问题优化（OPTIMIZE_QUESTION_PROMPT）

**输入**：用户自然语言  
**输出**：JSON 任务规格（task_type, graph_type, optimized_question, …）

## 3. 模型推荐（RECOMMEND_GNN_PROMPT）

**输入**：结构化任务 JSON  
**输出**：primary_model, alternatives, rationale, baseline_dataset

## 4. 报告摘要（REPORT_SUMMARY_PROMPT）

**输入**：全流程上下文  
**输出**：executive_summary + conclusions（JSON）

## 5. Function Calling 工具 schema

见 `src/tools.py` 中 `TOOL_DEFINITIONS`：
- `recommend_gnn`
- `run_gnn_baseline`
- `generate_pdf_report`
