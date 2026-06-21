# VLM / LLM 全流程材料

作业要求：整理 **完整 prompt、对话记录、结构化输出**。

## 模块划分

| 模块 | 文件 | 内容 |
|------|------|------|
| 提示词设计 | `prompts.md` | 各阶段 prompt 模板 |
| 问题优化 | `01_optimize_question.md` | DeepSeek 对话：模糊问题 → 结构化 JSON |
| 模型推荐 | `02_recommend_gnn.md` | DeepSeek 对话：任务规格 → GNN 推荐 |
| 工具调用 | `03_tool_calls.md` | baseline 与 PDF 工具轨迹 |
| 报告撰写 | `04_report_summary.md` | 执行摘要与结论生成对话 |

## 提交前

1. 用真实 DeepSeek API 跑通一次完整流程
2. 将 `app.py` 中 expander「LLM 对话日志」导出，或从 `outputs/` 对应 run 复制 `messages_log`
3. 替换本目录下占位文件中的 `User` / `Assistant` 示例
