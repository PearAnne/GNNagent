# Homework3 · 图机器学习问题优化与求解智能体

基于 **DeepSeek API**（公开大模型）+ **提示词工程** + **轻量工具调用**，将用户模糊的图机器学习需求优化为结构化任务，自动推荐 GNN 模型、运行 baseline 实验，并生成 **PDF 过程报告**。

## 作业要求对应

| 要求 | 实现 |
|------|------|
| 公开可访问大模型 API | DeepSeek API（`deepseek-chat`） |
| 智能体可公开访问 | Streamlit Web 应用，可部署至 [Streamlit Community Cloud](https://streamlit.io/cloud) |
| 解决一个问题 | **图学习任务规划 + 节点分类 baseline 求解** |
| 自动输出详细 PDF | `generate_pdf_report` 工具，含 8 个章节 |

## 环境要求

- Python **3.10+**
- DeepSeek API Key（[platform.deepseek.com](https://platform.deepseek.com)）

## 安装与运行

```bash
cd homework3/gnn_agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-full.txt

export DEEPSEEK_API_KEY=sk-your-key-here
streamlit run app.py
```

浏览器打开 `http://localhost:8501`，输入图机器学习问题描述，点击「启动智能体求解」。

## 命令行快速体验（无需 Streamlit）

```bash
export DEEPSEEK_API_KEY=sk-...
python run_agent.py "我有论文引用网络约2700节点，想做节点分类"
```

生成的 PDF 在 `outputs/` 目录。

## 公网部署

详细步骤见 **[DEPLOY.md](DEPLOY.md)**（自有服务器 systemd + Nginx、Streamlit Cloud、Docker）。

**最快公网（作业推荐）**：GitHub + [Streamlit Cloud](https://share.streamlit.io)  
→ Main file：`app.py`（仓库 [GNNagent](https://github.com/PearAnne/GNNagent) 根目录）  
→ Secrets：`DEEPSEEK_API_KEY = "sk-..."`  
→ 详见 **[DEPLOY.md](DEPLOY.md)** 方式二（云端用轻量 `requirements.txt`，无需 PyTorch）

**自有服务器（推荐，PyTorch/PyG 更稳）**：

```bash
export DEEPSEEK_API_KEY=sk-...
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy  # 无代理时
streamlit run app.py   # 已配置监听 0.0.0.0:8501
```

他人访问 `http://<服务器公网IP>:8501`；生产环境建议按 DEPLOY.md 配置 Nginx + HTTPS。

## 智能体架构

```
用户模糊描述
    → [Prompt] 问题优化器 → 结构化任务 JSON
    → [Prompt] GNN 推荐器 → 模型与训练建议
    → [Tool] run_gnn_baseline → Cora/Citeseer 上 GCN 实验
    → [Prompt] 报告摘要生成
    → [Tool] generate_pdf_report → PDF 下载
```

### 轻量工具

| 工具 | 说明 |
|------|------|
| `recommend_gnn` | 根据任务规格推荐 GCN/GAT/GraphSAGE 等 |
| `run_gnn_baseline` | PyTorch Geometric 两层 GCN 节点分类 |
| `generate_pdf_report` | 8 章 PDF（优化过程 + 实验 + 轨迹） |

## 目录结构

```
gnn_agent/
  app.py              # Streamlit 公网入口
  run_agent.py        # CLI 入口
  src/
    agent.py          # DeepSeek 智能体主循环
    prompts.py        # 提示词工程
    tools.py          # 工具定义与状态
    gnn_runner.py     # GNN baseline
    pdf_report.py     # PDF 生成
  data/
    example_inputs.json
  llm_logs/           # 提示词与对话材料
  outputs/            # 生成的 PDF
  tests/
    test_smoke.py
  requirements.txt
  README.md
```

## 提示词工程材料

见 [`llm_logs/README.md`](llm_logs/README.md) 与 [`llm_logs/prompts.md`](llm_logs/prompts.md)。正式提交前请补充真实 DeepSeek 对话导出。

## 单元测试（无需 API Key）

```bash
python -m unittest tests/test_smoke.py -v
```

测试 GNN baseline 与 PDF 生成（不调用 DeepSeek）。

## 学术诚信

- 本仓库 `llm_logs/` 含提示词模板；请替换为你的真实 DeepSeek 对话记录。
- 实验数值来自本地 `run_gnn_baseline`，勿编造准确率。
