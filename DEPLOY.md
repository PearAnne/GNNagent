# 部署指南：让其他人公网访问智能体

两种常用方式：**自有服务器（推荐，PyTorch/PyG 更稳）** 和 **Streamlit Cloud（免运维，但依赖较重可能失败）**。

---

## 方式一：自有服务器 / 云主机（推荐）

适用：阿里云、腾讯云、华为云、学校服务器等 Linux 主机（建议 **2GB+ 内存**，CPU 即可）。

### 1. 上传代码

```bash
# 在服务器上
git clone <你的仓库地址>
cd algorithm_homework/homework3/gnn_agent
```

或本地上传 `homework3/gnn_agent` 整个目录。

### 2. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行会下载 Cora 数据集（约数 MB），需能访问 GitHub。

### 3. 配置 API Key（不要写进代码）

```bash
# 写入仅 root/本人可读的文件
sudo mkdir -p /etc/gnn-agent
echo 'DEEPSEEK_API_KEY=sk-你的密钥' | sudo tee /etc/gnn-agent/env
sudo chmod 600 /etc/gnn-agent/env
```

若服务器**不需要**走本地代理访问 DeepSeek，在 env 里可加：

```bash
HTTP_PROXY=
HTTPS_PROXY=
ALL_PROXY=
```

### 4. 用 systemd 常驻运行

```bash
# 修改 deploy/gnn-agent.service 里的 User、WorkingDirectory 路径后：
sudo cp deploy/gnn-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gnn-agent
sudo systemctl start gnn-agent
sudo systemctl status gnn-agent
```

默认监听 **8501**。局域网访问：`http://服务器IP:8501`。

### 5. Nginx 反向代理 + HTTPS（公网标准做法）

```bash
sudo apt install nginx certbot python3-certbot-nginx   # Debian/Ubuntu
sudo cp deploy/nginx.conf.example /etc/nginx/sites-available/gnn-agent
# 编辑其中的域名 your-domain.example.com
sudo ln -s /etc/nginx/sites-available/gnn-agent /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d your-domain.example.com
```

他人访问：`https://your-domain.example.com`。

### 6. 防火墙

```bash
# 若用 Nginx，只开放 80/443
sudo ufw allow 80
sudo ufw allow 443
# 若暂时不用 Nginx，临时开放 8501
sudo ufw allow 8501
```

### 7. 安全建议（对外公开时很重要）

| 项 | 建议 |
|----|------|
| API Key | 只放服务器环境变量，用 Secrets 预填，**不要让访客各自填 Key**（会泄露你的额度） |
| 访问控制 | 作业演示可加 Nginx Basic Auth，或仅校内 IP 白名单 |
| 限流 | 单次求解约 1–2 分钟，避免被刷爆 API 费用 |

若要让访客**不用填 Key**，需改 `app.py`：仅从 `st.secrets` / 环境变量读 Key，侧边栏隐藏 Key 输入（部署前可再改一版）。

---

## 方式二：Streamlit Community Cloud（**作业推荐，免买服务器**）

满足「智能体可公开访问」。本项目 `requirements.txt` **已针对云端优化**（不含 PyTorch），会自动使用轻量图 baseline；本地若要真实 GCN 请用 `requirements-full.txt`。

### 步骤

1. **推送 GitHub**  
   确保仓库包含 `homework3/gnn_agent/` 下全部文件（`app.py`、`requirements.txt`、`src/` 等）。

2. **登录** [share.streamlit.io](https://share.streamlit.io)（用 GitHub 授权）。

3. **New app** 填写：
   | 项 | 值 |
   |----|-----|
   | Repository | 你的仓库 |
   | Branch | `main`（或你的分支） |
   | **Main file path** | `app.py`（若仓库根目录即本工程） |

4. **Advanced settings → Secrets**（TOML，仅填 Key，不要提交到 Git）：

```toml
DEEPSEEK_API_KEY = "sk-你的密钥"
```

5. 点击 **Deploy**，等待构建（约 2–5 分钟）。成功后地址形如：

   `https://<你的名字>-<仓库名>-homework3-gnn-agent-app-xxxxx.streamlit.app`

6. **验证**：打开链接 → 输入图 ML 问题 → 求解 → 能下载 PDF。

### 常见问题

| 问题 | 处理 |
|------|------|
| 构建失败 | 确认 Main file path 正确；查看 Cloud 日志 |
| API 连接失败 | DeepSeek 不需你本机代理；Key 是否正确 |
| 访客要填 Key | 正常：Secrets 已配置则侧边栏可留空；也可隐藏 Key 输入框 |
| 和本地结果不一致 | 云端为轻量 backend（合成图 + 1-hop 聚合），本地 full 为 Cora+GCN |

### 作业提交写法示例

> 智能体公网地址：`https://xxx.streamlit.app`  
> 技术栈：Streamlit + DeepSeek API + 自动 PDF 报告

---

## 方式三：Docker（可选）

```bash
cd homework3/gnn_agent
docker build -t gnn-agent .
docker run -d --name gnn-agent \
  -p 8501:8501 \
  -e DEEPSEEK_API_KEY=sk-你的密钥 \
  -v gnn_outputs:/app/outputs \
  gnn-agent
```

访问 `http://服务器IP:8501`。

---

## 验证部署成功

1. 浏览器打开公网 URL。
2. 输入示例问题，点击「启动智能体求解」。
3. 约 1–2 分钟后应出现准确率与 **PDF 下载** 按钮。
4. 查看日志：`journalctl -u gnn-agent -f`（systemd）或 `docker logs gnn-agent`。

---

## 作业提交时可写

- **公网地址**：`https://your-domain.example.com` 或 Streamlit Cloud URL  
- **说明**：Streamlit H5 + DeepSeek API + 自动 PDF 报告  
- **复现**：README 中的 `requirements.txt` 与部署文档
