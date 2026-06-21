"""在公开图数据集或轻量 fallback 上运行节点分类 baseline。"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    task_type: str
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float
    epochs: int
    hidden_dim: int
    num_nodes: int
    num_edges: int
    num_features: int
    num_classes: int
    duration_seconds: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _use_light_backend() -> bool:
    flag = os.environ.get("GNN_AGENT_LIGHT", "").lower()
    if flag in ("1", "true", "yes"):
        return True
    try:
        import torch_geometric  # noqa: F401
        return False
    except ImportError:
        return True


def _make_synthetic_numpy(
    num_nodes: int = 400,
    num_features: int = 32,
    num_classes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((num_nodes, num_features))
    y = rng.integers(0, num_classes, size=num_nodes)

    src = np.arange(num_nodes)
    dst = (src + 1) % num_nodes
    extra_src = rng.integers(0, num_nodes, size=num_nodes * 2)
    extra_dst = rng.integers(0, num_nodes, size=num_nodes * 2)
    edges = np.stack([
        np.concatenate([src, extra_src]),
        np.concatenate([dst, extra_dst]),
    ], axis=0)

    perm = rng.permutation(num_nodes)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[perm[:int(0.6 * num_nodes)]] = True
    val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
    test_mask[perm[int(0.8 * num_nodes):]] = True
    return x, y, edges, train_mask, val_mask, test_mask


def _accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _run_light_baseline(
    dataset_name: str,
    model_name: str,
    epochs: int,
    hidden_dim: int,
    seed: int,
    start: float,
) -> ExperimentResult:
    """Streamlit Cloud 友好：1-hop 邻域聚合 + LogisticRegression（无需 PyTorch）。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    name = dataset_name.capitalize()
    x, y, edges, train_mask, val_mask, test_mask = _make_synthetic_numpy(seed=seed)
    num_nodes, num_features = x.shape
    num_classes = int(y.max()) + 1

    # 邻接矩阵行归一化后做 1-hop mean pooling（类似一层 GCN 的消息传递）
    row = edges[0]
    col = edges[1]
    agg = np.zeros_like(x)
    counts = np.zeros(num_nodes)
    for r, c in zip(row, col):
        agg[r] += x[c]
        counts[r] += 1
    counts = np.maximum(counts, 1)
    agg = agg / counts[:, None]
    features = np.concatenate([x, agg], axis=1)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    clf = LogisticRegression(
        max_iter=max(epochs, 100),
        random_state=seed,
    )
    clf.fit(features[train_mask], y[train_mask])

    pred = clf.predict(features)
    train_acc = _accuracy_np(y[train_mask], pred[train_mask])
    val_acc = _accuracy_np(y[val_mask], pred[val_mask])
    test_acc = _accuracy_np(y[test_mask], pred[test_mask])

    used_name = f"Synthetic({name})"
    notes = (
        f"轻量 backend（1-hop mean pool + LogisticRegression），"
        f"适用于 Streamlit Cloud；演示 {model_name} 类节点分类流程。"
    )

    return ExperimentResult(
        dataset=used_name,
        model=f"{model_name}(light)",
        task_type="node_classification",
        train_accuracy=round(train_acc, 4),
        val_accuracy=round(val_acc, 4),
        test_accuracy=round(test_acc, 4),
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_nodes=num_nodes,
        num_edges=edges.shape[1],
        num_features=num_features,
        num_classes=num_classes,
        duration_seconds=round(time.time() - start, 2),
        notes=notes,
    )


def _run_pyg_gcn(
    dataset_name: str,
    model_name: str,
    epochs: int,
    hidden_dim: int,
    seed: int,
    start: float,
) -> ExperimentResult:
    import torch
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GCNConv

    class GCN(torch.nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    def accuracy(logits, y, mask):
        pred = logits.argmax(dim=-1)
        correct = int((pred[mask] == y[mask]).sum())
        return correct / int(mask.sum())

    def make_synthetic_graph():
        gen = torch.Generator().manual_seed(seed)
        num_nodes, num_features, num_classes = 400, 32, 5
        x = torch.randn(num_nodes, num_features, generator=gen)
        y = torch.randint(0, num_classes, (num_nodes,), generator=gen)
        src = torch.arange(num_nodes)
        dst = (src + 1) % num_nodes
        extra_n = num_nodes * 2
        extra_src = torch.randint(0, num_nodes, (extra_n,), generator=gen)
        extra_dst = torch.randint(0, num_nodes, (extra_n,), generator=gen)
        edge_index = torch.stack([
            torch.cat([src, extra_src]),
            torch.cat([dst, extra_dst]),
        ], dim=0)
        perm = torch.randperm(num_nodes, generator=gen)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:int(0.6 * num_nodes)]] = True
        val_mask[perm[int(0.6 * num_nodes):int(0.8 * num_nodes)]] = True
        test_mask[perm[int(0.8 * num_nodes):]] = True
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return data, num_classes

    torch.manual_seed(seed)
    name = dataset_name.capitalize()
    if name not in ("Cora", "Citeseer", "Pubmed"):
        name = "Cora"

    used_name = name
    try:
        dataset = Planetoid(root=f"/tmp/gnn_agent_{name.lower()}", name=name)
        data = dataset[0]
        num_classes = dataset.num_classes
        notes_prefix = f"在 {used_name} 上使用 {model_name} 完成节点分类 baseline"
    except Exception as exc:
        data, num_classes = make_synthetic_graph()
        used_name = f"Synthetic({name})"
        notes_prefix = (
            f"无法下载 Planetoid/{name}（{str(exc)[:80]}），"
            f"改用合成图演示 {model_name} 节点分类"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = GCN(data.num_node_features, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)

    notes = f"{notes_prefix}；训练 {epochs} epoch，hidden={hidden_dim}。"

    return ExperimentResult(
        dataset=used_name,
        model=model_name,
        task_type="node_classification",
        train_accuracy=round(accuracy(logits, data.y, data.train_mask), 4),
        val_accuracy=round(accuracy(logits, data.y, data.val_mask), 4),
        test_accuracy=round(accuracy(logits, data.y, data.test_mask), 4),
        epochs=epochs,
        hidden_dim=hidden_dim,
        num_nodes=data.num_nodes,
        num_edges=data.edge_index.size(1),
        num_features=data.num_node_features,
        num_classes=num_classes,
        duration_seconds=round(time.time() - start, 2),
        notes=notes,
    )


def run_gnn_baseline(
    dataset_name: str = "Cora",
    model_name: str = "GCN",
    epochs: int = 100,
    hidden_dim: int = 64,
    seed: int = 42,
) -> ExperimentResult:
    """节点分类 baseline：本地优先 PyG+GCN，云端自动降级为轻量图特征模型。"""
    start = time.time()
    if _use_light_backend():
        return _run_light_baseline(
            dataset_name, model_name, epochs, hidden_dim, seed, start,
        )
    return _run_pyg_gcn(
        dataset_name, model_name, epochs, hidden_dim, seed, start,
    )
