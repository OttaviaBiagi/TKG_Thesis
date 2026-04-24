"""
TGN — Temporal Graph Network (Rossi et al. 2020)
Exported as a module for use by UseCase2 and UseCase4 notebooks.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score


class MemoryModule(nn.Module):
    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.memory = nn.Parameter(
            torch.zeros(num_nodes, memory_dim), requires_grad=False)
        self.gru = nn.GRUCell(memory_dim, memory_dim)

    def get_memory(self, node_ids):
        return self.memory[node_ids]

    def update_memory(self, node_ids, messages):
        current = self.memory[node_ids]
        self.memory[node_ids] = self.gru(messages, current).detach()

    def reset(self):
        nn.init.zeros_(self.memory)


class MessageFunction(nn.Module):
    def __init__(self, memory_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim * 2 + edge_dim + 1, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(self, src_memory, dst_memory, edge_feat, delta_t):
        x = torch.cat([src_memory, dst_memory, edge_feat, delta_t], dim=-1)
        return self.mlp(x)


class TemporalEmbedding(nn.Module):
    def __init__(self, memory_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=memory_dim, num_heads=2, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim + edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, memory, edge_feat):
        mem_seq = memory.unsqueeze(1)
        attn_out, _ = self.attn(mem_seq, mem_seq, mem_seq)
        attn_out = attn_out.squeeze(1)
        return self.mlp(torch.cat([attn_out, edge_feat], dim=-1))


class AnomalyClassifier(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, embed):
        return torch.sigmoid(self.net(embed)).squeeze(-1)


class TGN(nn.Module):
    def __init__(self, num_nodes, memory_dim=32, message_dim=32,
                 embed_dim=32, edge_dim=2):
        super().__init__()
        self.memory = MemoryModule(num_nodes, memory_dim)
        self.message = MessageFunction(memory_dim, edge_dim, message_dim)
        self.embedding = TemporalEmbedding(memory_dim, edge_dim, embed_dim)
        self.classifier = AnomalyClassifier(embed_dim)

    def forward(self, src_ids, dst_ids, edge_feat, delta_t, update_memory=True):
        src_mem = self.memory.get_memory(src_ids)
        dst_mem = self.memory.get_memory(dst_ids)
        msg = self.message(src_mem, dst_mem, edge_feat, delta_t)
        if update_memory:
            self.memory.update_memory(dst_ids, msg)
        embed = self.embedding(dst_mem, edge_feat)
        return self.classifier(embed)


def train_tgn(model, train_data, epochs=3, batch_size=512, lr=1e-3):
    src, dst, ef, dt, y = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

    for epoch in range(epochs):
        model.train()
        model.memory.reset()
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(src), batch_size):
            s = src[i:i+batch_size]
            d = dst[i:i+batch_size]
            e = ef[i:i+batch_size]
            t = dt[i:i+batch_size]
            yb = y[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(s, d, e, t, update_memory=True)
            w = torch.where(yb == 1,
                            torch.full_like(yb, pos_weight),
                            torch.ones_like(yb))
            loss = (criterion(pred, yb) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f'  Epoch {epoch+1}/{epochs} — loss: {total_loss/n_batches:.4f}')


def evaluate_tgn(model, test_data, batch_size=512):
    model.eval()
    src, dst, ef, dt, y = test_data
    all_preds, all_scores = [], []
    with torch.no_grad():
        for i in range(0, len(src), batch_size):
            scores = model(src[i:i+batch_size], dst[i:i+batch_size],
                           ef[i:i+batch_size], dt[i:i+batch_size],
                           update_memory=False)
            all_scores.extend(scores.numpy())
            all_preds.extend((scores > 0.5).int().numpy())

    y_np = y.numpy()
    pred_np = np.array(all_preds)
    score_np = np.array(all_scores)

    print(classification_report(y_np, pred_np,
                                 target_names=['Normal', 'Anomaly'], digits=3))
    try:
        auc = roc_auc_score(y_np, score_np)
        print(f'  AUC-ROC: {auc:.4f}')
    except Exception:
        print('  AUC-ROC: N/A')
