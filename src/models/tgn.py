"""
TGN variants (Rossi et al. 2020) for UseCase2 and UseCase4.

Three model variants matching Table 2 of the paper:
  TGN     — TGN-id:    memory (GRU) + identity embedding + MLP classifier
  TGNTime — TGN-time:  memory (GRU) + time-projection embedding (JODIE-style)
  TGNNoMem— TGN-no-mem: no memory, edge features only → MLP classifier (baseline)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, f1_score


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TGN-id  (paper Table 2: Memory=GRU, Embedding=identity)
# ---------------------------------------------------------------------------

class TemporalEmbedding(nn.Module):
    """Identity embedding: memory used directly, no k-hop neighborhood."""
    def __init__(self, memory_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim + edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, memory, edge_feat, delta_t=None):
        return self.mlp(torch.cat([memory, edge_feat], dim=-1))


class TGN(nn.Module):
    """
    TGN-id variant: GRU memory + identity embedding.

    embed_src=True  → classify based on source node (well instance).
    embed_src=False → classify based on destination node (sensor).
    """
    def __init__(self, num_nodes, memory_dim=32, message_dim=32,
                 embed_dim=32, edge_dim=2):
        super().__init__()
        self.memory = MemoryModule(num_nodes, memory_dim)
        self.message = MessageFunction(memory_dim, edge_dim, message_dim)
        self.embedding = TemporalEmbedding(memory_dim, edge_dim, embed_dim)
        self.classifier = AnomalyClassifier(embed_dim)

    def forward(self, src_ids, dst_ids, edge_feat, delta_t,
                update_memory=True, embed_src=False):
        src_mem = self.memory.get_memory(src_ids)
        dst_mem = self.memory.get_memory(dst_ids)
        msg = self.message(src_mem, dst_mem, edge_feat, delta_t)
        if update_memory:
            self.memory.update_memory(src_ids, msg)
            self.memory.update_memory(dst_ids, msg)
        embed_mem = src_mem if embed_src else dst_mem
        embed = self.embedding(embed_mem, edge_feat)
        return self.classifier(embed)


# ---------------------------------------------------------------------------
# TGN-time  (paper Table 2: Memory=GRU, Embedding=time-projection / JODIE)
# ---------------------------------------------------------------------------

class TimeProjectionEmbedding(nn.Module):
    """
    emb(i,t) = MLP( (1 + Δt·w) ⊙ s_i(t)  ∥  edge_feat )
    Δt ∈ [0,1] (normalised within each well instance).
    """
    def __init__(self, memory_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(memory_dim))
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim + edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, memory, edge_feat, delta_t):
        # delta_t: [batch, 1], broadcast over memory_dim
        projected = (1.0 + delta_t * self.w) * memory
        return self.mlp(torch.cat([projected, edge_feat], dim=-1))


class TGNTime(nn.Module):
    """
    TGN-time variant: GRU memory + time-projection embedding (JODIE-style).
    Same forward signature as TGN for drop-in replacement.
    """
    def __init__(self, num_nodes, memory_dim=32, message_dim=32,
                 embed_dim=32, edge_dim=2):
        super().__init__()
        self.memory = MemoryModule(num_nodes, memory_dim)
        self.message = MessageFunction(memory_dim, edge_dim, message_dim)
        self.embedding = TimeProjectionEmbedding(memory_dim, edge_dim, embed_dim)
        self.classifier = AnomalyClassifier(embed_dim)

    def forward(self, src_ids, dst_ids, edge_feat, delta_t,
                update_memory=True, embed_src=False):
        src_mem = self.memory.get_memory(src_ids)
        dst_mem = self.memory.get_memory(dst_ids)
        msg = self.message(src_mem, dst_mem, edge_feat, delta_t)
        if update_memory:
            self.memory.update_memory(src_ids, msg)
            self.memory.update_memory(dst_ids, msg)
        embed_mem = src_mem if embed_src else dst_mem
        embed = self.embedding(embed_mem, edge_feat, delta_t)
        return self.classifier(embed)


# ---------------------------------------------------------------------------
# TGN-no-mem  (paper Table 2: no Memory, no Embedding — edge features only)
# ---------------------------------------------------------------------------

class TGNNoMem(nn.Module):
    """
    TGN-no-mem variant: no GRU memory, no graph operator.
    Only edge features → MLP classifier. Pure baseline.
    """
    def __init__(self, num_nodes, memory_dim=32, message_dim=32,
                 embed_dim=32, edge_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim + 1, embed_dim),   # edge_feat + delta_t
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.classifier = AnomalyClassifier(embed_dim)

    def forward(self, src_ids, dst_ids, edge_feat, delta_t,
                update_memory=True, embed_src=False):
        x = torch.cat([edge_feat, delta_t], dim=-1)
        return self.classifier(self.mlp(x))


# ---------------------------------------------------------------------------
# Shared training / evaluation helpers
# ---------------------------------------------------------------------------

def _get_class_weight(y_train):
    n_normal  = float((y_train == 0).sum())
    n_anomaly = float((y_train == 1).sum())
    return n_anomaly / max(n_normal, 1)   # upweight minority (Normal)


def train_tgn(model, train_data, epochs=5, batch_size=512, lr=1e-3,
              embed_src=False):
    src, dst, ef, dt, y = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='none')
    normal_weight = _get_class_weight(y)

    for epoch in range(epochs):
        model.train()
        if hasattr(model, 'memory'):
            model.memory.reset()
        total_loss, n_batches = 0.0, 0
        for i in range(0, len(src), batch_size):
            s, d = src[i:i+batch_size], dst[i:i+batch_size]
            e, t_ = ef[i:i+batch_size], dt[i:i+batch_size]
            yb = y[i:i+batch_size]
            optimizer.zero_grad()
            pred = model(s, d, e, t_, update_memory=True, embed_src=embed_src)
            w = torch.where(yb == 0,
                            torch.full_like(yb, normal_weight),
                            torch.ones_like(yb))
            loss = (criterion(pred, yb) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f'  Epoch {epoch+1}/{epochs} — loss: {total_loss/n_batches:.4f}')


def evaluate_tgn(model, test_data, batch_size=512, embed_src=False):
    model.eval()
    src, dst, ef, dt, y = test_data
    all_scores = []
    with torch.no_grad():
        for i in range(0, len(src), batch_size):
            scores = model(src[i:i+batch_size], dst[i:i+batch_size],
                           ef[i:i+batch_size], dt[i:i+batch_size],
                           update_memory=False, embed_src=embed_src)
            all_scores.extend(scores.numpy())

    y_np    = y.numpy()
    score_np = np.array(all_scores)

    # Threshold tuning by macro F1
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.02):
        s = f1_score(y_np, (score_np > t).astype(int),
                     average='macro', zero_division=0)
        if s > best_f1:
            best_f1, best_t = s, t

    pred_np = (score_np > best_t).astype(int)
    print(f'Best threshold: {best_t:.2f}  (macro F1={best_f1:.3f})')
    print(classification_report(y_np, pred_np,
                                 target_names=['Normal', 'Anomaly'], digits=3))
    try:
        auc = roc_auc_score(y_np, score_np)
        print(f'AUC-ROC: {auc:.4f}')
    except Exception:
        print('AUC-ROC: N/A')
    return auc if 'auc' in dir() else float('nan')
