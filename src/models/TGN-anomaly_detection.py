"""
TGN — Temporal Graph Network (Rossi et al. 2020)
Implementazione da zero con PyTorch puro.

Architettura:
  Memory Module     → stato persistente per ogni nodo
  Message Function  → aggrega eventi su un arco
  Embedding Module  → GNN temporale per predizione
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ── Riproducibilità ─────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = "data/raw/synthetic_turbine.csv"


# ════════════════════════════════════════════════════════════
# 1. MEMORY MODULE
# ════════════════════════════════════════════════════════════

class MemoryModule(nn.Module):
    """
    Mantiene uno stato (memory) per ogni nodo del grafo.
    Aggiornato con GRU ad ogni nuovo evento sul nodo.
    """

    def __init__(self, num_nodes: int, memory_dim: int):
        super().__init__()
        self.num_nodes  = num_nodes
        self.memory_dim = memory_dim

        # memoria persistente per ogni nodo
        self.memory = nn.Parameter(
            torch.zeros(num_nodes, memory_dim),
            requires_grad=False
        )
        # GRU per aggiornare la memoria
        self.gru = nn.GRUCell(memory_dim, memory_dim)

    def get_memory(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.memory[node_ids]

    def update_memory(self, node_ids: torch.Tensor, messages: torch.Tensor):
        """Aggiorna la memoria dei nodi con i nuovi messaggi."""
        current = self.memory[node_ids]
        updated = self.gru(messages, current)
        self.memory[node_ids] = updated.detach()

    def reset(self):
        nn.init.zeros_(self.memory)


# ════════════════════════════════════════════════════════════
# 2. MESSAGE FUNCTION
# ════════════════════════════════════════════════════════════

class MessageFunction(nn.Module):
    """
    Calcola il messaggio da un evento (arco temporale).
    Input:  memoria src + memoria dst + features evento + Δt
    Output: messaggio aggregato
    """

    def __init__(self, memory_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        input_dim = memory_dim * 2 + edge_dim + 1  # +1 per Δt
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(
        self,
        src_memory: torch.Tensor,   # [B, memory_dim]
        dst_memory: torch.Tensor,   # [B, memory_dim]
        edge_feat:  torch.Tensor,   # [B, edge_dim]
        delta_t:    torch.Tensor,   # [B, 1]
    ) -> torch.Tensor:
        x = torch.cat([src_memory, dst_memory, edge_feat, delta_t], dim=-1)
        return self.mlp(x)


# ════════════════════════════════════════════════════════════
# 3. TEMPORAL EMBEDDING + CLASSIFIER
# ════════════════════════════════════════════════════════════

class TemporalEmbedding(nn.Module):
    """
    Embedding temporale del nodo per predizione anomalia.
    Combina memoria corrente + features evento.
    """

    def __init__(self, memory_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=2,
            batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(memory_dim + edge_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        memory:    torch.Tensor,   # [B, memory_dim]
        edge_feat: torch.Tensor,   # [B, edge_dim]
    ) -> torch.Tensor:
        # self-attention sulla memoria (aggiungi dim sequenza)
        mem_seq = memory.unsqueeze(1)                        # [B, 1, D]
        attn_out, _ = self.attn(mem_seq, mem_seq, mem_seq)
        attn_out = attn_out.squeeze(1)                       # [B, D]
        x = torch.cat([attn_out, edge_feat], dim=-1)
        return self.mlp(x)


class AnomalyClassifier(nn.Module):
    """Classificatore binario anomalia/normale sull'embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(embed)).squeeze(-1)


# ════════════════════════════════════════════════════════════
# 4. TGN COMPLETO
# ════════════════════════════════════════════════════════════

class TGN(nn.Module):
    def __init__(
        self,
        num_nodes:   int,
        memory_dim:  int = 32,
        message_dim: int = 32,
        embed_dim:   int = 32,
        edge_dim:    int = 2,    # [value_normalizzato, delta_t_normalizzato]
    ):
        super().__init__()
        self.memory    = MemoryModule(num_nodes, memory_dim)
        self.message   = MessageFunction(memory_dim, edge_dim, message_dim)
        self.embedding = TemporalEmbedding(memory_dim, edge_dim, embed_dim)
        self.classifier = AnomalyClassifier(embed_dim)

    def forward(
        self,
        src_ids:   torch.Tensor,   # [B] indici nodo sorgente (component)
        dst_ids:   torch.Tensor,   # [B] indici nodo destinazione (sensor)
        edge_feat: torch.Tensor,   # [B, edge_dim]
        delta_t:   torch.Tensor,   # [B, 1]
        update_memory: bool = True,
    ) -> torch.Tensor:

        src_mem = self.memory.get_memory(src_ids)
        dst_mem = self.memory.get_memory(dst_ids)

        # calcola messaggi
        msg = self.message(src_mem, dst_mem, edge_feat, delta_t)

        # aggiorna memoria
        if update_memory:
            self.memory.update_memory(dst_ids, msg)

        # embedding e classificazione
        embed = self.embedding(dst_mem, edge_feat)
        return self.classifier(embed)


# ════════════════════════════════════════════════════════════
# 5. PREPARAZIONE DATI
# ════════════════════════════════════════════════════════════

def prepare_tgn_data(df: pd.DataFrame):
    """
    Converte il DataFrame in sequenza di eventi per TGN.
    Ogni evento = (src, dst, timestamp, features, label)
    """
    # mappa nodi a indici interi
    components = df["component_id"].unique()
    sensors    = df["sensor_id"].unique()
    all_nodes  = list(components) + list(sensors)
    node2idx   = {n: i for i, n in enumerate(all_nodes)}

    # normalizza values
    scaler = StandardScaler()
    df = df.copy()
    df["value_norm"] = scaler.fit_transform(df[["value"]])

    # converti timestamp in secondi
    df["ts_sec"] = pd.to_datetime(df["timestamp"]).astype(np.int64) // 10**9

    # calcola delta_t per ogni sensore
    df = df.sort_values("timestamp")
    df["delta_t"] = df.groupby("sensor_id")["ts_sec"].diff().fillna(0)
    df["delta_t_norm"] = df["delta_t"] / (df["delta_t"].max() + 1e-8)

    # split temporale 70/30
    timestamps_sorted = df["timestamp"].sort_values().unique()
    split_ts = timestamps_sorted[int(len(timestamps_sorted) * 0.7)]
    train_df = df[df["timestamp"] <= split_ts]
    test_df  = df[df["timestamp"] > split_ts]

    def to_tensors(d):
        src = torch.tensor([node2idx[c] for c in d["component_id"]], dtype=torch.long)
        dst = torch.tensor([node2idx[s] for s in d["sensor_id"]],    dtype=torch.long)
        ef  = torch.tensor(
            d[["value_norm", "delta_t_norm"]].values,
            dtype=torch.float32
        )
        dt  = torch.tensor(d["delta_t_norm"].values, dtype=torch.float32).unsqueeze(1)
        y   = torch.tensor(d["is_anomaly"].astype(int).values, dtype=torch.float32)
        return src, dst, ef, dt, y

    return to_tensors(train_df), to_tensors(test_df), len(all_nodes)


# ════════════════════════════════════════════════════════════
# 6. TRAINING E VALUTAZIONE
# ════════════════════════════════════════════════════════════

def train_tgn(model, train_data, epochs=3, batch_size=512, lr=1e-3):
    src, dst, ef, dt, y = train_data
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # peso classe anomalia (upsampling implicito)
    pos_weight = (y == 0).sum() / (y == 1).sum()

    for epoch in range(epochs):
        model.train()
        model.memory.reset()
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(src), batch_size):
            s  = src[i:i+batch_size]
            d  = dst[i:i+batch_size]
            e  = ef[i:i+batch_size]
            t  = dt[i:i+batch_size]
            yb = y[i:i+batch_size]

            optimizer.zero_grad()
            pred = model(s, d, e, t, update_memory=True)

            # weighted BCE per classi sbilanciate
            weights = torch.where(yb == 1,
                                  torch.full_like(yb, pos_weight),
                                  torch.ones_like(yb))
            loss = (criterion(pred, yb) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")


def evaluate_tgn(model, test_data, batch_size=512):
    model.eval()
    src, dst, ef, dt, y = test_data
    all_preds  = []
    all_scores = []

    with torch.no_grad():
        for i in range(0, len(src), batch_size):
            s = src[i:i+batch_size]
            d = dst[i:i+batch_size]
            e = ef[i:i+batch_size]
            t = dt[i:i+batch_size]

            scores = model(s, d, e, t, update_memory=False)
            preds  = (scores > 0.5).int()
            all_scores.extend(scores.numpy())
            all_preds.extend(preds.numpy())

    y_np     = y.numpy()
    pred_np  = np.array(all_preds)
    score_np = np.array(all_scores)

    print(f"\n{'='*55}")
    print("📈 Risultati: TGN (Temporal Graph Network)")
    print(f"{'='*55}")
    print(classification_report(
        y_np, pred_np,
        target_names=["Normal", "Anomaly"],
        digits=3
    ))
    try:
        auc = roc_auc_score(y_np, score_np)
        print(f"  AUC-ROC: {auc:.4f}")
    except Exception:
        print("  AUC-ROC: N/A")


# ════════════════════════════════════════════════════════════
# 7. MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("🚀 TGN — Temporal Graph Network\n")

    print(f"📂 Lettura {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} righe | {df['is_anomaly'].sum():,} anomalie\n")

    print("🔧 Preparazione dati per TGN...")
    train_data, test_data, num_nodes = prepare_tgn_data(df)
    src_tr, _, _, _, y_tr = train_data
    src_te, _, _, _, y_te = test_data
    print(f"  Nodi nel grafo : {num_nodes}")
    print(f"  Train events   : {len(src_tr):,}")
    print(f"  Test events    : {len(src_te):,}")
    print(f"  Anomalie train : {int(y_tr.sum()):,} ({float(y_tr.mean())*100:.1f}%)")
    print(f"  Anomalie test  : {int(y_te.sum()):,} ({float(y_te.mean())*100:.1f}%)\n")

    print("🧠 Inizializzazione TGN...")
    model = TGN(
        num_nodes=num_nodes,
        memory_dim=32,
        message_dim=32,
        embed_dim=32,
        edge_dim=2,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametri totali: {n_params:,}\n")

    print("🏋️  Training TGN (3 epoche)...")
    train_tgn(model, train_data, epochs=3, batch_size=512)

    print("\n📊 Valutazione sul test set...")
    evaluate_tgn(model, test_data)

    print("\n✅ TGN completato!")
    print("\n📋 Confronto modelli:")
    print("  Isolation Forest  → F1=0.932, AUC-ROC=0.991 (no graph context)")
    print("  TGN               → vedi risultati sopra   (con graph context)")


if __name__ == "__main__":
    main()