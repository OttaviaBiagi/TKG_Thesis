"""
Step 4A — Baseline anomaly detection con Isolation Forest.
Confronta: modello senza grafo vs modello con features dal grafo TKG.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from neo4j import GraphDatabase

# ── Config ──────────────────────────────────────────────────
NEO4J_URI      = "bolt://172.22.43.151:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password"
DATA_PATH      = "data/raw/synthetic_turbine.csv"
CONTAMINATION  = 0.07   # ~7% anomalie attese (dal nostro dataset: 6.8%)


# ════════════════════════════════════════════════════════════
# PARTE 1 — Baseline: Isolation Forest su features raw
# ════════════════════════════════════════════════════════════

def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering base: pivot sensori + rolling statistics.
    Nessun contesto dal grafo — baseline puro.
    """
    # pivot: una riga per timestamp, una colonna per sensore
    pivot = df.pivot_table(
        index="timestamp",
        columns="sensor_id",
        values="value",
        aggfunc="mean"
    ).reset_index()

    # rolling mean e std su finestra di 6 letture (1 minuto)
    sensor_cols = [c for c in pivot.columns if c != "timestamp"]
    for col in sensor_cols:
        pivot[f"{col}_roll_mean"] = pivot[col].rolling(6, min_periods=1).mean()
        pivot[f"{col}_roll_std"]  = pivot[col].rolling(6, min_periods=1).std().fillna(0)

    return pivot


def run_baseline(df: pd.DataFrame) -> dict:
    """Allena Isolation Forest su features raw e valuta."""
    print("📊 Costruzione features raw...")
    features_df = build_raw_features(df)

    # ground truth: timestamp anomalo se ALMENO un sensore è anomalo
    anomaly_by_ts = df.groupby("timestamp")["is_anomaly"].any().reset_index()
    anomaly_by_ts.columns = ["timestamp", "is_anomaly"]
    features_df = features_df.merge(anomaly_by_ts, on="timestamp")

    feature_cols = [c for c in features_df.columns
                    if c not in ["timestamp", "is_anomaly"]]

    X = features_df[feature_cols].values
    y = features_df["is_anomaly"].astype(int).values

    # normalizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train su primi 70% (split temporale — no random!)
    split = int(len(X_scaled) * 0.7)
    X_train = X_scaled[:split]
    X_test  = X_scaled[split:]
    y_test  = y[split:]

    y_arr = np.array(y_test)
    print(f"  Train: {len(X_train):,} timestamp | Test: {len(X_test):,} timestamp")
    print(f"  Anomalie nel test set: {y_arr.sum():,} ({y_arr.mean()*100:.1f}%)")

    # Isolation Forest
    print("\n🌲 Training Isolation Forest (baseline)...")
    model = IsolationForest(
        contamination=CONTAMINATION,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # predizioni: IF restituisce -1 (anomaly) o 1 (normal)
    y_pred_raw = model.predict(X_test)
    y_pred     = (y_pred_raw == -1).astype(int)
    scores     = -model.score_samples(X_test)   # anomaly score

    return {
        "y_test":  y_test,
        "y_pred":  y_pred,
        "scores":  scores,
        "model":   "Isolation Forest (baseline)",
    }


# ════════════════════════════════════════════════════════════
# PARTE 2 — Graph-enhanced: features dal TKG Neo4j
# ════════════════════════════════════════════════════════════

def fetch_graph_features(start: str, end: str) -> pd.DataFrame:
    """
    Estrae features temporali dal grafo TKG.
    Aggiunge: anomaly_rate nell'ora precedente per ogni sensore.
    Questo è il vantaggio del TKG rispetto al baseline.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        result = session.run("""
            MATCH (s:Sensor)-[:MADE_OBSERVATION]->(o:Observation)
            WHERE o.valid_from >= $start
            AND   o.valid_from <= $end
            RETURN substring(o.valid_from, 0, 13) AS hour,
                   s.id                            AS sensor_id,
                   avg(o.value)                    AS avg_value,
                   count(o)                        AS obs_count,
                   sum(CASE WHEN o.is_anomaly <> false
                       THEN 1 ELSE 0 END)          AS anomaly_count
            ORDER BY hour, sensor_id
        """, start=start, end=end)
        rows = [dict(r) for r in result]
    driver.close()

    df = pd.DataFrame(rows)
    df["anomaly_rate"] = df["anomaly_count"] / df["obs_count"]
    return df


# ════════════════════════════════════════════════════════════
# PARTE 3 — Valutazione e confronto
# ════════════════════════════════════════════════════════════

def evaluate(result: dict) -> None:
    """Stampa metriche di valutazione."""
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    scores = result["scores"]
    name   = result["model"]

    print(f"\n{'='*55}")
    print(f"📈 Risultati: {name}")
    print(f"{'='*55}")
    print(classification_report(
        y_test, y_pred,
        target_names=["Normal", "Anomaly"],
        digits=3
    ))
    try:
        auc = roc_auc_score(y_test, scores)
        print(f"  AUC-ROC: {auc:.4f}")
    except Exception:
        print("  AUC-ROC: N/A (solo una classe nel test set)")


def main():
    print("🚀 Step 4 — Anomaly Detection con Isolation Forest\n")

    # carica dati
    print(f"📂 Lettura {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} righe | {df['is_anomaly'].sum():,} anomalie\n")

    # ── Baseline ─────────────────────────────────────────────
    baseline_result = run_baseline(df)
    evaluate(baseline_result)

    # ── Graph features preview ───────────────────────────────
    print(f"\n{'='*55}")
    print("🔗 Graph-enhanced features (preview da Neo4j)")
    print(f"{'='*55}")
    print("  Estrazione anomaly_rate orario dal TKG...")
    graph_df = fetch_graph_features(
        start="2024-01-08T09:00:00",
        end="2024-01-08T12:00:00"
    )
    print(f"  {len(graph_df)} righe estratte dal grafo")
    print("\n  Esempio — anomaly_rate durante lo spike VIB_001:")
    vib = graph_df[graph_df["sensor_id"] == "VIB_001"]
    print(vib[["hour", "avg_value", "anomaly_rate"]].to_string(index=False))

    print("\n✅ Step 4A completato!")
    print("\nProssimo step: confrontare questo baseline con TGN")
    print("→ Il grafo aggiunge contesto temporale che IF non vede")


if __name__ == "__main__":
    main()