"""
Generatore di dati sintetici per turbina industriale.
Simula 5 sensori per 30 giorni con anomalie iniettate.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Configurazione ──────────────────────────────────────────
SENSORS = [
    {"id": "TEMP_001",  "name": "temperature",  "unit": "°C",  "base": 85.0,  "noise": 1.5},
    {"id": "PRES_001",  "name": "pressure",     "unit": "bar", "base": 12.0,  "noise": 0.3},
    {"id": "VIB_001",   "name": "vibration",    "unit": "mm/s","base": 2.5,   "noise": 0.2},
    {"id": "FLOW_001",  "name": "flow_rate",    "unit": "L/s", "base": 45.0,  "noise": 1.0},
    {"id": "RPM_001",   "name": "rpm",          "unit": "RPM", "base": 3000.0,"noise": 20.0},
]

COMPONENT_ID  = "TURBINE_001"
START_DATE    = datetime(2024, 1, 1)
DAYS          = 30
FREQ_SECONDS  = 10        # una lettura ogni 10 secondi
RANDOM_SEED   = 42

# ── Anomalie ────────────────────────────────────────────────
ANOMALIES = [
    # spike improvviso su vibrazione (giorno 7, durata 2 ore)
    {
        "type":       "spike",
        "sensor_id":  "VIB_001",
        "start":      START_DATE + timedelta(days=7, hours=10),
        "end":        START_DATE + timedelta(days=7, hours=12),
        "magnitude":  5.0,   # moltiplicatore del valore base
    },
    # degradazione graduale su temperatura (giorni 15-25)
    {
        "type":       "gradual_degradation",
        "sensor_id":  "TEMP_001",
        "start":      START_DATE + timedelta(days=15),
        "end":        START_DATE + timedelta(days=25),
        "magnitude":  20.0,  # incremento totale in °C
    },
    # pattern ciclico anomalo su pressione (giorno 20, 4 ore)
    {
        "type":       "cyclic_anomaly",
        "sensor_id":  "PRES_001",
        "start":      START_DATE + timedelta(days=20, hours=6),
        "end":        START_DATE + timedelta(days=20, hours=10),
        "magnitude":  3.0,   # ampiezza oscillazione
    },
]


def _apply_anomaly(value: float, ts: datetime, anomaly: dict) -> tuple[float, str]:
    """Applica una singola anomalia al valore se il timestamp è nell'intervallo."""
    if not (anomaly["start"] <= ts <= anomaly["end"]):
        return value, "none"

    t = anomaly["type"]

    if t == "spike":
        return value * anomaly["magnitude"], t

    if t == "gradual_degradation":
        duration = (anomaly["end"] - anomaly["start"]).total_seconds()
        elapsed  = (ts - anomaly["start"]).total_seconds()
        progress = elapsed / duration          # 0 → 1
        return value + anomaly["magnitude"] * progress, t

    if t == "cyclic_anomaly":
        elapsed = (ts - anomaly["start"]).total_seconds()
        return value + anomaly["magnitude"] * np.sin(2 * np.pi * elapsed / 600), t

    return value, "none"


def generate(output_path: str = "data/raw/synthetic_turbine.csv") -> pd.DataFrame:
    rng       = np.random.default_rng(RANDOM_SEED)
    total_sec = DAYS * 24 * 3600
    timestamps = [START_DATE + timedelta(seconds=i) for i in range(0, total_sec, FREQ_SECONDS)]

    rows: list[dict] = []

    for sensor in SENSORS:
        print(f"  Generando {sensor['id']} ({len(timestamps):,} letture)…")
        for ts in timestamps:
            # valore base + rumore gaussiano
            value        = sensor["base"] + rng.normal(0, sensor["noise"])
            is_anomaly   = False
            anomaly_type = "none"

            # controlla tutte le anomalie per questo sensore
            for anom in ANOMALIES:
                if anom["sensor_id"] == sensor["id"]:
                    value, atype = _apply_anomaly(value, ts, anom)
                    if atype != "none":
                        is_anomaly   = True
                        anomaly_type = atype

            rows.append({
                "timestamp":    ts.isoformat(),
                "sensor_id":    sensor["id"],
                "sensor_name":  sensor["name"],
                "component_id": COMPONENT_ID,
                "value":        round(value, 4),
                "unit":         sensor["unit"],
                "is_anomaly":   is_anomaly,
                "anomaly_type": anomaly_type,
            })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Dataset salvato in {output_path}")
    print(f"   Righe totali : {len(df):,}")
    print(f"   Anomalie     : {df['is_anomaly'].sum():,} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"   Periodo      : {df['timestamp'].min()} → {df['timestamp'].max()}")
    return df


if __name__ == "__main__":
    print("🔧 Generazione dati sintetici turbina…\n")
    df = generate()
    print("\nPrima anteprima:")
    print(df.head(3).to_string())
    print("\nDistribuzione anomalie per tipo:")
    print(df[df["is_anomaly"]].groupby(["sensor_id", "anomaly_type"]).size())
