"""
Carica i dati sintetici della turbina in Neo4j come Temporal Knowledge Graph.

Schema:
  (Component)-[:HAS_SENSOR]->(Sensor)-[:MADE_OBSERVATION]->(Observation)
  (Observation)-[:DETECTED_ANOMALY]->(AnomalyEvent)
"""

import os
import pandas as pd
from neo4j import GraphDatabase
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────
NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password"
DATA_PATH      = "data/raw/synthetic_turbine.csv"
BATCH_SIZE     = 2000   # righe per batch


def create_constraints(session):
    """Crea constraints e indici per performance."""
    print("📐 Creazione constraints...")
    queries = [
        "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT sensor_id IF NOT EXISTS FOR (s:Sensor) REQUIRE s.id IS UNIQUE",
        "CREATE INDEX obs_timestamp IF NOT EXISTS FOR (o:Observation) ON (o.valid_from)",
        "CREATE INDEX obs_anomaly IF NOT EXISTS FOR (o:Observation) ON (o.is_anomaly)",
    ]
    for q in queries:
        try:
            session.run(q)
        except Exception as e:
            print(f"  ⚠️  {e}")
    print("  ✅ Constraints pronti")


def create_component_and_sensors(session, df):
    """Crea il nodo Component e i nodi Sensor."""
    print("🏭 Creazione Component e Sensor...")

    # Component
    session.run("""
        MERGE (c:Component {id: $id})
        SET c.name         = $name,
            c.type         = 'Turbine',
            c.valid_from   = $valid_from,
            c.tx_time      = $tx_time
    """, id="TURBINE_001", name="Industrial Turbine Unit 1",
         valid_from=df["timestamp"].min(),
         tx_time=datetime.now(timezone.utc).isoformat())

    # Sensori
    sensors = df[["sensor_id", "sensor_name", "unit"]].drop_duplicates()
    for _, row in sensors.iterrows():
        session.run("""
            MERGE (s:Sensor {id: $id})
            SET s.name       = $name,
                s.unit       = $unit,
                s.valid_from = $valid_from,
                s.tx_time    = $tx_time
            WITH s
            MATCH (c:Component {id: 'TURBINE_001'})
            MERGE (c)-[:HAS_SENSOR]->(s)
        """, id=row["sensor_id"],
             name=row["sensor_name"],
             unit=row["unit"],
             valid_from=df["timestamp"].min(),
             tx_time=datetime.now(timezone.utc).isoformat())

    print(f"  ✅ 1 Component + {len(sensors)} Sensor creati")


def load_observations_batched(session, df):
    """Carica le Observation in batch con bitemporalità."""
    print(f"📊 Caricamento {len(df):,} osservazioni in batch da {BATCH_SIZE}...")

    tx_time = datetime.now(timezone.utc).isoformat()
    total   = 0

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i:i + BATCH_SIZE].to_dict("records")

        session.run("""
            UNWIND $rows AS row
            MATCH (s:Sensor {id: row.sensor_id})
            CREATE (o:Observation {
                sensor_id    : row.sensor_id,
                value        : row.value,
                unit         : row.unit,
                valid_from   : row.timestamp,
                valid_to     : null,
                tx_time      : row.tx_time,
                is_anomaly   : row.is_anomaly,
                anomaly_type : row.anomaly_type
            })
            CREATE (s)-[:MADE_OBSERVATION {valid_from: row.timestamp}]->(o)
            WITH o, row
            WHERE row.is_anomaly = true
            CREATE (a:AnomalyEvent {
                sensor_id    : row.sensor_id,
                anomaly_type : row.anomaly_type,
                valid_from   : row.timestamp,
                tx_time      : row.tx_time,
                value        : row.value
            })
            CREATE (o)-[:DETECTED_ANOMALY]->(a)
        """, rows=[{**r, "tx_time": tx_time} for r in batch])

        total += len(batch)
        if total % 50000 == 0 or total >= len(df):
            pct = total / len(df) * 100
            print(f"  → {total:,} / {len(df):,} ({pct:.0f}%)")

    print(f"  ✅ Osservazioni caricate")


def print_summary(session):
    """Stampa un riepilogo del grafo creato."""
    print("\n📈 Riepilogo grafo TKG:")
    counts = {
        "Component"    : "MATCH (n:Component) RETURN count(n) AS c",
        "Sensor"       : "MATCH (n:Sensor) RETURN count(n) AS c",
        "Observation"  : "MATCH (n:Observation) RETURN count(n) AS c",
        "AnomalyEvent" : "MATCH (n:AnomalyEvent) RETURN count(n) AS c",
    }
    for label, query in counts.items():
        result = session.run(query).single()["c"]
        print(f"  {label:<15} : {result:,}")


def main():
    print("🚀 TKG Loader — Turbina Sintetica\n")

    # Carica CSV
    print(f"📂 Lettura {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} righe caricate\n")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        create_constraints(session)
        create_component_and_sensors(session, df)
        load_observations_batched(session, df)
        print_summary(session)

    driver.close()
    print("\n✅ TKG caricato con successo in Neo4j!")


if __name__ == "__main__":
    main()