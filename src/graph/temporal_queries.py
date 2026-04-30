"""
Query temporali per monitoring e prediction sul TKG.
Use case: rilevare anomalie, tracciare catene causali, predire guasti.
"""

from neo4j import GraphDatabase
from datetime import datetime, timezone

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "your_password"


class TKGMonitor:

    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    # ── Query 1: Sensori anomali in una finestra temporale ──
    def anomalies_in_window(self, start: str, end: str) -> list[dict]:
        """
        Trova tutti i sensori con anomalie in un intervallo temporale.
        Use case: dashboard monitoring real-time.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor)-[:MADE_OBSERVATION]->(o:Observation)
                WHERE o.is_anomaly <> false
                AND   o.valid_from >= $start
                AND   o.valid_from <= $end
                RETURN s.id          AS sensor_id,
                       s.name        AS sensor_name,
                       o.anomaly_type AS anomaly_type,
                       o.value        AS value,
                       o.valid_from   AS timestamp
                ORDER BY o.valid_from
            """, start=start, end=end)
            return [dict(r) for r in result]

    # ── Query 2: Catena causale da un'anomalia ──────────────
    def causal_chain(self, sensor_id: str, anomaly_timestamp: str) -> dict:
        """
        Traccia la catena causale: Component → Sensor → Observation → AnomalyEvent.
        Use case: root cause analysis per operatore.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Component)-[:HAS_SENSOR]->(s:Sensor {id: $sensor_id})
                      -[:MADE_OBSERVATION]->(o:Observation)
                      -[:DETECTED_ANOMALY]->(a:AnomalyEvent)
                WHERE o.valid_from >= $ts_start
                AND   o.valid_from <= $ts_end
                RETURN c.id          AS component,
                       s.id          AS sensor,
                       s.name        AS sensor_name,
                       o.value        AS value,
                       o.valid_from   AS timestamp,
                       a.anomaly_type AS anomaly_type
                ORDER BY o.valid_from
                LIMIT 20
            """, sensor_id=sensor_id,
                 ts_start=anomaly_timestamp[:10] + "T00:00:00",
                 ts_end=anomaly_timestamp[:10] + "T23:59:59")
            rows = [dict(r) for r in result]
            return {
                "sensor_id":    sensor_id,
                "anomaly_date": anomaly_timestamp[:10],
                "chain":        rows,
                "count":        len(rows),
            }

    # ── Query 3: Trend di degradazione ─────────────────────
    def degradation_trend(self, sensor_id: str, start: str, end: str) -> list[dict]:
        """
        Calcola la media del valore del sensore per ora.
        Use case: rilevare degradazione graduale nel tempo.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {id: $sensor_id})-[:MADE_OBSERVATION]->(o:Observation)
                WHERE o.valid_from >= $start
                AND   o.valid_from <= $end
                RETURN substring(o.valid_from, 0, 13) AS hour,
                       avg(o.value)                    AS avg_value,
                       max(o.value)                    AS max_value,
                       count(o)                        AS num_observations,
                       sum(CASE WHEN o.is_anomaly <> false THEN 1 ELSE 0 END) AS anomaly_count
                ORDER BY hour
            """, sensor_id=sensor_id, start=start, end=end)
            return [dict(r) for r in result]

    # ── Query 4: Alert predittivo ───────────────────────────
    def predictive_alert(self, sensor_id: str, last_n_hours: int = 6) -> dict:
        """
        Conta anomalie nelle ultime N ore e lancia alert se superano soglia.
        Use case: early warning prima di un guasto.
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Sensor {id: $sensor_id})-[:MADE_OBSERVATION]->(o:Observation)
                WHERE o.valid_from >= $start
                AND   o.is_anomaly <> false
                RETURN count(o)    AS anomaly_count,
                       avg(o.value) AS avg_anomaly_value,
                       max(o.value) AS max_anomaly_value
            """, sensor_id=sensor_id,
                 start=f"2024-01-07T{24 - last_n_hours:02d}:00:00")
            row = result.single()
            count = row["anomaly_count"] if row else 0
            return {
                "sensor_id":         sensor_id,
                "window_hours":      last_n_hours,
                "anomaly_count":     count,
                "alert":             count > 10,
                "severity":          "HIGH" if count > 100 else "MEDIUM" if count > 10 else "OK",
                "avg_anomaly_value": row["avg_anomaly_value"] if row else None,
                "max_anomaly_value": row["max_anomaly_value"] if row else None,
            }


def main():
    monitor = TKGMonitor()
    print("🔍 TKG Monitoring — Use Case Demo\n")

    # ── Use case 1: anomalie durante lo spike di vibrazione ─
    print("=" * 55)
    print("USE CASE 1: Anomalie durante spike vibrazione (giorno 7)")
    print("=" * 55)
    anomalies = monitor.anomalies_in_window(
        start="2024-01-08T10:00:00",
        end="2024-01-08T12:00:00"
    )
    print(f"  Trovate {len(anomalies)} anomalie")
    for a in anomalies[:5]:
        print(f"  [{a['timestamp']}] {a['sensor_id']} → {a['anomaly_type']} (value={a['value']:.2f})")

    # ── Use case 2: catena causale ──────────────────────────
    print("\n" + "=" * 55)
    print("USE CASE 2: Catena causale — VIB_001 giorno 8")
    print("=" * 55)
    chain = monitor.causal_chain("VIB_001", "2024-01-08T10:30:00")
    print(f"  Componente  : {chain['chain'][0]['component'] if chain['chain'] else 'N/A'}")
    print(f"  Sensore     : {chain['chain'][0]['sensor'] if chain['chain'] else 'N/A'}")
    print(f"  Anomalie    : {chain['count']}")
    if chain["chain"]:
        first = chain["chain"][0]
        print(f"  Prima anomalia: {first['timestamp']} → value={first['value']:.2f} ({first['anomaly_type']})")

    # ── Use case 3: trend degradazione temperatura ──────────
    print("\n" + "=" * 55)
    print("USE CASE 3: Trend degradazione TEMP_001 (giorni 16-18)")
    print("=" * 55)
    trend = monitor.degradation_trend(
        sensor_id="TEMP_001",
        start="2024-01-16T00:00:00",
        end="2024-01-18T23:59:59"
    )
    print(f"  {'Ora':<15} {'Media':>8} {'Max':>8} {'Anomalie':>10}")
    print("  " + "-" * 45)
    for row in trend[:10]:
        print(f"  {row['hour']:<15} {row['avg_value']:>8.2f} {row['max_value']:>8.2f} {row['anomaly_count']:>10}")

    # ── Use case 4: alert predittivo ────────────────────────
    print("\n" + "=" * 55)
    print("USE CASE 4: Alert predittivo — VIB_001")
    print("=" * 55)
    alert = monitor.predictive_alert("VIB_001", last_n_hours=6)
    print(f"  Sensore     : {alert['sensor_id']}")
    print(f"  Finestra    : ultime {alert['window_hours']} ore")
    print(f"  Anomalie    : {alert['anomaly_count']}")
    print(f"  Severità    : {alert['severity']}")
    print(f"  🚨 ALERT    : {'SÌ' if alert['alert'] else 'NO'}")

    monitor.close()
    print("\n✅ Demo completata!")


if __name__ == "__main__":
    main()