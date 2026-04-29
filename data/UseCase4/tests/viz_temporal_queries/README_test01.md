# UseCase4 — Test 01: Temporal Query Validation

## Obiettivo
Verificare che il TKG EPC modellato con dati reali TR supporti correttamente
le query temporali e la bitemporalità.

## Dataset
- **Fonte**: TR Family_Steps_macro.xlsm (dati aziendali reali)
- **Attività**: 276 (tutte le discipline EPC standard TR)
- **Step**: 1518 (sequenze reali di costruzione)
- **Work Permits**: 8 tipi (hot_work, excavation, lifting, electrical, ecc.)
- **Certificazioni**: 33
- **Lavoratori**: 50 (sintetici)
- **Evento bitemporale**: cambio regola hot_work al mese 6

## Grafo Neo4j — Schema
```
(Project)-[:INCLUDES]->(Activity)-[:BELONGS_TO]->(Family)
(Activity)-[:HAS_STEP {order, weight_pct}]->(Step)
(Step)-[:PRECEDES]->(Step)
(Step)-[:REQUIRES_PERMIT]->(WorkPermit)
(WorkPermit)-[:REQUIRES_CERT {valid_from, valid_to, tx_time}]->(Certification)
(Worker)-[:HAS_CERT {valid_from, valid_to, tx_time}]->(Certification)
```

## Query Testate

### Q1 — Snapshot Query (cosa è attivo in un dato momento)
```cypher
MATCH (s:Step)-[:REQUIRES_PERMIT]->(p:WorkPermit)
WHERE s.permit_type = 'hot_work'
AND s.valid_from <= '2024-07-01' AND s.valid_to >= '2024-07-01'
RETURN s.name, p.name
```
**Risultato**: 3 step di hot work attivi nel mese 6
**Significato**: Il TKG sa esattamente quali lavori pericolosi sono in corso in ogni momento

### Q2 — Gap Analysis (chi manca della nuova certificazione)
```cypher
MATCH (w:Worker)
WHERE NOT EXISTS {
  MATCH (w)-[r:HAS_CERT]->(c:Certification {name: 'Advanced Fire Watch'})
  WHERE r.valid_from <= '2024-07-01'
}
AND EXISTS { MATCH (w)-[:ASSIGNED_TO]->(s:Step {permit_type: 'hot_work'}) }
RETURN w.id, w.name AS workers_needing_retraining
```
**Risultato**: 50 lavoratori necessitano riqualifica dopo il cambio regola
**Significato**: Identifica automaticamente chi deve fare formazione aggiuntiva

### Q3 — Bitemporal Audit (cosa era richiesto in un dato momento storico)
```cypher
MATCH (p:WorkPermit {id: 'hot_work'})-[r:REQUIRES_CERT]->(c:Certification)
WHERE r.valid_from <= $query_date 
AND (r.valid_to IS NULL OR r.valid_to >= $query_date)
RETURN $query_date AS as_of, collect(c.name) AS required_certs
```
**Mese 5**: Hot Work Safety, Fire Watch, Welding Certification
**Mese 7**: + Advanced Fire Watch
**Significato**: Audit retroattivo completo — impossibile con sistemi tradizionali

## Visualizzazioni Prodotte

| Chart | File | Cosa mostra |
|-------|------|-------------|
| 1 | 1_permit_distribution.png | Distribuzione tipi di permesso tra i 1518 step |
| 2 | 2_project_gantt.png | Timeline 18 mesi per disciplina con marker cambio regola |
| 3 | 3_precedes_chain.png | Sequenza step Heat Exchangers con permessi associati |
| 4 | 4_bitemporal_rule_change.png | Evoluzione certificazioni hot_work nel tempo |

## Come riprodurre

```bash
# 1. Genera dataset
cd data/UseCase4
python3 generate_epc_dataset.py

# 2. Carica in Neo4j
python3 import_graph_real.py

# 3. Genera visualizzazioni e testa query
python3 tests/test_01_temporal_queries/visualize_epc_tkg.py
```

## Prossimi Test Pianificati
- **Test 02**: Worker assignment optimization (chi assegnare a quali step)
- **Test 03**: Critical path analysis via PRECEDES chain
- **Test 04**: Cross-discipline dependency detection
