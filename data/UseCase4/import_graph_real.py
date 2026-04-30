"""
UseCase4 — Import EPC TKG into Neo4j  (database: uc4)
Reads epc_dataset_real.json and loads all nodes + relations.
Each use-case uses its own Neo4j database to avoid label collisions.
"""
import json
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI      = 'bolt://localhost:7687'
NEO4J_USER     = 'neo4j'
NEO4J_PASSWORD = 'your_password'
DATA_FILE      = Path(__file__).parent / 'epc_dataset_real.json'


def load(session, dataset):
    tx = dataset

    # Project — use_case property disambiguates UC4 nodes from other use cases
    p = tx['project']
    session.run('MERGE (pr:Project {id:$id}) SET pr += $props SET pr.use_case=$uc',
                id=p['id'], props=p, uc='uc4')
    print(f"  Project: {p['name']}")

    # Families
    for f in tx['families']:
        session.run('MERGE (f:Family {id:$id}) SET f += $props',
                    id=f['id'], props=f)
    print(f"  Families: {len(tx['families'])}")

    # Activities
    project_id = tx['project']['id']
    BATCH = 200
    acts = tx['activities']
    for i in range(0, len(acts), BATCH):
        batch = acts[i:i+BATCH]
        session.run('''
            UNWIND $rows AS a
            MERGE (act:Activity {id:a.id}) SET act += a
            WITH act, a
            MATCH (f:Family {id:a.family})
            MERGE (act)-[:BELONGS_TO]->(f)
        ''', rows=batch)
    # Filter by family IS NOT NULL to exclude UC3 Activity nodes (which have no family)
    session.run('''
        MATCH (pr:Project {id:$pid}), (act:Activity)
        WHERE act.family IS NOT NULL
        MERGE (pr)-[:INCLUDES]->(act)
    ''', pid=project_id)
    print(f"  Activities: {len(acts)}")

    # Work Permits + Certifications
    for p in tx['work_permits']:
        session.run('MERGE (wp:WorkPermit {id:$id}) SET wp += $props',
                    id=p['id'], props=p)
    for c in tx['certifications']:
        session.run('MERGE (cert:Certification {id:$id}) SET cert += $props',
                    id=c['id'], props=c)
    print(f"  WorkPermits: {len(tx['work_permits'])}, Certifications: {len(tx['certifications'])}")

    # Permit → Cert relations (bitemporal)
    from datetime import datetime, timedelta, timezone
    PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
    RULE_CHANGE   = datetime(2024, 6, 29, tzinfo=timezone.utc)

    CERT_REQS = {
        'hot_work':       ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
        'excavation':     ['Excavation Safety', 'Confined Space Entry', 'Soil Assessment'],
        'lifting':        ['Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'],
        'electrical':     ['Electrical Safety', 'LOTO Certification', 'HV Awareness'],
        'confined_space': ['Confined Space Entry', 'Gas Testing', 'Emergency Response'],
        'radiography':    ['NDT Level II', 'Radiation Safety', 'RT Operator'],
        'work_at_height': ['Working at Height', 'Scaffold Inspection', 'Fall Arrest'],
        'general_work':   ['General Safety Induction', 'Site Orientation'],
    }
    for permit_id, certs in CERT_REQS.items():
        for cert_name in certs:
            cert_id = cert_name.replace(' ', '_')
            session.run('''
                MATCH (wp:WorkPermit {id:$pid}), (c:Certification {id:$cid})
                MERGE (wp)-[r:REQUIRES_CERT]->(c)
                SET r.valid_from=$vf, r.valid_to=null, r.tx_time=$tx
            ''', pid=permit_id, cid=cert_id,
                 vf=PROJECT_START.isoformat(),
                 tx=datetime.now(timezone.utc).isoformat())

    # Bitemporal rule-change at month 6: hot_work now requires Advanced Fire Watch
    session.run('''
        MATCH (wp:WorkPermit {id:'hot_work'}), (c:Certification {id:'Advanced_Fire_Watch'})
        MERGE (wp)-[r:REQUIRES_CERT]->(c)
        SET r.valid_from=$vf, r.valid_to=null, r.tx_time=$tx
    ''', vf=RULE_CHANGE.isoformat(),
         tx=datetime.now(timezone.utc).isoformat())
    print("  Permit->Cert relations + bitemporal rule change")

    # Steps — split into two passes to avoid REQUIRES_PERMIT silently failing
    steps = tx['steps']
    for s in steps:
        if s.get('weight_pct') != s.get('weight_pct'):  # NaN guard
            s['weight_pct'] = 0.0
        if s.get('estimated_hours') != s.get('estimated_hours'):
            s['estimated_hours'] = 0.0

    for i in range(0, len(steps), BATCH):
        batch = steps[i:i+BATCH]
        # Pass 1: create Step + HAS_STEP
        session.run('''
            UNWIND $rows AS s
            MERGE (step:Step {id:s.id})
            SET step += s
            WITH step, s
            MATCH (act:Activity {id:s.activity_id})
            MERGE (act)-[:HAS_STEP {order:s.order, weight_pct:s.weight_pct}]->(step)
        ''', rows=batch)
        # Pass 2: REQUIRES_PERMIT (separate so a missing WorkPermit never drops the step)
        session.run('''
            UNWIND $rows AS s
            MATCH (step:Step {id:s.id}), (wp:WorkPermit {id:s.permit_type})
            MERGE (step)-[:REQUIRES_PERMIT]->(wp)
        ''', rows=batch)
    print(f"  Steps: {len(steps)}")

    # PRECEDES relations
    seqs = tx['step_sequences']
    for i in range(0, len(seqs), BATCH):
        batch = seqs[i:i+BATCH]
        session.run('''
            UNWIND $rows AS r
            MATCH (s1:Step {id:r.from}), (s2:Step {id:r.to})
            MERGE (s1)-[:PRECEDES]->(s2)
        ''', rows=batch)
    print(f"  PRECEDES: {len(seqs)}")

    # Workers
    for w in tx['workers']:
        session.run('MERGE (worker:Worker {id:$id}) SET worker.name=$name, worker.discipline=$disc',
                    id=w['id'], name=w['name'], disc=w['discipline'])
        for cert in w['certifications']:
            cert_id = cert['cert'].replace(' ', '_')
            session.run('''
                MATCH (worker:Worker {id:$wid}), (c:Certification {id:$cid})
                MERGE (worker)-[r:HAS_CERT {valid_from:$vf, valid_to:$vt, tx_time:$tx}]->(c)
            ''', wid=w['id'], cid=cert_id,
                 vf=cert['valid_from'], vt=cert['valid_to'], tx=cert['tx_time'])
    print(f"  Workers: {len(tx['workers'])}")


if __name__ == '__main__':
    with open(DATA_FILE) as f:
        dataset = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        load(session, dataset)
    driver.close()
    print("\nUseCase4 TKG loaded into Neo4j")
