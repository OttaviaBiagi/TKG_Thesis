"""
UseCase3 — Import EPC TKG into Neo4j
Reads epc_dataset_real.json and loads all nodes + relations
"""
import json
from pathlib import Path
from neo4j import GraphDatabase

NEO4J_URI      = 'bolt://172.22.43.151:7687'
NEO4J_USER     = 'neo4j'
NEO4J_PASSWORD = 'your_password'
DATA_FILE      = Path(__file__).parent / 'epc_dataset_real.json'

def load(session, dataset):
    tx = dataset

    # Project
    p = tx['project']
    session.run('MERGE (pr:Project {id:$id}) SET pr += $props',
                id=p['id'], props=p)
    print(f"✅ Project: {p['name']}")

    # Families
    for f in tx['families']:
        session.run('MERGE (f:Family {id:$id}) SET f += $props',
                    id=f['id'], props=f)
    print(f"✅ Families: {len(tx['families'])}")

    # Activities
    for a in tx['activities']:
        session.run('''
            MERGE (act:Activity {id:$id}) SET act += $props
            WITH act
            MATCH (f:Family {id:$fam})
            MERGE (act)-[:BELONGS_TO]->(f)
            WITH act
            MATCH (pr:Project {id:'PROJ-001'})
            MERGE (pr)-[:INCLUDES]->(act)
        ''', id=a['id'], props=a, fam=a['family'])
    print(f"✅ Activities: {len(tx['activities'])}")

    # Work Permits + Certifications
    for p in tx['work_permits']:
        session.run('MERGE (wp:WorkPermit {id:$id}) SET wp += $props',
                    id=p['id'], props=p)
    for c in tx['certifications']:
        session.run('MERGE (cert:Certification {id:$id}) SET cert += $props',
                    id=c['id'], props=c)
    print(f"✅ WorkPermits: {len(tx['work_permits'])}, Certifications: {len(tx['certifications'])}")

    # Permit → Cert relations (with bitemporality)
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
                MERGE (wp)-[r:REQUIRES_CERT {valid_from:$vf, valid_to:null, tx_time:$tx}]->(c)
            ''', pid=permit_id, cid=cert_id,
                 vf=PROJECT_START.isoformat(),
                 tx=datetime.now(timezone.utc).isoformat())

    # Update event: new cert for hot_work after month 6
    session.run('''
        MATCH (wp:WorkPermit {id:'hot_work'}), (c:Certification {id:'Advanced_Fire_Watch'})
        MERGE (wp)-[r:REQUIRES_CERT {valid_from:$vf, valid_to:null, tx_time:$tx}]->(c)
    ''', vf=RULE_CHANGE.isoformat(),
         tx=datetime.now(timezone.utc).isoformat())
    print("✅ Permit→Cert relations + bitemporal rule change")

    # Steps (batch)
    BATCH = 200
    steps = tx['steps']
    for i in range(0, len(steps), BATCH):
        batch = steps[i:i+BATCH]
        session.run('''
            UNWIND $rows AS s
            MERGE (step:Step {id:s.id})
            SET step += s
            WITH step, s
            MATCH (act:Activity {id:s.activity_id})
            MERGE (act)-[:HAS_STEP {order:s.order, weight_pct:s.weight_pct}]->(step)
            WITH step, s
            MATCH (wp:WorkPermit {id:s.permit_type})
            MERGE (step)-[:REQUIRES_PERMIT]->(wp)
        ''', rows=batch)
    print(f"✅ Steps: {len(steps)}")

    # PRECEDES relations
    for seq in tx['step_sequences']:
        session.run('''
            MATCH (s1:Step {id:$from}), (s2:Step {id:$to})
            MERGE (s1)-[:PRECEDES]->(s2)
        ''', **seq)
    print(f"✅ PRECEDES relations: {len(tx['step_sequences'])}")

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
    print(f"✅ Workers: {len(tx['workers'])}")


if __name__ == '__main__':
    with open(DATA_FILE) as f:
        dataset = json.load(f)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as session:
        load(session, dataset)
    driver.close()
    print("\n🎉 UseCase3 TKG loaded into Neo4j!")
