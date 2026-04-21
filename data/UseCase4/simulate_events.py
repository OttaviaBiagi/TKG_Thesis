"""
UseCase4 — EPC Dynamic TKG Event Simulator

Generates synthetic temporal events on top of the static EPC graph:
  - ASSIGNED_TO  : worker assigned to a step (cert-compatible)
  - COMPLETED    : step completed (planned vs actual date, delay)
  - PERMIT_DENIED: worker blocked due to missing cert after rule change

Delay propagation: if step S is delayed N days, all downstream
steps (via PRECEDES) inherit at least that delay.

Output: events saved to epc_events.json + imported into Neo4j.
"""

import json
import random
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from neo4j import GraphDatabase

random.seed(42)

DATA_FILE   = Path(__file__).parent / 'epc_dataset_real.json'
OUTPUT_FILE = Path(__file__).parent / 'epc_events.json'

NEO4J_URI      = 'bolt://172.22.43.151:7687'
NEO4J_USER     = 'neo4j'
NEO4J_PASSWORD = 'your_password'

RULE_CHANGE = datetime(2024, 6, 29, tzinfo=timezone.utc)

CERT_REQS = {
    'hot_work':       {'Hot_Work_Safety', 'Fire_Watch', 'Welding_Certification'},
    'excavation':     {'Excavation_Safety', 'Confined_Space_Entry', 'Soil_Assessment'},
    'lifting':        {'Rigging_&_Lifting', 'Crane_Operator', 'Slinging_Certificate'},
    'electrical':     {'Electrical_Safety', 'LOTO_Certification', 'HV_Awareness'},
    'confined_space': {'Confined_Space_Entry', 'Gas_Testing', 'Emergency_Response'},
    'radiography':    {'NDT_Level_II', 'Radiation_Safety', 'RT_Operator'},
    'work_at_height': {'Working_at_Height', 'Scaffold_Inspection', 'Fall_Arrest'},
    'general_work':   {'General_Safety_Induction', 'Site_Orientation'},
}

HOT_WORK_NEW_CERT = 'Advanced_Fire_Watch'

# Steps on critical path (ME.CT activity) get higher delay probability
CRITICAL_ACTIVITIES = {'ME.CT', 'CI.TF1', 'ME.TK3', 'BU.PC.AR', 'CI.DM2'}
HIGH_VARIANCE_DISCIPLINES = {'ME', 'CI', 'PI', 'ST'}


def load_dataset():
    with open(DATA_FILE) as f:
        return json.load(f)


def build_worker_cert_index(workers):
    """Returns {worker_id: {cert_id: (valid_from, valid_to)}}"""
    index = {}
    for w in workers:
        certs = {}
        for c in w['certifications']:
            cid = c['cert'].replace(' ', '_')
            certs[cid] = (c['valid_from'], c['valid_to'])
        index[w['id']] = {'discipline': w['discipline'], 'certs': certs}
    return index


def worker_has_certs(worker_id, permit_type, step_date_str, worker_index, after_rule_change=False):
    """Check if worker holds all required certs for a permit type on a given date."""
    w = worker_index.get(worker_id)
    if not w:
        return False
    required = set(CERT_REQS.get(permit_type, set()))
    if after_rule_change and permit_type == 'hot_work':
        required.add(HOT_WORK_NEW_CERT)
    step_date = datetime.fromisoformat(step_date_str)
    for cert_id in required:
        if cert_id not in w['certs']:
            return False
        vf, vt = w['certs'][cert_id]
        if datetime.fromisoformat(vf) > step_date:
            return False
        if datetime.fromisoformat(vt) < step_date:
            return False
    return True


def assign_workers(steps, workers, worker_index):
    """Assign a compatible worker to each step. Returns {step_id: worker_id}."""
    assignments = {}
    worker_ids = [w['id'] for w in workers]

    for step in steps:
        step_date = step['valid_from']
        after_rc = datetime.fromisoformat(step_date) >= RULE_CHANGE
        permit = step['permit_type']

        candidates = [
            wid for wid in worker_ids
            if worker_has_certs(wid, permit, step_date, worker_index, after_rc)
        ]

        if candidates:
            assignments[step['id']] = random.choice(candidates)
        else:
            # No fully qualified worker — assign one anyway (will trigger PERMIT_DENIED)
            assignments[step['id']] = random.choice(worker_ids)

    return assignments


def simulate_delays(steps, step_sequences):
    """
    Simulate delays with propagation through PRECEDES.
    Returns {step_id: delay_days}
    """
    # Base delay probability
    def delay_prob(step):
        if step['activity_id'] in CRITICAL_ACTIVITIES:
            return 0.50
        disc = step.get('discipline', step['activity_id'].split('.')[0])
        if disc in HIGH_VARIANCE_DISCIPLINES:
            return 0.30
        return 0.15

    def max_delay_days(step):
        if step['activity_id'] in CRITICAL_ACTIVITIES:
            return 21
        return 10

    step_by_id = {s['id']: s for s in steps}
    delays = {}

    # Assign base delays
    for step in steps:
        if random.random() < delay_prob(step):
            delays[step['id']] = random.randint(1, max_delay_days(step))
        else:
            delays[step['id']] = 0

    # Propagate: build adjacency list
    successors = defaultdict(list)
    for seq in step_sequences:
        successors[seq['from']].append(seq['to'])

    # Topological propagation (simple BFS)
    changed = True
    iterations = 0
    while changed and iterations < 20:
        changed = False
        iterations += 1
        for step_id, delay in list(delays.items()):
            if delay > 0:
                for succ in successors.get(step_id, []):
                    if delays.get(succ, 0) < delay:
                        delays[succ] = delay
                        changed = True

    return delays


def generate_events(steps, workers, assignments, delays, worker_index):
    """Generate ASSIGNED_TO, COMPLETED, and PERMIT_DENIED events."""
    tx_now = datetime.now(timezone.utc).isoformat()
    events = {'assigned_to': [], 'completed': [], 'permit_denied': []}

    for step in steps:
        sid = step['id']
        wid = assignments.get(sid)
        planned = datetime.fromisoformat(step['valid_from'])
        delay = delays.get(sid, 0)
        actual = planned + timedelta(days=delay)
        after_rc = planned >= RULE_CHANGE
        permit = step['permit_type']

        # ASSIGNED_TO event
        assign_date = planned - timedelta(days=random.randint(1, 7))
        events['assigned_to'].append({
            'worker_id':  wid,
            'step_id':    sid,
            'date':       assign_date.isoformat(),
            'permit_type': permit,
            'tx_time':    tx_now,
        })

        # COMPLETED event
        status = 'delayed' if delay > 0 else 'on_time'
        events['completed'].append({
            'step_id':      sid,
            'worker_id':    wid,
            'planned_date': planned.isoformat(),
            'actual_date':  actual.isoformat(),
            'delay_days':   delay,
            'status':       status,
            'tx_time':      tx_now,
        })

        # PERMIT_DENIED: worker lacks required certs at step execution date
        if not worker_has_certs(wid, permit, step['valid_from'], worker_index, after_rc):
            missing = set(CERT_REQS.get(permit, set()))
            if after_rc and permit == 'hot_work':
                missing.add(HOT_WORK_NEW_CERT)
            w_certs = set(worker_index.get(wid, {}).get('certs', {}).keys())
            missing -= w_certs
            events['permit_denied'].append({
                'worker_id':   wid,
                'step_id':     sid,
                'date':        assign_date.isoformat(),
                'permit_type': permit,
                'missing_certs': list(missing),
                'after_rule_change': after_rc,
                'tx_time':     tx_now,
            })

    return events


def import_to_neo4j(events, driver):
    with driver.session() as s:
        # ASSIGNED_TO
        for e in events['assigned_to']:
            s.run('''
                MATCH (w:Worker {id:$wid}), (step:Step {id:$sid})
                MERGE (w)-[r:ASSIGNED_TO {date:$date}]->(step)
                SET r.permit_type=$permit, r.tx_time=$tx
            ''', wid=e['worker_id'], sid=e['step_id'],
                 date=e['date'], permit=e['permit_type'], tx=e['tx_time'])

        print(f'✅ ASSIGNED_TO: {len(events["assigned_to"])}')

        # COMPLETED — store as properties on Step node
        for e in events['completed']:
            s.run('''
                MATCH (step:Step {id:$sid})
                SET step.planned_date=$planned,
                    step.actual_date=$actual,
                    step.delay_days=$delay,
                    step.status=$status
            ''', sid=e['step_id'], planned=e['planned_date'],
                 actual=e['actual_date'], delay=e['delay_days'], status=e['status'])

        n_delayed = sum(1 for e in events['completed'] if e['delay_days'] > 0)
        print(f'✅ COMPLETED: {len(events["completed"])} steps '
              f'({n_delayed} delayed, {len(events["completed"])-n_delayed} on time)')

        # PERMIT_DENIED
        for e in events['permit_denied']:
            s.run('''
                MATCH (w:Worker {id:$wid}), (step:Step {id:$sid})
                MERGE (w)-[r:PERMIT_DENIED {date:$date}]->(step)
                SET r.permit_type=$permit,
                    r.missing_certs=$missing,
                    r.after_rule_change=$arc,
                    r.tx_time=$tx
            ''', wid=e['worker_id'], sid=e['step_id'],
                 date=e['date'], permit=e['permit_type'],
                 missing=e['missing_certs'],
                 arc=e['after_rule_change'], tx=e['tx_time'])

        print(f'✅ PERMIT_DENIED: {len(events["permit_denied"])}')


if __name__ == '__main__':
    print('Loading dataset...')
    d = load_dataset()

    worker_index = build_worker_cert_index(d['workers'])
    assignments  = assign_workers(d['steps'], d['workers'], worker_index)
    delays       = simulate_delays(d['steps'], d['step_sequences'])
    events       = generate_events(d['steps'], d['workers'], assignments, delays, worker_index)

    # Summary
    n_delayed = sum(1 for v in delays.values() if v > 0)
    print(f'\n📊 Simulation summary:')
    print(f'   Steps total:    {len(d["steps"])}')
    print(f'   Steps delayed:  {n_delayed} ({n_delayed/len(d["steps"])*100:.1f}%)')
    print(f'   PERMIT_DENIED:  {len(events["permit_denied"])}')
    print(f'   Avg delay:      {sum(delays.values())/len(delays):.1f} days')

    # Save to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(events, f, indent=2)
    print(f'\n✅ Events saved to {OUTPUT_FILE}')

    # Import to Neo4j
    print('\nImporting to Neo4j...')
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    import_to_neo4j(events, driver)
    driver.close()
    print('\n🎉 Dynamic TKG events loaded into Neo4j!')
