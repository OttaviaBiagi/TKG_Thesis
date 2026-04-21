"""
UseCase4 — EPC Dynamic TKG Event Simulator

Generates synthetic temporal events on top of the static EPC graph:
  - ASSIGNED_TO  : worker assigned to a step (cert-compatible, 95% correct)
  - COMPLETED    : step completed with planned vs actual date and delay
  - PERMIT_DENIED: violation event (~5-8% of assignments)

Violation sources:
  1. Post-rule-change hot_work: workers W-001..W-006 lack Advanced Fire Watch
  2. Human error: 5% chance of wrong assignment on specialized permits

Delay propagation: a delay on step S cascades to all downstream PRECEDES steps.

Output: epc_events.json + Neo4j import.
"""

import json
import random
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

# general_work: all workers qualify (basic site induction, given on day 1)
CERT_REQS = {
    'hot_work':       {'Hot_Work_Safety', 'Fire_Watch', 'Welding_Certification'},
    'excavation':     {'Excavation_Safety', 'Confined_Space_Entry', 'Soil_Assessment'},
    'lifting':        {'Rigging_&_Lifting', 'Crane_Operator', 'Slinging_Certificate'},
    'electrical':     {'Electrical_Safety', 'LOTO_Certification', 'HV_Awareness'},
    'confined_space': {'Confined_Space_Entry', 'Gas_Testing', 'Emergency_Response'},
    'radiography':    {'NDT_Level_II', 'Radiation_Safety', 'RT_Operator'},
    'work_at_height': {'Working_at_Height', 'Scaffold_Inspection', 'Fall_Arrest'},
    'general_work':   set(),
}

HOT_WORK_NEW_CERT   = 'Advanced_Fire_Watch'
HUMAN_ERROR_RATE    = 0.05   # 5% wrong assignment on specialized permits
CRITICAL_ACTIVITIES = {'ME.CT', 'CI.TF1', 'ME.TK3', 'BU.PC.AR', 'CI.DM2'}
HIGH_VARIANCE_DISC  = {'ME', 'CI', 'PI', 'ST'}


def load_dataset():
    with open(DATA_FILE) as f:
        return json.load(f)


def build_worker_cert_index(workers):
    index = {}
    for w in workers:
        certs = {}
        for c in w['certifications']:
            cid = c['cert'].replace(' ', '_')
            certs[cid] = (c['valid_from'], c['valid_to'])
        index[w['id']] = {'discipline': w['discipline'], 'certs': certs}
    return index


def worker_qualifies(worker_id, permit_type, step_date_str, worker_index, after_rc=False):
    """True if worker holds all required (and current) certs for permit on step_date."""
    if permit_type == 'general_work':
        return True
    w = worker_index.get(worker_id)
    if not w:
        return False
    required = set(CERT_REQS.get(permit_type, set()))
    if after_rc and permit_type == 'hot_work':
        required.add(HOT_WORK_NEW_CERT)
    if not required:
        return True
    step_date = datetime.fromisoformat(step_date_str)
    for cid in required:
        if cid not in w['certs']:
            return False
        vf, vt = w['certs'][cid]
        if datetime.fromisoformat(vf) > step_date:
            return False
        if datetime.fromisoformat(vt) < step_date:
            return False
    return True


def assign_workers(steps, workers, worker_index):
    """
    Smart assignment:
    - Always prefer a qualified worker.
    - 5% human error: assign a non-qualified worker on specialized permits.
    Returns {step_id: (worker_id, is_violation)}
    """
    assignments = {}
    worker_ids  = [w['id'] for w in workers]

    for step in steps:
        sid        = step['id']
        step_date  = step['valid_from']
        after_rc   = datetime.fromisoformat(step_date) >= RULE_CHANGE
        permit     = step['permit_type']

        qualified   = [wid for wid in worker_ids
                       if worker_qualifies(wid, permit, step_date, worker_index, after_rc)]
        unqualified = [wid for wid in worker_ids if wid not in qualified]

        # Simulate human error on specialized permits only
        is_violation = False
        if permit != 'general_work' and unqualified and random.random() < HUMAN_ERROR_RATE:
            chosen       = random.choice(unqualified)
            is_violation = True
        elif qualified:
            chosen = random.choice(qualified)
        else:
            # No qualified worker exists (e.g. post-rule-change hot_work with no Advanced FW)
            chosen       = random.choice(worker_ids)
            is_violation = True

        assignments[sid] = (chosen, is_violation)

    return assignments


def simulate_delays(steps, step_sequences):
    """Returns {step_id: delay_days} with cascade propagation."""
    def delay_prob(step):
        if step['activity_id'] in CRITICAL_ACTIVITIES:
            return 0.50
        disc = step['activity_id'].split('.')[0]
        return 0.30 if disc in HIGH_VARIANCE_DISC else 0.15

    def max_delay(step):
        return 21 if step['activity_id'] in CRITICAL_ACTIVITIES else 10

    delays = {}
    for step in steps:
        delays[step['id']] = random.randint(1, max_delay(step)) \
            if random.random() < delay_prob(step) else 0

    # Propagate downstream
    successors = defaultdict(list)
    for seq in step_sequences:
        successors[seq['from']].append(seq['to'])

    changed, iters = True, 0
    while changed and iters < 20:
        changed = False
        iters  += 1
        for sid, delay in list(delays.items()):
            if delay > 0:
                for succ in successors.get(sid, []):
                    if delays.get(succ, 0) < delay:
                        delays[succ] = delay
                        changed = True

    return delays


def get_missing_certs(worker_id, permit_type, worker_index, after_rc):
    required = set(CERT_REQS.get(permit_type, set()))
    if after_rc and permit_type == 'hot_work':
        required.add(HOT_WORK_NEW_CERT)
    held = set(worker_index.get(worker_id, {}).get('certs', {}).keys())
    return list(required - held)


def generate_events(steps, workers, assignments, delays, worker_index):
    tx_now = datetime.now(timezone.utc).isoformat()
    events = {'assigned_to': [], 'completed': [], 'permit_denied': []}

    for step in steps:
        sid       = step['id']
        wid, is_v = assignments[sid]
        planned   = datetime.fromisoformat(step['valid_from'])
        delay     = delays.get(sid, 0)
        actual    = planned + timedelta(days=delay)
        after_rc  = planned >= RULE_CHANGE
        permit    = step['permit_type']
        disc      = step['activity_id'].split('.')[0]
        on_cp     = step['activity_id'] in CRITICAL_ACTIVITIES
        assign_dt = planned - timedelta(days=random.randint(1, 7))

        events['assigned_to'].append({
            'worker_id':      wid,
            'step_id':        sid,
            'date':           assign_dt.isoformat(),
            'permit_type':    permit,
            'discipline':     disc,
            'on_critical_path': on_cp,
            'weight_pct':     step.get('weight_pct', 0),
            'tx_time':        tx_now,
        })

        events['completed'].append({
            'step_id':       sid,
            'worker_id':     wid,
            'planned_date':  planned.isoformat(),
            'actual_date':   actual.isoformat(),
            'delay_days':    delay,
            'status':        'delayed' if delay > 0 else 'on_time',
            'discipline':    disc,
            'on_critical_path': on_cp,
            'weight_pct':    step.get('weight_pct', 0),
            'tx_time':       tx_now,
        })

        if is_v:
            missing = get_missing_certs(wid, permit, worker_index, after_rc)
            events['permit_denied'].append({
                'worker_id':        wid,
                'step_id':          sid,
                'date':             assign_dt.isoformat(),
                'permit_type':      permit,
                'missing_certs':    missing,
                'after_rule_change': after_rc,
                'tx_time':          tx_now,
            })

    return events


def clear_dynamic_relations(driver):
    """Remove previous simulation data from Neo4j."""
    with driver.session() as s:
        s.run('MATCH ()-[r:ASSIGNED_TO]->() DELETE r')
        s.run('MATCH ()-[r:PERMIT_DENIED]->() DELETE r')
        s.run('''MATCH (step:Step)
                 REMOVE step.planned_date, step.actual_date,
                        step.delay_days, step.status''')
    print('✅ Previous simulation cleared')


def import_to_neo4j(events, driver):
    with driver.session() as s:
        for e in events['assigned_to']:
            s.run('''
                MATCH (w:Worker {id:$wid}), (step:Step {id:$sid})
                MERGE (w)-[r:ASSIGNED_TO {date:$date}]->(step)
                SET r.permit_type=$permit, r.discipline=$disc,
                    r.on_critical_path=$cp, r.weight_pct=$wp, r.tx_time=$tx
            ''', wid=e['worker_id'], sid=e['step_id'], date=e['date'],
                 permit=e['permit_type'], disc=e['discipline'],
                 cp=e['on_critical_path'], wp=e['weight_pct'], tx=e['tx_time'])
        print(f'✅ ASSIGNED_TO: {len(events["assigned_to"])}')

        for e in events['completed']:
            s.run('''
                MATCH (step:Step {id:$sid})
                SET step.planned_date=$planned, step.actual_date=$actual,
                    step.delay_days=$delay, step.status=$status,
                    step.on_critical_path=$cp, step.weight_pct=$wp
            ''', sid=e['step_id'], planned=e['planned_date'], actual=e['actual_date'],
                 delay=e['delay_days'], status=e['status'],
                 cp=e['on_critical_path'], wp=e['weight_pct'])

        n_delayed = sum(1 for e in events['completed'] if e['delay_days'] > 0)
        print(f'✅ COMPLETED: {len(events["completed"])} '
              f'({n_delayed} delayed, {len(events["completed"])-n_delayed} on time)')

        for e in events['permit_denied']:
            s.run('''
                MATCH (w:Worker {id:$wid}), (step:Step {id:$sid})
                MERGE (w)-[r:PERMIT_DENIED {date:$date}]->(step)
                SET r.permit_type=$permit, r.missing_certs=$missing,
                    r.after_rule_change=$arc, r.tx_time=$tx
            ''', wid=e['worker_id'], sid=e['step_id'], date=e['date'],
                 permit=e['permit_type'], missing=e['missing_certs'],
                 arc=e['after_rule_change'], tx=e['tx_time'])
        print(f'✅ PERMIT_DENIED: {len(events["permit_denied"])}')


if __name__ == '__main__':
    print('Loading dataset...')
    d = load_dataset()

    worker_index = build_worker_cert_index(d['workers'])
    assignments  = assign_workers(d['steps'], d['workers'], worker_index)
    delays       = simulate_delays(d['steps'], d['step_sequences'])
    events       = generate_events(d['steps'], d['workers'], assignments, delays, worker_index)

    n_delayed    = sum(1 for v in delays.values() if v > 0)
    n_violations = len(events['permit_denied'])
    n_total      = len(d['steps'])

    print(f'\n📊 Simulation summary:')
    print(f'   Steps total:        {n_total}')
    print(f'   Steps delayed:      {n_delayed} ({n_delayed/n_total*100:.1f}%)')
    print(f'   PERMIT_DENIED:      {n_violations} ({n_violations/n_total*100:.1f}%)')
    print(f'   Avg delay:          {sum(delays.values())/len(delays):.1f} days')

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(events, f, indent=2)
    print(f'\n✅ Events saved to {OUTPUT_FILE}')

    print('\nImporting to Neo4j...')
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    clear_dynamic_relations(driver)
    import_to_neo4j(events, driver)
    driver.close()
    print('\n🎉 Dynamic TKG ready!')
