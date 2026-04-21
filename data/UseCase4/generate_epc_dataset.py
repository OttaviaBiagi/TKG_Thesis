"""
UseCase3 — EPC Planning & Control TKG Dataset Generator
Based on real TR Family Steps data (Family_Steps_macro.xlsm)

Entities:
  - Activity (from ACTIVITY column)
  - Family (from FAMILY column)  
  - Step (from STEP column)
  - WorkPermit (derived from step type)
  - Certification (required for work permit)
  - Worker (synthetic)
  - Project (synthetic EPC project)

Relations:
  - (Activity)-[HAS_STEP {order, weight_pct}]->(Step)
  - (Step)-[PRECEDES]->(Step)
  - (Step)-[REQUIRES_PERMIT]->(WorkPermit)
  - (WorkPermit)-[REQUIRES_CERT]->(Certification)
  - (Worker)-[HAS_CERT {valid_from, valid_to}]->(Certification)
  - (Worker)-[ASSIGNED_TO {valid_from, valid_to}]->(Step)
  - (Project)-[INCLUDES]->(Activity)
  - (Activity)-[BELONGS_TO]->(Family)

Bitemporal:
  - valid_from / valid_to: when true in real world
  - tx_time: when recorded in TKG
"""

import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

random.seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
INPUT_FILE  = Path(__file__).parent / 'Family_Steps_macro.xlsm'
OUTPUT_DIR  = Path(__file__).parent
PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)

DISCIPLINE_TIMELINE = {
    'SP': (0,  3),   # Site Preparation: months 0-3
    'CI': (2,  10),  # Civil: months 2-10
    'BU': (3,  10),  # Buildings
    'ST': (4,  12),  # Structural
    'ME': (6,  15),  # Mechanical
    'PI': (7,  16),  # Piping
    'EL': (9,  17),  # Electrical
    'IN': (11, 18),  # Instrumentation
    'PR': (5,  14),  # Pressure vessels
    'PE': (3,  11),  # Piling/Earthworks
    'CO': (2,  8),   # Concrete
    'MD': (6,  14),  # Mechanical misc
    'PA': (14, 18),  # Painting (last)
    'IS': (15, 18),  # Insulation (last)
    'FP': (13, 17),  # Fire protection
    'HV': (10, 16),  # HVAC
    'PL': (1,  5),   # Piling
}

WORK_PERMIT_RULES = {
    'hot_work':      ['weld', 'cutting', 'grinding', 'brazing', 'solder', 'torch', 'burn'],
    'excavation':    ['excavat', 'earth moving', 'grading', 'backfill', 'foundation', 'piling', 'pile'],
    'lifting':       ['erection', 'lifting', 'crane', 'module lift', 'rigging', 'hoisting'],
    'electrical':    ['electrical', 'cabling', 'termination', 'wiring', 'cable pull'],
    'confined_space':['underground', 'pit', 'sump', 'manhole', 'tank interior', 'vessel intern'],
    'radiography':   ['ndt', 'x-ray', 'radiograph', 'ultrasonic', 'static inspection'],
    'work_at_height':['scaffold', 'formwork', 'height', 'roof', 'tower'],
    'general_work':  [],  # fallback
}

CERT_REQUIREMENTS = {
    'hot_work':       ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    'excavation':     ['Excavation Safety', 'Confined Space Entry', 'Soil Assessment'],
    'lifting':        ['Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'],
    'electrical':     ['Electrical Safety', 'LOTO Certification', 'HV Awareness'],
    'confined_space': ['Confined Space Entry', 'Gas Testing', 'Emergency Response'],
    'radiography':    ['NDT Level II', 'Radiation Safety', 'RT Operator'],
    'work_at_height': ['Working at Height', 'Scaffold Inspection', 'Fall Arrest'],
    'general_work':   ['General Safety Induction', 'Site Orientation'],
}

DISCIPLINE_EXTRA_CERTS = {
    'ME': ['Mechanical Erection', 'Torque Certification'],
    'EL': ['Electrical Safety', 'LOTO Certification'],
    'PI': ['Welding Certification', 'Pressure Testing'],
    'CI': ['Concrete Works', 'Reinforcement Inspection'],
    'ST': ['Structural Steel', 'Working at Height'],
    'IN': ['Instrumentation Calibration', 'ATEX Awareness'],
    'PR': ['Pressure Vessel Inspection', 'ASME Awareness'],
}

# ─── Load real data ───────────────────────────────────────────────────────────
def load_family_steps(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name='Family Steps')
    df['discipline'] = df['CODE'].str.split('.').str[0]
    df = df[df['Active'] == True].copy()
    print(f"✅ Loaded {len(df)} steps across {df['ACTIVITY'].nunique()} activities")
    return df

# ─── Work permit classification ───────────────────────────────────────────────
def classify_permit(step_name: str) -> str:
    s = step_name.lower()
    for permit, keywords in WORK_PERMIT_RULES.items():
        if permit == 'general_work':
            continue
        if any(kw in s for kw in keywords):
            return permit
    return 'general_work'

# ─── Timestamp generation ─────────────────────────────────────────────────────
def get_step_timestamp(discipline: str, step_order: int, total_steps: int, weight_pct: float) -> datetime:
    start_m, end_m = DISCIPLINE_TIMELINE.get(discipline, (6, 12))
    duration_days = (end_m - start_m) * 30
    progress = (step_order - 1) / max(total_steps - 1, 1)
    offset_days = int(start_m * 30 + progress * duration_days)
    jitter = random.randint(-3, 3)
    return PROJECT_START + timedelta(days=offset_days + jitter)

def add_bitemporality(valid_from: datetime, duration_days: int = 30) -> dict:
    tx_time = datetime.now(timezone.utc)
    return {
        'valid_from': valid_from.isoformat(),
        'valid_to':   (valid_from + timedelta(days=duration_days)).isoformat(),
        'tx_time':    tx_time.isoformat(),
    }

# ─── Generate workers ─────────────────────────────────────────────────────────
# First 10 workers have guaranteed cert sets to make bitemporal queries meaningful:
#   W-001..W-006: qualified for hot_work BEFORE rule change (3 original certs)
#   W-007..W-010: qualified AFTER rule change too (have Advanced Fire Watch as well)
#   W-011..W-050: random certs as before
GUARANTEED_SETS = [
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification', 'Advanced Fire Watch'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification', 'Advanced Fire Watch'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification', 'Advanced Fire Watch'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification', 'Advanced Fire Watch'],
]

def generate_workers(all_certs: set, n: int = 50) -> list:
    workers = []
    cert_list = list(all_certs)
    tx_now = datetime.now(timezone.utc)

    for i in range(n):
        discipline = random.choice(list(DISCIPLINE_TIMELINE.keys()))
        cert_valid_from = PROJECT_START - timedelta(days=random.randint(30, 365))

        if i < len(GUARANTEED_SETS):
            certs = GUARANTEED_SETS[i]
        else:
            n_certs = random.randint(1, 4)
            certs = random.sample(cert_list, min(n_certs, len(cert_list)))

        workers.append({
            'id':         f'W-{i+1:03d}',
            'name':       f'Worker_{i+1}',
            'discipline': discipline,
            'certifications': [{
                'cert': c,
                'valid_from': cert_valid_from.isoformat(),
                'valid_to':   (cert_valid_from + timedelta(days=random.choice([365, 730, 1095]))).isoformat(),
                'tx_time':    tx_now.isoformat(),
            } for c in certs]
        })
    return workers

# ─── Main pipeline ────────────────────────────────────────────────────────────
def generate_epc_dataset(df: pd.DataFrame) -> dict:
    tx_now = datetime.now(timezone.utc).isoformat()
    
    activities, steps, families = [], [], []
    work_permits, certifications = {}, {}
    step_sequences, activity_steps = [], []
    
    all_certs_needed = set()
    
    for activity_name, group in df.groupby('ACTIVITY'):
        group = group.sort_values('#')
        discipline = group['discipline'].iloc[0]
        code = group['CODE'].iloc[0]
        family = group['FAMILY'].iloc[0]
        total_steps = len(group)
        
        start_m, end_m = DISCIPLINE_TIMELINE.get(discipline, (6, 12))
        act_start = PROJECT_START + timedelta(days=start_m * 30)
        act_end   = PROJECT_START + timedelta(days=end_m * 30)
        
        activities.append({
            'id':         code,
            'name':       activity_name,
            'family':     family,
            'discipline': discipline,
            'valid_from': act_start.isoformat(),
            'valid_to':   act_end.isoformat(),
            'tx_time':    tx_now,
        })
        
        if family not in [f['id'] for f in families]:
            families.append({'id': family, 'name': activity_name, 'discipline': discipline, 'tx_time': tx_now})
        
        prev_step_id = None
        for _, row in group.iterrows():
            step_name = str(row['STEP']).strip()
            step_id   = f"{code}_S{int(row['#']):02d}"
            permit_type = classify_permit(step_name)
            ts = get_step_timestamp(discipline, int(row['#']), total_steps, float(row['%'] or 0))
            
            steps.append({
                'id':          step_id,
                'name':        step_name,
                'order':       int(row['#']),
                'weight_pct':  float(row['%'] or 0),
                'permit_type': permit_type,
                'activity_id': code,
                'discipline':  discipline,
                'valid_from':  ts.isoformat(),
                'valid_to':    (ts + timedelta(days=14)).isoformat(),
                'tx_time':     tx_now,
            })
            
            activity_steps.append({'activity_id': code, 'step_id': step_id, 'order': int(row['#']), 'weight_pct': float(row['%'] or 0)})
            
            if permit_type not in work_permits:
                work_permits[permit_type] = {
                    'id': permit_type,
                    'name': permit_type.replace('_', ' ').title(),
                    'tx_time': tx_now,
                }
                for cert_name in CERT_REQUIREMENTS.get(permit_type, []):
                    all_certs_needed.add(cert_name)
                    if cert_name not in certifications:
                        certifications[cert_name] = {'id': cert_name.replace(' ', '_'), 'name': cert_name, 'tx_time': tx_now}
            
            if prev_step_id:
                step_sequences.append({'from': prev_step_id, 'to': step_id, 'activity_id': code})
            prev_step_id = step_id
    
    # Add discipline-specific certs
    for disc, certs in DISCIPLINE_EXTRA_CERTS.items():
        for c in certs:
            all_certs_needed.add(c)
            if c not in certifications:
                certifications[c] = {'id': c.replace(' ', '_'), 'name': c, 'tx_time': tx_now}
    
    workers = generate_workers(all_certs_needed, n=50)
    
    # Synthetic project
    project = {
        'id':         'PROJ-001',
        'name':       'TR Refinery Expansion Project',
        'start_date': PROJECT_START.isoformat(),
        'end_date':   (PROJECT_START + timedelta(days=18*30)).isoformat(),
        'disciplines': list(DISCIPLINE_TIMELINE.keys()),
        'tx_time':    tx_now,
    }
    
    # Update scenario: simulate a rule change at month 6
    update_event = {
        'type':        'permit_rule_change',
        'description': 'Hot work permit now requires additional Fire Watch certification',
        'valid_from':  (PROJECT_START + timedelta(days=6*30)).isoformat(),
        'tx_time':     tx_now,
        'affected':    'hot_work',
        'new_cert':    'Advanced Fire Watch',
    }
    all_certs_needed.add('Advanced Fire Watch')
    certifications['Advanced Fire Watch'] = {'id': 'Advanced_Fire_Watch', 'name': 'Advanced Fire Watch', 'tx_time': (PROJECT_START + timedelta(days=6*30)).isoformat()}
    
    return {
        'project':        project,
        'activities':     activities,
        'families':       families,
        'steps':          steps,
        'activity_steps': activity_steps,
        'step_sequences': step_sequences,
        'work_permits':   list(work_permits.values()),
        'certifications': list(certifications.values()),
        'workers':        workers,
        'update_events':  [update_event],
        'metadata': {
            'generated_at':     tx_now,
            'source':           'TR Family_Steps_macro.xlsm',
            'total_activities': len(activities),
            'total_steps':      len(steps),
            'total_workers':    len(workers),
            'total_permits':    len(work_permits),
            'total_certs':      len(certifications),
        }
    }

# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_family_steps(INPUT_FILE)
    dataset = generate_epc_dataset(df)
    
    out = OUTPUT_DIR / 'epc_dataset_real.json'
    with open(out, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    m = dataset['metadata']
    print(f"\n✅ Dataset generated: {out}")
    print(f"   Activities:   {m['total_activities']}")
    print(f"   Steps:        {m['total_steps']}")
    print(f"   Workers:      {m['total_workers']}")
    print(f"   Work permits: {m['total_permits']}")
    print(f"   Certs:        {m['total_certs']}")
    print(f"\n📋 Update event: hot_work permit rule change at month 6")
    print(f"   → This tests bitemporal querying!")
