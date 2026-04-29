"""
UseCase4 — EPC Planning & Control TKG Dataset Generator
Data sources:
  - REAL: data/UseCase4/meram/Meram_PCS_Progress.xlsx  (8,762 TR Meram activities)
  - REAL: Family_Steps_macro.xlsm (step templates per family)
  - SYNTHETIC: Workers, Certifications, HSE events (sensitive real data unavailable)

Entities:
  - Activity  (from Meram: real ActID, ActivityName, Disc, Fami, Area, CWP, hours)
  - Family    (from Family Steps: step template families)
  - Step      (from Family Steps templates, instantiated per Meram activity)
  - WorkPermit, Certification (derived from step keywords)
  - Worker    (synthetic — 50 workers)
  - Project   (synthetic wrapper — TR Refinery Expansion)

Relations:
  - (Activity)-[HAS_STEP {order, weight_pct}]->(Step)
  - (Step)-[PRECEDES]->(Step)
  - (Step)-[REQUIRES_PERMIT]->(WorkPermit)
  - (WorkPermit)-[REQUIRES_CERT]->(Certification)
  - (Worker)-[HAS_CERT {valid_from, valid_to}]->(Certification)
  - (Worker)-[ASSIGNED_TO {valid_from, valid_to}]->(Step)
  - (Project)-[INCLUDES]->(Activity)
  - (Activity)-[BELONGS_TO]->(Family)
"""

import pandas as pd
import json
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone

random.seed(42)

# ─── Config ──────────────────────────────────────────────────────────────────
FAMILY_STEPS_FILE = Path(
    r'C:\Users\obiagi\OneDrive - Tecnicas Reunidas, S.A\Documents\Family Steps macro.xlsm'
)
MERAM_FILE  = Path(__file__).parent / 'meram' / 'Meram_PCS_Progress.xlsx'
OUTPUT_DIR  = Path(__file__).parent
PROJECT_START = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Estimated discipline execution windows (months from project start)
DISCIPLINE_TIMELINE = {
    'SP': (0,  3),
    'CI': (2,  10),
    'BU': (3,  10),
    'ST': (4,  12),
    'ME': (6,  15),
    'PI': (7,  16),
    'EL': (9,  17),
    'IN': (11, 18),
    'PR': (5,  14),
    'PE': (3,  11),
    'CO': (2,  8),
    'MD': (6,  14),
    'PA': (14, 18),
    'IS': (15, 18),
    'FP': (13, 17),
    'HV': (10, 16),
    'PL': (1,  5),
}

WORK_PERMIT_RULES = {
    'hot_work':       ['weld', 'cutting', 'grinding', 'brazing', 'solder', 'torch', 'burn'],
    'excavation':     ['excavat', 'earth moving', 'grading', 'backfill', 'foundation', 'piling', 'pile'],
    'lifting':        ['erection', 'lifting', 'crane', 'module lift', 'rigging', 'hoisting'],
    'electrical':     ['electrical', 'cabling', 'termination', 'wiring', 'cable pull'],
    'confined_space': ['underground', 'pit', 'sump', 'manhole', 'tank interior', 'vessel intern'],
    'radiography':    ['ndt', 'x-ray', 'radiograph', 'ultrasonic', 'static inspection'],
    'work_at_height': ['scaffold', 'formwork', 'height', 'roof', 'tower'],
    'general_work':   [],
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

# Fallback steps for Meram activities whose Fami code has no match in Family Steps
DEFAULT_STEPS = {
    'default': [
        ('Preparation & Mobilisation', 15.0),
        ('Main Execution',             75.0),
        ('QA/QC & Handover',           10.0),
    ]
}

# ─── Guaranteed worker cert sets (synthetic) ─────────────────────────────────
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
    ['Excavation Safety', 'Confined Space Entry', 'Soil Assessment'],
    ['Excavation Safety', 'Confined Space Entry', 'Soil Assessment'],
    ['Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'],
    ['Rigging & Lifting', 'Crane Operator', 'Slinging Certificate'],
    ['Electrical Safety', 'LOTO Certification', 'HV Awareness'],
    ['Electrical Safety', 'LOTO Certification', 'HV Awareness'],
    ['Confined Space Entry', 'Gas Testing', 'Emergency Response'],
    ['Confined Space Entry', 'Gas Testing', 'Emergency Response'],
    ['NDT Level II', 'Radiation Safety', 'RT Operator'],
    ['NDT Level II', 'Radiation Safety', 'RT Operator'],
    ['Working at Height', 'Scaffold Inspection', 'Fall Arrest'],
    ['Working at Height', 'Scaffold Inspection', 'Fall Arrest'],
    ['Hot Work Safety', 'Fire Watch', 'Welding Certification', 'Advanced Fire Watch',
     'Working at Height', 'Scaffold Inspection', 'Fall Arrest'],
    ['Electrical Safety', 'LOTO Certification', 'HV Awareness',
     'Confined Space Entry', 'Gas Testing', 'Emergency Response'],
]

# ─── Helpers ─────────────────────────────────────────────────────────────────
def classify_permit(step_name: str) -> str:
    s = step_name.lower()
    for permit, keywords in WORK_PERMIT_RULES.items():
        if permit == 'general_work':
            continue
        if any(kw in s for kw in keywords):
            return permit
    return 'general_work'


def get_step_timestamp(discipline: str, step_order: int, total_steps: int) -> datetime:
    start_m, end_m = DISCIPLINE_TIMELINE.get(discipline, (6, 12))
    duration_days = (end_m - start_m) * 30
    progress = (step_order - 1) / max(total_steps - 1, 1)
    offset_days = int(start_m * 30 + progress * duration_days)
    jitter = random.randint(-3, 3)
    return PROJECT_START + timedelta(days=offset_days + jitter)


def generate_workers(all_certs: set, n: int = 50) -> list:
    workers = []
    cert_list = list(all_certs)
    tx_now = datetime.now(timezone.utc)
    GUARANTEED_VALID_FROM = PROJECT_START - timedelta(days=90)
    GUARANTEED_VALID_TO   = PROJECT_START + timedelta(days=3 * 365)

    for i in range(n):
        discipline = random.choice(list(DISCIPLINE_TIMELINE.keys()))
        if i < len(GUARANTEED_SETS):
            certs = GUARANTEED_SETS[i]
            cert_valid_from = GUARANTEED_VALID_FROM
            cert_valid_to   = GUARANTEED_VALID_TO
        else:
            certs = random.sample(cert_list, min(random.randint(1, 4), len(cert_list)))
            cert_valid_from = PROJECT_START - timedelta(days=random.randint(30, 365))
            cert_valid_to   = cert_valid_from + timedelta(days=random.choice([365, 730, 1095]))

        workers.append({
            'id':         f'W-{i+1:03d}',
            'name':       f'Worker_{i+1}',
            'discipline': discipline,
            'certifications': [{
                'cert':       c,
                'valid_from': cert_valid_from.isoformat(),
                'valid_to':   cert_valid_to.isoformat(),
                'tx_time':    tx_now.isoformat(),
            } for c in certs]
        })
    return workers


# ─── Load ─────────────────────────────────────────────────────────────────────
def load_data():
    # Family Steps: step templates per family code
    fs = pd.read_excel(FAMILY_STEPS_FILE, sheet_name='Family Steps')
    fs = fs[fs['Active'] == True].copy()
    fs['FAMILY'] = fs['FAMILY'].str.strip()
    # Build dict: family_code → list of (step_name, order, weight_pct)
    family_steps = {}
    for fam, grp in fs.groupby('FAMILY'):
        grp = grp.sort_values('#')
        family_steps[fam] = [
            (str(row['STEP']).strip(), int(row['#']), float(row['%'] or 0))
            for _, row in grp.iterrows()
        ]
    print(f"✅ Loaded {len(fs)} step templates across {len(family_steps)} families")

    # Meram: real TR activities
    meram = pd.read_excel(MERAM_FILE, sheet_name='Activities PCS')
    meram['Fami'] = meram['Fami'].str.strip()
    meram['Disc'] = meram['Disc'].str.strip()
    meram['Estimated Hours'] = pd.to_numeric(meram['Estimated Hours'], errors='coerce').fillna(0)
    meram['EarnedHours']     = pd.to_numeric(meram['EarnedHours'],     errors='coerce').fillna(0)
    meram['Area']            = meram['Area'].fillna('').astype(str).str.strip()
    meram['Module']          = meram['Module'].fillna('').astype(str).str.strip()
    meram['CWP']             = meram['CWP'].fillna('').astype(str).str.strip()
    print(f"✅ Loaded {len(meram)} Meram activities ({meram['Disc'].nunique()} disciplines, "
          f"{meram['Fami'].nunique()} families)")
    matched = meram['Fami'].isin(family_steps).sum()
    print(f"   {matched} activities ({matched/len(meram)*100:.1f}%) have matching step templates")

    return family_steps, meram


# ─── Main pipeline ────────────────────────────────────────────────────────────
def generate_epc_dataset(family_steps: dict, meram: pd.DataFrame) -> dict:
    tx_now = datetime.now(timezone.utc).isoformat()

    activities, steps, families = [], [], []
    work_permits, certifications = {}, {}
    step_sequences, activity_steps = [], []
    all_certs_needed = set()

    seen_families = set()

    for _, row in meram.iterrows():
        act_id   = str(row['ActID']).strip()
        act_name = str(row['ActivityName']).strip()
        disc     = str(row['Disc']).strip() if row['Disc'] else 'CI'
        fami     = str(row['Fami']).strip() if row['Fami'] else ''
        area     = str(row['Area'])
        module   = str(row['Module'])
        cwp      = str(row['CWP'])
        est_h    = float(row['Estimated Hours'])
        earn_h   = float(row['EarnedHours'])
        progress = earn_h / est_h if est_h > 0 else 0.0

        start_m, end_m = DISCIPLINE_TIMELINE.get(disc, (6, 12))
        act_start = PROJECT_START + timedelta(days=start_m * 30)
        act_end   = PROJECT_START + timedelta(days=end_m   * 30)

        activities.append({
            'id':               act_id,
            'name':             act_name,
            'family':           fami,
            'discipline':       disc,
            'area':             area,
            'module':           module,
            'cwp':              cwp,
            'estimated_hours':  est_h,
            'earned_hours':     earn_h,
            'progress_pct':     round(progress * 100, 2),
            'valid_from':       act_start.isoformat(),
            'valid_to':         act_end.isoformat(),
            'tx_time':          tx_now,
        })

        if fami and fami not in seen_families:
            seen_families.add(fami)
            families.append({'id': fami, 'name': act_name, 'discipline': disc, 'tx_time': tx_now})

        # Resolve step templates
        if fami and fami in family_steps:
            step_templates = family_steps[fami]
        else:
            step_templates = [
                (sname, sord, spct)
                for sname, sord, spct in zip(
                    [s[0] for s in DEFAULT_STEPS['default']],
                    range(1, len(DEFAULT_STEPS['default']) + 1),
                    [s[1] for s in DEFAULT_STEPS['default']],
                )
            ]

        total_steps = len(step_templates)
        prev_step_id = None

        for step_name, step_order, weight_pct in step_templates:
            step_id     = f"{act_id}_S{step_order:02d}"
            permit_type = classify_permit(step_name)
            ts          = get_step_timestamp(disc, step_order, total_steps)

            steps.append({
                'id':          step_id,
                'name':        step_name,
                'order':       step_order,
                'weight_pct':  weight_pct,
                'permit_type': permit_type,
                'activity_id': act_id,
                'discipline':  disc,
                'valid_from':  ts.isoformat(),
                'valid_to':    (ts + timedelta(days=14)).isoformat(),
                'tx_time':     tx_now,
            })

            activity_steps.append({
                'activity_id': act_id,
                'step_id':     step_id,
                'order':       step_order,
                'weight_pct':  weight_pct,
            })

            if permit_type not in work_permits:
                work_permits[permit_type] = {
                    'id':      permit_type,
                    'name':    permit_type.replace('_', ' ').title(),
                    'tx_time': tx_now,
                }
                for cert_name in CERT_REQUIREMENTS.get(permit_type, []):
                    all_certs_needed.add(cert_name)
                    if cert_name not in certifications:
                        certifications[cert_name] = {
                            'id':   cert_name.replace(' ', '_'),
                            'name': cert_name,
                            'tx_time': tx_now,
                        }

            if prev_step_id:
                step_sequences.append({
                    'from':        prev_step_id,
                    'to':          step_id,
                    'activity_id': act_id,
                })
            prev_step_id = step_id

    # Discipline-specific certs
    for disc, certs in DISCIPLINE_EXTRA_CERTS.items():
        for c in certs:
            all_certs_needed.add(c)
            if c not in certifications:
                certifications[c] = {'id': c.replace(' ', '_'), 'name': c, 'tx_time': tx_now}

    workers = generate_workers(all_certs_needed, n=50)

    # Bitemporal rule-change event at month 6
    update_event = {
        'type':        'permit_rule_change',
        'description': 'Hot work permit now requires Advanced Fire Watch certification',
        'valid_from':  (PROJECT_START + timedelta(days=6 * 30)).isoformat(),
        'tx_time':     tx_now,
        'affected':    'hot_work',
        'new_cert':    'Advanced Fire Watch',
    }
    all_certs_needed.add('Advanced Fire Watch')
    certifications['Advanced Fire Watch'] = {
        'id':      'Advanced_Fire_Watch',
        'name':    'Advanced Fire Watch',
        'tx_time': (PROJECT_START + timedelta(days=6 * 30)).isoformat(),
    }

    cwp_counts = meram['CWP'].value_counts().to_dict()
    project = {
        'id':          'PROJ-MERAM',
        'name':        'TR Meram Refinery Expansion Project',
        'start_date':  PROJECT_START.isoformat(),
        'end_date':    (PROJECT_START + timedelta(days=18 * 30)).isoformat(),
        'disciplines': sorted(meram['Disc'].dropna().unique().tolist()),
        'cwps':        list(cwp_counts.keys()),
        'tx_time':     tx_now,
    }

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
            'generated_at':          tx_now,
            'source_activities':     'Meram_PCS_Progress.xlsx (real TR data)',
            'source_steps':          'Family_Steps_macro.xlsm (real TR templates)',
            'source_workers':        'Synthetic (50 workers)',
            'total_activities':      len(activities),
            'total_families':        len(families),
            'total_steps':           len(steps),
            'total_workers':         len(workers),
            'total_permits':         len(work_permits),
            'total_certs':           len(certifications),
            'total_step_sequences':  len(step_sequences),
        }
    }


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    family_steps, meram = load_data()
    dataset = generate_epc_dataset(family_steps, meram)

    out = OUTPUT_DIR / 'epc_dataset_real.json'
    with open(out, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)

    m = dataset['metadata']
    print(f"\n✅ Dataset generated: {out}")
    print(f"   Activities:      {m['total_activities']}")
    print(f"   Families:        {m['total_families']}")
    print(f"   Steps:           {m['total_steps']}")
    print(f"   Step sequences:  {m['total_step_sequences']}")
    print(f"   Workers:         {m['total_workers']}")
    print(f"   Work permits:    {m['total_permits']}")
    print(f"   Certs:           {m['total_certs']}")
    print(f"\n📋 Bitemporal update event: hot_work rule change at month 6")
    print(f"   Real data: {m['source_activities']}")
