"""
simulate_data.py
================
Generates three synthetic CSV files for a healthcare eligibility triage project:
  - data/payer_master.csv        (~40 rows)
  - data/appointments.csv        (~250 rows)
  - data/last_check_history.csv  (~150 rows, ~70% of patients)
 
Run with:  python simulate_data.py
"""
 
import os
import random
from datetime import datetime, timedelta
from typing import List, Tuple
 
import numpy as np
import pandas as pd
from faker import Faker
 
# ─────────────────────────────────────────────
# Global seed – change this to get a different
# but still reproducible dataset.
# ─────────────────────────────────────────────
RANDOM_SEED = 42
DATA_DIR = "././data/c2"
 
fake = Faker()
Faker.seed(RANDOM_SEED)
 
 
# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
 
def ensure_data_dir() -> None:
    """Create the data/ folder if it doesn't already exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
 
 
# ─────────────────────────────────────────────
# 1.  Payer Master
# ─────────────────────────────────────────────
 
def generate_payer_master() -> pd.DataFrame:
    """
    Build the reference table of canonical payers (~40 rows).
 
    Columns:
        payer_code      – short uppercase identifier used in history table
        canonical_name  – human-readable full name
        plan_type       – Commercial | Medicare | Medicaid | Medicare Advantage
        high_turnover   – True for Medicaid managed care + some BCBS commercial plans
    """
 
    # Each entry: (payer_code, canonical_name, plan_type, high_turnover)
    payers = [
        # ── Commercial ──────────────────────────────────────────────────────
        ("AETNA_COMM",    "Aetna Commercial PPO",                      "Commercial",         False),
        ("AETNA_HMO",     "Aetna HMO",                                 "Commercial",         False),
        ("CIGNA_COMM",    "Cigna Commercial PPO",                      "Commercial",         False),
        ("CIGNA_HMO",     "Cigna HMO",                                 "Commercial",         False),
        ("UHC_CHOICE",    "UnitedHealthcare Choice Plus",               "Commercial",         False),
        ("UHC_COMM",      "UnitedHealthcare Commercial",                "Commercial",         False),
        ("HUMANA_COMM",   "Humana Commercial PPO",                      "Commercial",         False),
        ("BCBS_IL_PPO",   "Blue Cross Blue Shield of Illinois PPO",     "Commercial",         True),   # high-turnover BCBS
        ("BCBS_IL_HMO",   "Blue Cross Blue Shield of Illinois HMO",    "Commercial",         False),
        ("BCBS_TX_HMO",   "Blue Cross Blue Shield of Texas HMO",       "Commercial",         True),   # high-turnover BCBS
        ("BCBS_TX_PPO",   "Blue Cross Blue Shield of Texas PPO",       "Commercial",         False),
        ("BCBS_FL",       "Florida Blue (BCBS Florida)",                "Commercial",         False),
        ("ANTHEM",        "Anthem Blue Cross",                          "Commercial",         False),
        ("OSCAR",         "Oscar Health",                               "Commercial",         False),
        ("KAISER",        "Kaiser Permanente",                          "Commercial",         False),
        ("TRICARE",       "Tricare",                                    "Commercial",         False),
        ("MAGELLAN",      "Magellan Health",                            "Commercial",         False),
        ("AMERIHEALTH",   "AmeriHealth Commercial",                     "Commercial",         False),
        ("HEALTHNET",     "Health Net Commercial",                      "Commercial",         False),
        ("CVS_AETNA",     "CVS Health / Aetna",                        "Commercial",         False),
 
        # ── Medicare ────────────────────────────────────────────────────────
        ("MEDICARE_TFI",  "Medicare Traditional Fee-for-Service",       "Medicare",           False),
        ("MEDICARE_B",    "Medicare Part B",                            "Medicare",           False),
 
        # ── Medicare Advantage ───────────────────────────────────────────────
        ("HUMANA_MA",     "Humana Medicare Advantage",                  "Medicare Advantage", False),
        ("UHC_MA",        "UnitedHealthcare Medicare Advantage",        "Medicare Advantage", False),
        ("AETNA_MA",      "Aetna Medicare Advantage",                   "Medicare Advantage", False),
        ("BCBS_MA",       "Blue Cross Blue Shield Medicare Advantage",  "Medicare Advantage", False),
        ("CIGNA_MA",      "Cigna Medicare Advantage",                   "Medicare Advantage", False),
        ("ANTHEM_MA",     "Anthem Medicare Advantage",                  "Medicare Advantage", False),
 
        # ── Medicaid (managed care – all high_turnover=True) ─────────────────
        ("MOLINA",        "Molina Healthcare",                          "Medicaid",           True),
        ("CENTENE",       "Centene Corporation",                        "Medicaid",           True),
        ("AMBETTER",      "Ambetter from Centene",                      "Medicaid",           True),
        ("WELLCARE",      "WellCare Health Plans",                      "Medicaid",           True),
        ("CARESOURCE",    "CareSource",                                 "Medicaid",           True),
        ("FIDELIS",       "Fidelis Care",                               "Medicaid",           True),
        ("MERIDIAN",      "Meridian Health Plan",                       "Medicaid",           True),
        ("IL_MEDICAID",   "Medicaid Illinois Managed Care",             "Medicaid",           True),
        ("TX_MEDICAID",   "Medicaid Texas Managed Care",                "Medicaid",           True),
        ("FL_MEDICAID",   "Medicaid Florida Managed Care",              "Medicaid",           True),
        ("AMERIGROUP",    "Amerigroup Medicaid",                        "Medicaid",           True),
        ("SUNFLOWER",     "Sunflower Health Plan (Medicaid)",           "Medicaid",           True),
    ]
 
    df = pd.DataFrame(payers, columns=["payer_code", "canonical_name", "plan_type", "high_turnover"])
    return df
 
 
# ─────────────────────────────────────────────
# 2.  Appointments  (+ internal patient list)
# ─────────────────────────────────────────────
 
# At least 25 messy insurance_on_file variations – grouped for readability.
INSURANCE_VARIATIONS = [
    # Clean / canonical names
    "Aetna",
    "Cigna",
    "UnitedHealthcare",
    "Humana",
    "Blue Cross Blue Shield of Illinois",
    "Medicare",
    "Medicaid Illinois",
 
    # Abbreviations
    "BCBSIL",
    "BCBS IL",
    "UHC",
    "HCSC",
 
    # With plan tier
    "Aetna PPO",
    "BCBS IL HMO",
    "BCBS IL PPO",
    "UnitedHealthcare Choice Plus",
    "Cigna HMO",
 
    # Long / formal names
    "Blue Cross Blue Shield of Texas",
    "Blue Cross Blue Shield of Florida",
    "Molina Healthcare",
    "WellCare Health Plans",
    "Centene / Ambetter",
    "Humana Medicare Advantage",
    "UnitedHealthcare Medicare Advantage",
    "Anthem Blue Cross",
    "Kaiser Permanente",
    "Oscar Health",
    "Tricare",
    "AmeriHealth",
    "Health Net",
    "CVS/Aetna",
 
    # Typos / OCR errors
    "Aetna.",
    "Humanna",
    "United Health Care",
    "Blue Cros Blu Sheild IL",
    "Cigna.",
    "Molina Helthcare",
    "Well Care",
 
    # Member ID stuffed into the field
    "BCBSIL #MBR12345",
    "Aetna / ID: 987654",
    "UHC - 334455XY",
 
    # Ambiguous / garbage
    "self-pay",
    "??",
    "Medicare or Medicaid",
    "BCBS / Aetna",
    "unknown",
    "cash pay",
    "see notes",
    "N/A",
    "BCBS/UHC",
]
 
# Weights: heavier on normal entries, lighter on ambiguous/missing.
# We build a weighted pool so the proportions feel realistic.
_NORMAL_POOL    = INSURANCE_VARIATIONS[:36]   # clean / abbrev / typo entries
_AMBIGUOUS_POOL = INSURANCE_VARIATIONS[36:]   # ambiguous entries
 
PROVIDERS = [
    "Dr. Sarah Patel",
    "Dr. James Rodriguez",
    "Dr. Emily Chen",
    "Dr. Michael O'Brien",
    "Dr. Aisha Washington",
    "Dr. Thomas Nguyen",
]
 
 
def _random_member_id() -> str:
    """Return a plausible member-ID string like 'MBR-48302'."""
    return f"MBR-{random.randint(10000, 99999)}"
 
 
def generate_patients_and_appointments(
    payer_master: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate ~250 appointment rows.
 
    Returns
    -------
    patients_df      – thin table of (patient_id, dob, member_id)
                       used later to build history.
    appointments_df  – full appointments CSV.
    """
 
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    n = 250
 
    rows = []
    patient_records = []   # store (patient_id, member_id) for history generation
 
    for i in range(1, n + 1):
        patient_id = f"P{i:03d}"
 
        # ── Demographics ──────────────────────────────────────────────────
        patient_name = fake.name()
        # Adults: 18–85 years old
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d")
 
        # ── Appointment time: 8:00 – 17:00 on today ──────────────────────
        appt_hour   = random.randint(8, 16)
        appt_minute = random.choice([0, 15, 30, 45])
        appt_dt     = today.replace(hour=appt_hour, minute=appt_minute)
 
        # ── Provider ──────────────────────────────────────────────────────
        provider_name = random.choice(PROVIDERS)
 
        # ── Member ID (present 75% of the time) ───────────────────────────
        if random.random() < 0.75:
            member_id = _random_member_id()
        else:
            member_id = ""
 
        # ── Insurance on file – messy free text ───────────────────────────
        roll = random.random()
        if roll < 0.03:
            # ~3% completely missing
            insurance_on_file = np.nan
        elif roll < 0.13:
            # ~10% genuinely ambiguous
            insurance_on_file = random.choice(_AMBIGUOUS_POOL)
        else:
            # The rest: pick from the broader messy pool
            insurance_on_file = random.choice(_NORMAL_POOL)
 
        rows.append({
            "patient_id":          patient_id,
            "patient_name":        patient_name,
            "dob":                 dob,
            "appointment_datetime": appt_dt.strftime("%Y-%m-%d %H:%M"),
            "provider_name":       provider_name,
            "insurance_on_file":   insurance_on_file,
            "member_id":           member_id,
        })
 
        patient_records.append({
            "patient_id": patient_id,
            "member_id":  member_id,
        })
 
    appointments_df = pd.DataFrame(rows)
    patients_df     = pd.DataFrame(patient_records)
    return patients_df, appointments_df
 
 
# ─────────────────────────────────────────────
# 3.  Last-Check History
# ─────────────────────────────────────────────
 
def generate_last_check_history(
    patients_df: pd.DataFrame,
    payer_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build eligibility check history covering ~70% of patients.
 
    Special cases built in:
      • ~15 patients have a member_id that differs from appointments.csv
        → triggers "member ID changed" rule in triage logic.
      • ~20 patients have a payer_code that doesn't match what's on file
        → triggers "payer changed" rule.
 
    Columns:
        patient_id, payer_code, member_id, last_check_date, result
    """
 
    today = datetime.today().date()
    payer_codes = payer_master["payer_code"].tolist()
 
    all_patients = patients_df["patient_id"].tolist()
    n_total      = len(all_patients)
 
    # Pick ~70% of patients to have history
    n_with_history = int(n_total * 0.70)
    patients_with_history = random.sample(all_patients, n_with_history)
 
    # Pre-select patients that will have mismatched member IDs / payer codes
    changed_member_id_patients = set(random.sample(patients_with_history, 15))
    changed_payer_patients     = set(random.sample(
        [p for p in patients_with_history if p not in changed_member_id_patients],
        20,
    ))
 
    # Result distribution: 70% Active, 20% Inactive, 10% Unknown
    results        = ["Active", "Inactive", "Unknown"]
    result_weights = [0.70,     0.20,       0.10]
 
    rows = []
    appt_member_id = dict(zip(patients_df["patient_id"], patients_df["member_id"]))
 
    for patient_id in patients_with_history:
        # Random date in the last 60 days
        days_ago       = random.randint(0, 35)
        last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
 
        # Payer code
        if patient_id in changed_payer_patients:
            # Deliberately use a different (random) payer to simulate payer change
            payer_code = random.choice(payer_codes)
        else:
            payer_code = random.choice(payer_codes)  # normal assignment
 
        # Member ID
        original_member_id = appt_member_id.get(patient_id, "")
        if patient_id in changed_member_id_patients:
            # Use a different member ID to trigger the "ID changed" rule
            new_id = _random_member_id()
            # Make sure it actually differs
            while new_id == original_member_id:
                new_id = _random_member_id()
            member_id = new_id
        else:
            member_id = original_member_id
 
        # Weighted result
        result = random.choices(results, weights=result_weights, k=1)[0]
 
        rows.append({
            "patient_id":     patient_id,
            "payer_code":     payer_code,
            "member_id":      member_id,
            "last_check_date": last_check_date,
            "result":         result,
        })
 
    return pd.DataFrame(rows)
 
 
# ─────────────────────────────────────────────
# 4.  Main
# ─────────────────────────────────────────────
 
def main() -> None:
    # Fix all random sources for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
 
    ensure_data_dir()
 
    # Generate tables
    payer_master                  = generate_payer_master()
    patients_df, appointments_df  = generate_patients_and_appointments(payer_master)
    last_check_df                 = generate_last_check_history(patients_df, payer_master)
 
    # Write CSVs
    payer_master.to_csv(    os.path.join(DATA_DIR, "payer_master.csv"),        index=False)
    appointments_df.to_csv( os.path.join(DATA_DIR, "appointments.csv"),        index=False)
    last_check_df.to_csv(   os.path.join(DATA_DIR, "last_check_history.csv"),  index=False)
 
    # Summary
    print("\n✅  Data generation complete")
    print(f"   payer_master.csv        → {len(payer_master):>4} rows")
    print(f"   appointments.csv        → {len(appointments_df):>4} rows")
    print(f"   last_check_history.csv  → {len(last_check_df):>4} rows  "
          f"({len(last_check_df) / len(appointments_df) * 100:.0f}% of patients covered)")
    print(f"\n   Files saved to ./{DATA_DIR}/\n")
 
 
if __name__ == "__main__":
    main()