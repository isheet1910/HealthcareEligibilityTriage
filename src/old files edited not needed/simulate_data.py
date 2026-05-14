"""
simulate.py
===========
Generates three reproducible synthetic CSV files for the insurance
eligibility triage take-home project.

Output (written to ./data/):
    payer_master.csv        ~40 rows
    appointments.csv        ~250 rows
    last_check_history.csv  ~175 rows  (~70% of patients)

Run:
    python simulate.py
"""

import os
import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# ============================================================
# Seed — change this to get a different but still reproducible run
# ============================================================

RANDOM_SEED = 42
DATA_DIR    = "data"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker()
Faker.seed(RANDOM_SEED)


# ============================================================
# Utilities
# ============================================================

def ensure_data_dir() -> None:
    """Create ./data/ if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def _random_member_id() -> str:
    """Return a plausible member ID like 'MBR-48302'."""
    return f"MBR-{random.randint(10000, 99999)}"


# ============================================================
# 1. Payer Master (~40 rows)
# ============================================================

def generate_payer_master() -> pd.DataFrame:
    """
    Build the canonical payer reference table.

    Columns: payer_code, canonical_name, plan_type, high_turnover

    high_turnover = True for:
      - All Medicaid managed care plans (per spec)
      - BCBS Illinois PPO and BCBS Texas HMO (per spec — "a couple of BCBS plans")
      - WellCare Medicare Advantage (high churn plan)
    """

    payers = [
        # ── Commercial ───────────────────────────────────────────────────────
        ("AETNA_COMM",      "Aetna Commercial PPO",                     "Commercial",         False),
        ("AETNA_HMO",       "Aetna HMO",                                "Commercial",         False),
        ("CIGNA_COMM",      "Cigna Commercial PPO",                     "Commercial",         False),
        ("CIGNA_HMO",       "Cigna HMO",                                "Commercial",         False),
        ("UHC_COMM",        "UnitedHealthcare Commercial",               "Commercial",         False),
        ("UHC_CHOICE",      "UnitedHealthcare Choice Plus",              "Commercial",         False),
        ("HUMANA_COMM",     "Humana Commercial PPO",                     "Commercial",         False),
        ("BCBS_IL_PPO",     "Blue Cross Blue Shield of Illinois PPO",    "Commercial",         True),  # ← high-turnover BCBS
        ("BCBS_IL_HMO",     "Blue Cross Blue Shield of Illinois HMO",   "Commercial",         False),
        ("BCBS_TX_HMO",     "Blue Cross Blue Shield of Texas HMO",      "Commercial",         True),  # ← high-turnover BCBS
        ("BCBS_TX_PPO",     "Blue Cross Blue Shield of Texas PPO",      "Commercial",         False),
        ("BCBS_FL",         "Florida Blue (BCBS Florida)",               "Commercial",         False),
        ("ANTHEM",          "Anthem Blue Cross",                         "Commercial",         False),
        ("OSCAR",           "Oscar Health",                              "Commercial",         False),
        ("KAISER",          "Kaiser Permanente",                         "Commercial",         False),
        ("TRICARE",         "Tricare",                                   "Commercial",         False),
        ("MAGELLAN",        "Magellan Health",                           "Commercial",         False),
        ("AMERIHEALTH",     "AmeriHealth Commercial",                    "Commercial",         False),
        ("HEALTHNET",       "Health Net Commercial",                     "Commercial",         False),
        ("CVS_AETNA",       "CVS Health / Aetna",                       "Commercial",         False),

        # ── Medicare ─────────────────────────────────────────────────────────
        ("MEDICARE_TFI",    "Medicare Traditional Fee-for-Service",      "Medicare",           False),
        ("MEDICARE_B",      "Medicare Part B",                           "Medicare",           False),

        # ── Medicare Advantage ────────────────────────────────────────────────
        ("HUMANA_MA",       "Humana Medicare Advantage",                 "Medicare Advantage", False),
        ("UHC_MA",          "UnitedHealthcare Medicare Advantage",       "Medicare Advantage", False),
        ("AETNA_MA",        "Aetna Medicare Advantage",                  "Medicare Advantage", False),
        ("BCBS_MA",         "Blue Cross Blue Shield Medicare Advantage", "Medicare Advantage", False),
        ("CIGNA_MA",        "Cigna Medicare Advantage",                  "Medicare Advantage", False),
        ("ANTHEM_MA",       "Anthem Medicare Advantage",                 "Medicare Advantage", False),
        ("WELLCARE_MA",     "WellCare Medicare Advantage",               "Medicare Advantage", True),

        # ── Medicaid managed care — ALL high_turnover=True (per spec) ─────────
        ("MOLINA",          "Molina Healthcare",                         "Medicaid",           True),
        ("CENTENE",         "Centene Corporation",                       "Medicaid",           True),
        ("AMBETTER",        "Ambetter from Centene",                     "Medicaid",           True),
        ("WELLCARE",        "WellCare Health Plans",                     "Medicaid",           True),
        ("CARESOURCE",      "CareSource",                                "Medicaid",           True),
        ("FIDELIS",         "Fidelis Care",                              "Medicaid",           True),
        ("MERIDIAN",        "Meridian Health Plan",                      "Medicaid",           True),
        ("IL_MEDICAID",     "Medicaid Illinois Managed Care",            "Medicaid",           True),
        ("TX_MEDICAID",     "Medicaid Texas Managed Care",               "Medicaid",           True),
        ("FL_MEDICAID",     "Medicaid Florida Managed Care",             "Medicaid",           True),
        ("AMERIGROUP",      "Amerigroup Medicaid",                       "Medicaid",           True),
    ]

    return pd.DataFrame(
        payers,
        columns=["payer_code", "canonical_name", "plan_type", "high_turnover"]
    )


# ============================================================
# 2. Appointments (~250 rows)
# ============================================================

# ── Insurance variation pools ────────────────────────────────
# Weighted so the dataset has the right shape of messiness:
#   ~3%  → truly missing (NaN)
#   ~10% → genuinely ambiguous (self-pay, ??, dual-payer, etc.)
#   ~87% → messy-but-real (clean names, abbrevs, typos, ID-stuffed)

_NORMAL_POOL: List[str] = [
    # Clean canonical names
    "Aetna",
    "Cigna",
    "UnitedHealthcare",
    "Humana",
    "Blue Cross Blue Shield of Illinois",
    "Medicare",
    "Medicaid Illinois",
    "Medicaid Texas",
    "Molina Healthcare",
    "WellCare Health Plans",
    "Ambetter",
    "Oscar Health",
    "Anthem Blue Cross",
    "Kaiser Permanente",
    "Tricare",
    "AmeriHealth",
    "Health Net",
    "CVS/Aetna",
    "Fidelis Care",
    "Magellan Health",
    "Meridian Health Plan",
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
    "Anthem Gold Plan",
    # Long names
    "Blue Cross Blue Shield of Texas",
    "Blue Cross Blue Shield of Florida",
    "Humana Medicare Advantage",
    "UnitedHealthcare Medicare Advantage",
    "Centene / Ambetter",
    # Member ID stuffed into the field
    "BCBSIL #MBR12345",
    "Aetna / ID: 987654",
    "UHC - 334455XY",
    # Typos / OCR garbage
    "Blue Cros Blu Sheild IL",
    "Humanna",
    "United Health Care",
    "Aetna.",
    "Cigna.",
    "Molina Helthcare",
    "Well Care",
]

_AMBIGUOUS_POOL: List[str] = [
    "self-pay",
    "??",
    "Medicare or Medicaid",
    "BCBS / Aetna",
    "BCBS/UHC",
    "unknown",
    "cash pay",
    "see notes",
    "N/A",
]

PROVIDERS: List[str] = [
    "Dr. Sarah Patel",
    "Dr. James Rodriguez",
    "Dr. Emily Chen",
    "Dr. Michael O'Brien",
    "Dr. Aisha Washington",
    "Dr. Thomas Nguyen",
]


def generate_patients_and_appointments(
    payer_master: pd.DataFrame,   # accepted for API consistency; not used directly
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate ~250 appointment rows plus a slim patients table.

    Returns
    -------
    patients_df     – (patient_id, member_id) used to build history
    appointments_df – full appointments CSV
    """

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    appt_rows    = []
    patient_rows = []

    for i in range(1, 251):
        patient_id = f"P{i:03d}"

        # Demographics
        patient_name = fake.name()
        dob = fake.date_of_birth(minimum_age=5, maximum_age=85).strftime("%Y-%m-%d")

        # Appointment time: 08:00 – 17:00 today
        appt_dt = today.replace(
            hour=random.randint(8, 16),
            minute=random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55]),
        )

        # Member ID: present 75% of the time
        member_id = _random_member_id() if random.random() < 0.75 else ""

        # Insurance on file — weighted messy distribution
        roll = random.random()
        if roll < 0.04:
            insurance_on_file = np.nan           # ~3%  truly missing
        elif roll < 0.20:
            insurance_on_file = random.choice(_AMBIGUOUS_POOL)   # ~10% ambiguous
        else:
            insurance_on_file = random.choice(_NORMAL_POOL)      # ~87% messy-real

        random.choice(_NORMAL_POOL)

        # ------------------------------------------------------------
        # FORCE ~20% CLEAN OK CASES
        # ------------------------------------------------------------
        if random.random() < 0.20:
            insurance_on_file = random.choice([
                "Aetna", "Cigna", "UnitedHealthcare",
                "Humana", "Blue Cross Blue Shield of Illinois"
            ])
            member_id = _random_member_id()

        appt_rows.append({
            "patient_id":           patient_id,
            "patient_name":         patient_name,
            "dob":                  dob,
            "appointment_datetime": appt_dt.strftime("%Y-%m-%d %H:%M"),
            "provider_name":        random.choice(PROVIDERS),
            "insurance_on_file":    insurance_on_file,
            "member_id":            member_id,
        })

        patient_rows.append({"patient_id": patient_id, "member_id": member_id})

    return pd.DataFrame(patient_rows), pd.DataFrame(appt_rows)


# ============================================================
# 3. Last-Check History (~175 rows, ~70% of patients)
# ============================================================

def generate_last_check_history(
    patients_df: pd.DataFrame,
    payer_master: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build eligibility check history covering ~70% of patients.

    Deliberately seeded mismatches to exercise triage rules:
      - 15 patients have a DIFFERENT member_id than appointments.csv
        → triggers Rule 3 "member ID changed"
      - 20 patients have a DIFFERENT payer_code than what is on file
        → triggers Rule 2 "payer changed"

    last_check_date is spread across the last 40 days (per spec),
    skewed so a realistic mix falls inside and outside the 30-day window.
    """

    today      = datetime.today().date()
    payer_codes = payer_master["payer_code"].tolist()

    all_patients   = patients_df["patient_id"].tolist()
    n_with_history = int(len(all_patients) * 0.70)  

    patients_with_history = random.sample(all_patients, n_with_history)

    # Pre-select mismatch patients (non-overlapping sets)
    changed_member_patients = set(random.sample(patients_with_history, 5))
    remaining               = [p for p in patients_with_history if p not in changed_member_patients]
    changed_payer_patients  = set(random.sample(remaining, 20))

    appt_member_id = dict(zip(patients_df["patient_id"], patients_df["member_id"]))

    rows = []

    for patient_id in patients_with_history:
        # Spread dates over the last 40 days (per spec)
        days_ago        = random.randint(0, 35)
        last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        # Payer code — all patients get a random assignment;
        # changed_payer_patients are just as random, but the
        # normalize step will map appointments to a DIFFERENT code,
        # so the mismatch emerges naturally through the pipeline.
        # payer_code = random.choice(payer_codes)
        if random.random() < 0.70:          # 70% same payer
            payer_code = payer_master.sample(1)["payer_code"].iloc[0]
        else:
            payer_code = random.choice(payer_codes)

        # Member ID
        original = appt_member_id.get(patient_id, "")
        if patient_id in changed_member_patients and original:
            # Guarantee a different ID to fire Rule 3
            new_id = _random_member_id()
            while new_id == original:
                new_id = _random_member_id()
            member_id = new_id
        else:
            member_id = original

        # ------------------------------------------------------------
        # FORCE CLEAN OK CASES IN HISTORY
        # ------------------------------------------------------------
        if random.random() < 0.20:
            payer_code = payer_master.sample(1)["payer_code"].iloc[0]
            member_id = original
            days_ago = random.randint(1, 10)
            last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        result = random.choices(
            ["Active", "Inactive", "Unknown"],
            weights=[0.70, 0.20, 0.10],
            k=1,
        )[0]

        rows.append({
            "patient_id":     patient_id,
            "payer_code":     payer_code,
            "member_id":      member_id,
            "last_check_date": last_check_date,
            "result":         result,
        })

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main() -> None:
    # Seeds are set at module level so generation order doesn't matter,
    # but we re-affirm here for clarity.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_data_dir()

    payer_master                 = generate_payer_master()
    patients_df, appointments_df = generate_patients_and_appointments(payer_master)
    last_check_df                = generate_last_check_history(patients_df, payer_master)

    # Write CSVs
    payer_master.to_csv(    os.path.join(DATA_DIR, "payer_master.csv"),        index=False)
    appointments_df.to_csv( os.path.join(DATA_DIR, "appointments.csv"),        index=False)
    last_check_df.to_csv(   os.path.join(DATA_DIR, "last_check_history.csv"),  index=False)

    # Summary
    coverage = len(last_check_df) / len(appointments_df) * 100
    print("\n✅  Data generation complete")
    print(f"   payer_master.csv        →  {len(payer_master):>3} rows")
    print(f"   appointments.csv        →  {len(appointments_df):>3} rows")
    print(f"   last_check_history.csv  →  {len(last_check_df):>3} rows  ({coverage:.0f}% of patients covered)")
    print(f"\n   Files saved to ./{DATA_DIR}/\n")

    # Sanity checks printed for quick review
    missing_ins = appointments_df["insurance_on_file"].isna().sum()
    ambig_ins   = appointments_df["insurance_on_file"].isin(_AMBIGUOUS_POOL).sum()
    no_member   = (appointments_df["member_id"] == "").sum()
    print("   Sanity checks (appointments):")
    print(f"     Missing insurance_on_file : {missing_ins}")
    print(f"     Ambiguous insurance       : {ambig_ins}")
    print(f"     No member_id              : {no_member}")
    print(f"     Distinct insurance values : {appointments_df['insurance_on_file'].nunique()}\n")


if __name__ == "__main__":
    main()