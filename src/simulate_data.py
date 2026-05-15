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
from typing import List, Tuple, Set

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
        # ── Commercial ────────────────────────────────────────────────────────
        ("AETNA_COMM",   "Aetna Commercial PPO",                      "Commercial",         False),
        ("AETNA_HMO",    "Aetna HMO",                                 "Commercial",         False),
        ("CIGNA_COMM",   "Cigna Commercial PPO",                      "Commercial",         False),
        ("CIGNA_HMO",    "Cigna HMO",                                 "Commercial",         False),
        ("UHC_COMM",     "UnitedHealthcare Commercial",                "Commercial",         False),
        ("UHC_CHOICE",   "UnitedHealthcare Choice Plus",               "Commercial",         False),
        ("HUMANA_COMM",  "Humana Commercial PPO",                      "Commercial",         False),
        ("BCBS_IL_PPO",  "Blue Cross Blue Shield of Illinois PPO",     "Commercial",         True),
        ("BCBS_IL_HMO",  "Blue Cross Blue Shield of Illinois HMO",     "Commercial",         False),
        ("BCBS_TX_HMO",  "Blue Cross Blue Shield of Texas HMO",        "Commercial",         True),
        ("BCBS_TX_PPO",  "Blue Cross Blue Shield of Texas PPO",        "Commercial",         False),
        ("BCBS_FL",      "Florida Blue (BCBS Florida)",                 "Commercial",         False),
        ("ANTHEM",       "Anthem Blue Cross",                          "Commercial",         False),
        ("OSCAR",        "Oscar Health",                               "Commercial",         False),
        ("KAISER",       "Kaiser Permanente",                          "Commercial",         False),
        ("TRICARE",      "Tricare",                                    "Commercial",         False),
        ("MAGELLAN",     "Magellan Health",                            "Commercial",         False),
        ("AMERIHEALTH",  "AmeriHealth Commercial",                     "Commercial",         False),
        ("HEALTHNET",    "Health Net Commercial",                      "Commercial",         False),
        ("CVS_AETNA",    "CVS Health / Aetna",                        "Commercial",         False),
 
        # ── Medicare ──────────────────────────────────────────────────────────
        ("MEDICARE_TFI", "Medicare Traditional Fee-for-Service",       "Medicare",           False),
        ("MEDICARE_B",   "Medicare Part B",                            "Medicare",           False),
 
        # ── Medicare Advantage ────────────────────────────────────────────────
        ("HUMANA_MA",    "Humana Medicare Advantage",                  "Medicare Advantage", False),
        ("UHC_MA",       "UnitedHealthcare Medicare Advantage",        "Medicare Advantage", False),
        ("AETNA_MA",     "Aetna Medicare Advantage",                   "Medicare Advantage", False),
        ("BCBS_MA",      "Blue Cross Blue Shield Medicare Advantage",  "Medicare Advantage", False),
        ("CIGNA_MA",     "Cigna Medicare Advantage",                   "Medicare Advantage", False),
        ("ANTHEM_MA",    "Anthem Medicare Advantage",                  "Medicare Advantage", False),
        ("WELLCARE_MA",  "WellCare Medicare Advantage",                "Medicare Advantage", True),
 
        # ── Medicaid managed care — ALL high_turnover=True ────────────────────
        ("MOLINA",       "Molina Healthcare",                          "Medicaid",           True),
        ("CENTENE",      "Centene Corporation",                        "Medicaid",           True),
        ("AMBETTER",     "Ambetter from Centene",                      "Medicaid",           True),
        ("WELLCARE",     "WellCare Health Plans",                      "Medicaid",           True),
        ("CARESOURCE",   "CareSource",                                 "Medicaid",           True),
        ("FIDELIS",      "Fidelis Care",                               "Medicaid",           True),
        ("MERIDIAN",     "Meridian Health Plan",                       "Medicaid",           True),
        ("IL_MEDICAID",  "Medicaid Illinois Managed Care",             "Medicaid",           True),
        ("TX_MEDICAID",  "Medicaid Texas Managed Care",                "Medicaid",           True),
        ("FL_MEDICAID",  "Medicaid Florida Managed Care",              "Medicaid",           True),
        ("AMERIGROUP",   "Amerigroup Medicaid",                        "Medicaid",           True),
    ]
    return pd.DataFrame(payers, columns=["payer_code", "canonical_name", "plan_type", "high_turnover"])


# ============================================================
# 2. Appointments (~250 rows)
# ============================================================

# ── Insurance variation pools ────────────────────────────────
# Weighted so the dataset has the right shape of messiness:
#   ~3%  → truly missing (NaN)
#   ~10% → genuinely ambiguous (self-pay, ??, dual-payer, etc.)
#   ~87% → messy-but-real (clean names, abbrevs, typos, ID-stuffed)

_NORMAL_POOL: List[str] = [
    # Clean-ish names 
    "Aetna", "Cigna", "UnitedHealthcare", "Humana",
    "Blue Cross Blue Shield of Illinois",
    "AmeriHealth", "Health Net",
    # Abbreviations
    "BCBSIL", "BCBS IL", "UHC", "HCSC",
    # With plan tier — these are NOT exact canonical names so no overlap risk
    "Aetna PPO", "BCBS IL HMO", "BCBS IL PPO",
    "UHC Choice Plus", "Anthem Gold Plan",
    # Long names
    "Blue Cross Blue Shield of Texas", "Blue Cross Blue Shield of Florida",
    "Humana Medicare Advantage plan", "UnitedHealthcare MA",
    "Centene / Ambetter",
    # Member ID stuffed in
    "BCBSIL #MBR12345", "Aetna / ID: 987654", "UHC - 334455XY",
    # Typos / OCR garbage — safe, won't match canonical exactly
    "Blue Cros Blu Sheild IL", "Humanna", "United Health Care",
    "Aetna.", "Cigna.", "Molina Helthcare", "Well Care",
    # Medicaid / Medicare shorthand (not exact canonical)
    "Medicaid Illinois", "Medicaid Texas", "Medicare",
    "WellCare", "Ambetter", "Centene", "CVS/Aetna",
    "Fidelis", "Meridian", "Molina", "Oscar", "Kaiser",
    "Tricare", "Magellan", "Fidelis Care NY", "Anthem BCBS",
]

_AMBIGUOUS_POOL = [
    "self-pay", "??", "unknown", "cash pay",
    "Medicare or Medicaid", "BCBS / Aetna", "BCBS/UHC",
    "see notes", "N/A"
]

PROVIDERS: List[str] = [
    "Dr. Sarah Patel",
    "Dr. James Rodriguez",
    "Dr. Emily Chen",
    "Dr. Michael O'Brien",
    "Dr. Aisha Washington",
    "Dr. Thomas Nguyen",
]


def generate_patients_and_appointments(payer_master) :
    """
    Generate ~250 appointment rows plus a slim patients table.

    Returns
    -------
    patients_df      (patient_id, member_id) used to build history
    appointments_df  full appointments CSV
    """

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

    canonical_names = payer_master["canonical_name"].tolist()

    appt_rows    = []
    patient_rows = []
    clean_patients = set()

    for i in range(1, 251):
        patient_id = f"P{i:03d}"

        # Demographics
        patient_name = fake.name()
        # minimum_age=13
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85).strftime("%Y-%m-%d")

        # Appointment time: 08:00 – 17:00 today if weekday usual non urgent hours
        appt_dt = today.replace(
            hour=random.randint(8, 16),
            minute=random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55]),
        )

        # Member ID: present 75% of the time
        member_id = _random_member_id() if random.random() < 0.75 else ""

# Decide insurance_on_file — three-way weighted split
        if random.random() < 0.25: #25% get perefectly clean data and insuracne name macthes
            # ~25% clean cases: exact canonical name so normalizer
            # hits instantly and rule engine marks them OK
            insurance_on_file = random.choice(canonical_names)
            if not member_id:
                member_id = _random_member_id()
            clean_patients.add(patient_id)
        else:
            # ~75% messy cases
            roll = random.random()
            if roll < 0.03:
                insurance_on_file = np.nan  # ~3% truly missing
            elif roll < 0.13:
                insurance_on_file = random.choice(_AMBIGUOUS_POOL) # ~10% genuinely ambiguous
            else:
                insurance_on_file = random.choice(_NORMAL_POOL) # ~87% messy-but-real

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

    return pd.DataFrame(patient_rows), pd.DataFrame(appt_rows), clean_patients

def generate_last_check_history(
    patients_df:     pd.DataFrame,
    appointments_df: pd.DataFrame,
    payer_master:    pd.DataFrame,
    clean_patients:  Set[str],
) -> pd.DataFrame:
 
    today      = datetime.today().date()
    payer_codes = payer_master["payer_code"].tolist()
 
    # Build dictionaries for easy of lookup
    canonical_to_code = dict(
        zip(payer_master["canonical_name"], payer_master["payer_code"])
    ) #build a dictionary to map canonical name to payer code for quick lookup
    appt_insurance = dict(
        zip(appointments_df["patient_id"], appointments_df["insurance_on_file"])
    ) #build a dictionary to map patient_id to insurance_on_file for quick lookup
    appt_member_id = dict(
        zip(patients_df["patient_id"], patients_df["member_id"])
    ) #build a dictionary to map patient_id to member_id for quick lookup
 
    all_patients      = patients_df["patient_id"].tolist()#list of all patient ids

    n_with_history    = int(len(all_patients) * 0.70)   #take 70% of all patients
    patients_with_history = random.sample(all_patients, n_with_history) #give 70% of patients history
 

    #list of patients with history but not in clean_patients
    non_clean = [p for p in patients_with_history if p not in clean_patients]  
    
    #pick 15 rto have non clean mismatch member IDS
    changed_member_patients = set(random.sample(non_clean, min(15, len(non_clean))))

    #list of patients with history but not in changed_member_patients
    remaining               = [p for p in non_clean if p not in changed_member_patients]
    
    changed_payer_patients  = set(random.sample(remaining, min(20, len(remaining)))) 
    #pick 20 to have non clean mismatch payer code
 
    rows = []
 
    for patient_id in patients_with_history:
        # For clean patients, we ensure the payer_code matches the canonical name in appointments_df,
        if patient_id in clean_patients:
            # Use the exact payer_code that matches the canonical name in appointments
            insurance_val = appt_insurance.get(patient_id, "")
            payer_code    = canonical_to_code.get(insurance_val, random.choice(payer_codes))
            member_id     = appt_member_id.get(patient_id, _random_member_id())
            # Fresh check — always within 10 days so it passes Rule 1
            days_ago      = random.randint(1, 20) 
            #can be up to 30 days ago but we skew towards more recent checks to ensure a good mix of pass/fail for Rule 1
 
        else: # For non-clean patients, we introduce messiness and potential mismatches.
            # Normal case: spread over last 40 days
            days_ago   = random.randint(0, 40)
            payer_code = random.choice(payer_codes)
             #get the original member_id from appointments_df for this patient_id, default to empty string if not found
            original = appt_member_id.get(patient_id, "")
            # If this patient is in the changed_member_patients set and has an original member_id, we generate a new random member_id that is different from the original to simulate a mismatch. Otherwise, we use the original member_id (which may be empty if not found).
            if patient_id in changed_member_patients and original:
                # Different member_id → fires Rule 3
                new_id = _random_member_id()
                while new_id == original:
                    new_id = _random_member_id()
                member_id = new_id
            else:
                member_id = original
        # Assign a last check date by subtracting today from days ago
        last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d") 
        result = random.choices(
            ["Active", "Inactive", "Unknown"],
            weights=[0.70, 0.20, 0.10],
            k=1,
        )[0] #give a random result from three 3 only give 1 but with weighted results probability
 
        rows.append({
            "patient_id":      patient_id,
            "payer_code":      payer_code,
            "member_id":       member_id,
            "last_check_date": last_check_date,
            "result":          result,
        })
 
    return pd.DataFrame(rows)
# ============================================================
# Main
# ============================================================

def main():

    # doin here again for clairty and recheck not needed
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_data_dir() #check directory for data exists and create if not

    #initialize dataframers for 3 tables each with their own generation function
    payer_master                 = generate_payer_master()
    patients_df, appointments_df, clean_patients = generate_patients_and_appointments(payer_master)
    last_check_df                = generate_last_check_history(patients_df, appointments_df, payer_master, clean_patients)


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