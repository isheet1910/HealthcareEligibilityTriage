#!/usr/bin/env python3
"""
simulate_data.py
Generates synthetic datasets for an insurance eligibility triage project.

Creates:
- data/payer_master.csv (~40 rows)
- data/appointments.csv (~250 rows)
- data/last_check_history.csv (~150 rows)

Uses reproducible RANDOM_SEED = 42.
"""

import os
import random
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker

RANDOM_SEED = 42
DATA_DIR = "././data/c3"


# ---------------------------------------------------------
# Utility: ensure data directory exists
# ---------------------------------------------------------
def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


# ---------------------------------------------------------
# Generate payer master (~40 rows)
# ---------------------------------------------------------
def generate_payer_master() -> pd.DataFrame:
    """
    Create a synthetic payer master table with:
    payer_code, canonical_name, plan_type, high_turnover
    """
    payers = [
        ("BCBS_IL", "Blue Cross Blue Shield of Illinois", "Commercial", True),
        ("BCBS_TX", "Blue Cross Blue Shield of Texas", "Commercial", True),
        ("BCBS_FL", "Blue Cross Blue Shield of Florida", "Commercial", False),
        ("AETNA_COMM", "Aetna Commercial PPO", "Commercial", False),
        ("AETNA_MA", "Aetna Medicare Advantage", "Medicare Advantage", False),
        ("UHC_COMM", "UnitedHealthcare Choice Plus", "Commercial", False),
        ("UHC_MA", "UnitedHealthcare Medicare Advantage", "Medicare Advantage", False),
        ("CIGNA_COMM", "Cigna Preferred", "Commercial", False),
        ("HUMANA_COMM", "Humana Commercial", "Commercial", False),
        ("HUMANA_MA", "Humana Gold Plus", "Medicare Advantage", False),
        ("MOLINA_IL", "Molina Healthcare Illinois", "Medicaid", True),
        ("MOLINA_TX", "Molina Healthcare Texas", "Medicaid", True),
        ("CENTENE_AMB", "Centene Ambetter", "Medicaid", True),
        ("WELLCARE", "WellCare Health Plans", "Medicaid", True),
        ("OSCAR_COMM", "Oscar Health Commercial", "Commercial", False),
        ("KAISER_COMM", "Kaiser Permanente Commercial", "Commercial", False),
        ("ANTHEM_COMM", "Anthem Blue Commercial", "Commercial", False),
        ("CVS_AETNA", "CVS/Aetna Integrated", "Commercial", False),
        ("TRICARE", "Tricare Prime", "Commercial", False),
        ("MAGELLAN", "Magellan Health", "Commercial", False),
        ("AMERIHEALTH", "AmeriHealth Caritas", "Medicaid", True),
        ("FIDELIS", "Fidelis Care", "Medicaid", True),
        ("CARESOURCE", "CareSource", "Medicaid", True),
        ("HEALTH_NET", "Health Net", "Commercial", False),
        ("MERIDIAN", "Meridian Health Plan", "Medicaid", True),
    ]

    # Expand to ~40 rows by duplicating with slight variations
    expanded = []
    for code, name, plan, turnover in payers:
        expanded.append((code, name, plan, turnover))

    # If fewer than 40, duplicate some with suffixes
    while len(expanded) < 40:
        base = random.choice(payers)
        suffix = random.randint(1, 99)
        expanded.append(
            (f"{base[0]}_{suffix}", f"{base[1]} Plan {suffix}", base[2], base[3])
        )

    df = pd.DataFrame(
        expanded,
        columns=["payer_code", "canonical_name", "plan_type", "high_turnover"],
    )
    return df


# ---------------------------------------------------------
# Generate patients + appointments (~250 rows)
# ---------------------------------------------------------
def generate_patients_and_appointments(
    payer_master: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create:
    - patients_df: patient_id, patient_name, dob
    - appointments_df: ~250 rows with messy insurance_on_file
    """
    fake = Faker()
    Faker.seed(RANDOM_SEED)

    num_patients = 250
    patient_ids = [f"P{str(i+1).zfill(3)}" for i in range(num_patients)]

    # Generate patient demographics
    patients = []
    for pid in patient_ids:
        name = fake.name()
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        patients.append((pid, name, dob))

    patients_df = pd.DataFrame(patients, columns=["patient_id", "patient_name", "dob"])

    # Provider names
    providers = [
        "Dr. Smith",
        "Dr. Patel",
        "Dr. Johnson",
        "Dr. Lee",
        "Dr. Martinez",
    ]

    # Insurance variations (25+ messy entries)
    insurance_variants = [
        "Aetna",
        "Cigna",
        "UnitedHealthcare",
        "Humana",
        "BCBSIL",
        "UHC",
        "HCSC",
        "Blue Cross Blue Shield of Illinois",
        "Aetna PPO",
        "BCBS IL HMO",
        "UnitedHealthcare Choice Plus",
        "BCBSIL #MBR12345",
        "Aetna / ID: 987654",
        "Cigna.",
        "Humanna",
        "United Health Care",
        "Aetna.",
        "",
        None,
        "self-pay",
        "??",
        "Medicare or Medicaid",
        "BCBS / Aetna",
        "unknown",
        "cash pay",
        "see notes",
        "BCBS Texas",
        "BCBS Florida",
        "Molina Healthcare",
        "Centene",
        "WellCare",
        "Ambetter",
        "Medicaid Illinois",
        "Medicaid Texas",
        "Oscar Health",
        "Kaiser Permanente",
        "Anthem",
        "CVS/Aetna",
        "Tricare",
        "Magellan Health",
        "AmeriHealth",
        "Fidelis Care",
        "CareSource",
        "Health Net",
        "Meridian Health Plan",
    ]

    # Generate appointments
    today = datetime.now().date()
    appointments = []

    for pid, name, dob in patients:
        appt_time = datetime.combine(
            today,
            datetime.min.time(),
        ) + timedelta(hours=random.randint(8, 17), minutes=random.randint(0, 59))

        provider = random.choice(providers)
        insurance = random.choice(insurance_variants)

        # Member ID: 75% present
        if random.random() < 0.75:
            member_id = f"MBR-{random.randint(10000, 99999)}"
        else:
            member_id = ""

        appointments.append(
            (
                pid,
                name,
                dob,
                appt_time,
                provider,
                insurance,
                member_id,
            )
        )

    appointments_df = pd.DataFrame(
        appointments,
        columns=[
            "patient_id",
            "patient_name",
            "dob",
            "appointment_datetime",
            "provider_name",
            "insurance_on_file",
            "member_id",
        ],
    )

    return patients_df, appointments_df


# ---------------------------------------------------------
# Generate last_check_history (~150 rows)
# ---------------------------------------------------------
def generate_last_check_history(
    patients_df: pd.DataFrame, payer_master: pd.DataFrame
) -> pd.DataFrame:
    """
    Create ~150 rows covering ~70% of patients.
    Includes mismatched payer_code and mismatched member_id for some patients.
    """
    num_patients = len(patients_df)
    num_history = 150  # ~70% of 250

    selected_patients = random.sample(
        list(patients_df["patient_id"]), num_history
    )

    payer_codes = list(payer_master["payer_code"])

    history_rows = []
    today = datetime.now().date()

    # Choose 15 patients for member_id mismatch
    mismatch_member_patients = set(random.sample(selected_patients, 15))

    # Choose 20 patients for payer mismatch
    mismatch_payer_patients = set(random.sample(selected_patients, 20))

    for pid in selected_patients:
        # Random payer
        payer = random.choice(payer_codes)

        # If payer mismatch, choose a different payer than appointments
        if pid in mismatch_payer_patients:
            payer = random.choice(payer_codes)

        # Member ID: usually matches appointments, but mismatch for some
        if pid in mismatch_member_patients:
            member_id = f"MBR-{random.randint(10000, 99999)}"
        else:
            member_id = f"MBR-{random.randint(10000, 99999)}"

        # Random date in last 60 days
        days_back = random.randint(0, 60)
        last_check_date = today - timedelta(days=days_back)

        # Result distribution
        result = random.choices(
            ["Active", "Inactive", "Unknown"],
            weights=[0.7, 0.2, 0.1],
        )[0]

        history_rows.append(
            (pid, payer, member_id, last_check_date, result)
        )

    df = pd.DataFrame(
        history_rows,
        columns=[
            "patient_id",
            "payer_code",
            "member_id",
            "last_check_date",
            "result",
        ],
    )
    return df


# ---------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    ensure_data_dir()

    payer_master = generate_payer_master()
    patients_df, appointments_df = generate_patients_and_appointments(payer_master)
    last_check_df = generate_last_check_history(patients_df, payer_master)

    payer_master.to_csv(os.path.join(DATA_DIR, "payer_master.csv"), index=False)
    appointments_df.to_csv(os.path.join(DATA_DIR, "appointments.csv"), index=False)
    last_check_df.to_csv(os.path.join(DATA_DIR, "last_check_history.csv"), index=False)

    print("Generated payer_master.csv, appointments.csv, last_check_history.csv in ./data")


if __name__ == "__main__":
    main()