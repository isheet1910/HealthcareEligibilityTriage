import os
import random
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
from faker import Faker

# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
DATA_DIR = "././data/c1"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker()
Faker.seed(RANDOM_SEED)

# ============================================================
# Utility Functions
# ============================================================


def ensure_data_dir():
    """Create the data directory if it does not exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def random_member_id() -> str:
    """Generate a random member ID."""
    letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=3))
    numbers = "".join(random.choices("0123456789", k=5))
    return f"{letters}{numbers}"


def random_dob(min_age: int = 18, max_age: int = 85):
    """Generate a realistic adult date of birth."""
    today = datetime.today().date()

    age = random.randint(min_age, max_age)
    days_offset = random.randint(0, 364)

    dob = today - timedelta(days=(age * 365 + days_offset))
    return dob


def appointment_time_today():
    """
    Generate an appointment datetime for today
    between 8 AM and 5 PM.
    """
    today = datetime.today()

    hour = random.randint(8, 16)
    minute = random.choice([0, 15, 30, 45])

    return today.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0
    )


# ============================================================
# Payer Master Generation
# ============================================================


def generate_payer_master() -> pd.DataFrame:
    """
    Generate payer master reference data.
    """

    payer_rows = [
        # Commercial
        ("BCBS_IL_PPO", "Blue Cross Blue Shield of Illinois PPO", "Commercial", True),
        ("BCBS_IL_HMO", "Blue Cross Blue Shield of Illinois HMO", "Commercial", False),
        ("BCBS_TX_HMO", "Blue Cross Blue Shield of Texas HMO", "Commercial", True),
        ("BCBS_FL_COMM", "Blue Cross Blue Shield of Florida", "Commercial", False),
        ("AETNA_COMM", "Aetna Commercial", "Commercial", False),
        ("AETNA_PPO", "Aetna PPO", "Commercial", False),
        ("UHC_COMM", "UnitedHealthcare Commercial", "Commercial", False),
        ("UHC_CHOICE", "UnitedHealthcare Choice Plus", "Commercial", False),
        ("CIGNA_COMM", "Cigna Commercial", "Commercial", False),
        ("HUMANA_COMM", "Humana Commercial", "Commercial", False),
        ("OSCAR_COMM", "Oscar Health", "Commercial", False),
        ("KAISER_COMM", "Kaiser Permanente", "Commercial", False),
        ("ANTHEM_COMM", "Anthem Commercial", "Commercial", False),
        ("HEALTHNET_COMM", "Health Net Commercial", "Commercial", False),
        ("MAGELLAN_COMM", "Magellan Health", "Commercial", False),
        ("AMERIHEALTH", "AmeriHealth", "Commercial", False),
        ("FIDELIS_COMM", "Fidelis Care", "Commercial", False),
        ("CARESOURCE_COMM", "CareSource", "Commercial", False),
        ("CVS_AETNA", "CVS Aetna", "Commercial", False),
        ("TRICARE_COMM", "Tricare", "Commercial", False),

        # Medicare
        ("MEDICARE_A", "Traditional Medicare Part A", "Medicare", False),
        ("MEDICARE_B", "Traditional Medicare Part B", "Medicare", False),
        ("MEDICARE_SUPP", "Medicare Supplement", "Medicare", False),

        # Medicare Advantage
        ("HUMANA_MA", "Humana Medicare Advantage", "Medicare Advantage", False),
        ("UHC_MA", "UnitedHealthcare Medicare Advantage", "Medicare Advantage", False),
        ("AETNA_MA", "Aetna Medicare Advantage", "Medicare Advantage", False),
        ("BCBS_MA", "BCBS Medicare Advantage", "Medicare Advantage", False),
        ("WELLCARE_MA", "WellCare Medicare Advantage", "Medicare Advantage", True),

        # Medicaid
        ("MEDICAID_IL", "Illinois Medicaid", "Medicaid", True),
        ("MEDICAID_TX", "Texas Medicaid", "Medicaid", True),
        ("MOLINA_MED", "Molina Healthcare", "Medicaid", True),
        ("CENTENE_MED", "Centene Medicaid", "Medicaid", True),
        ("AMBETTER_MED", "Ambetter", "Medicaid", True),
        ("MERIDIAN_MED", "Meridian Health Plan", "Medicaid", True),
        ("WELLCARE_MED", "WellCare Medicaid", "Medicaid", True),
        ("CARESOURCE_MED", "CareSource Medicaid", "Medicaid", True),
        ("FIDELIS_MED", "Fidelis Medicaid", "Medicaid", True),
        ("AMERIHEALTH_MED", "AmeriHealth Medicaid", "Medicaid", True),
        ("HEALTHNET_MED", "Health Net Medicaid", "Medicaid", True),
        ("MOLINA_IL", "Molina Illinois", "Medicaid", True),
    ]

    payer_master = pd.DataFrame(
        payer_rows,
        columns=[
            "payer_code",
            "canonical_name",
            "plan_type",
            "high_turnover"
        ]
    )

    return payer_master


# ============================================================
# Insurance Variations
# ============================================================


def build_insurance_variations(member_id: str) -> List[str]:
    """
    Create messy insurance name variations.
    """

    variations = [
        # Clean names
        "Aetna",
        "Cigna",
        "UnitedHealthcare",
        "Humana",
        "Molina Healthcare",
        "Oscar Health",
        "Anthem",
        "Kaiser Permanente",
        "CareSource",
        "Health Net",

        # Abbreviations
        "BCBSIL",
        "BCBS IL",
        "UHC",
        "HCSC",

        # Long names
        "Blue Cross Blue Shield of Illinois",
        "Blue Cross Blue Shield of Texas",
        "Blue Cross Blue Shield of Florida",
        "Aetna Commercial PPO",
        "UnitedHealthcare Choice Plus",
        "Humana Medicare Advantage",

        # Plan tiers
        "BCBS IL PPO",
        "BCBS IL HMO",
        "Aetna PPO",
        "Cigna HMO",
        "UHC Choice Plus",
        "Anthem Gold Plan",

        # Member IDs embedded
        f"BCBSIL #{member_id}",
        f"Aetna / ID: {member_id}",
        f"UHC Member {member_id}",
        f"Cigna - {member_id}",

        # Typos
        "Blue Cros Blu Sheild IL",
        "Humanna",
        "United Health Care",
        "Aetna.",
        "Cingna",

        # Ambiguous values
        "self-pay",
        "??",
        "unknown",
        "cash pay",
        "see notes",
        "BCBS or Aetna?",
        "Medicare or Medicaid",
        "BCBS/UHC",

        # Medicaid / Medicare
        "Medicaid Illinois",
        "Medicaid Texas",
        "Medicare",
        "WellCare",
        "Ambetter",
        "Centene",
        "Tricare",
        "AmeriHealth",
        "Fidelis Care",
        "Magellan Health",
        "Meridian Health Plan",
        "CVS/Aetna",
    ]

    return variations


# ============================================================
# Appointments Generation
# ============================================================


def generate_patients_and_appointments(
    payer_master: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic patient and appointment data.
    """

    providers = [
        "Dr. Emily Smith",
        "Dr. Raj Patel",
        "Dr. Michael Johnson",
        "Dr. Sarah Lee",
        "Dr. Kevin Brown",
        "Dr. Lisa Garcia",
    ]

    patient_rows = []
    appointment_rows = []

    for i in range(1, 251):

        patient_id = f"P{i:03d}"
        patient_name = fake.name()
        dob = random_dob()

        # Member ID present ~75% of the time
        if random.random() < 0.75:
            member_id = random_member_id()
        else:
            member_id = ""

        insurance_variations = build_insurance_variations(member_id)

        # About 3% missing insurance values
        if random.random() < 0.03:
            insurance_value = random.choice(["", None])

        else:
            insurance_value = random.choice(insurance_variations)

        appointment_row = {
            "patient_id": patient_id,
            "patient_name": patient_name,
            "dob": dob,
            "appointment_datetime": appointment_time_today(),
            "provider_name": random.choice(providers),
            "insurance_on_file": insurance_value,
            "member_id": member_id
        }

        patient_row = {
            "patient_id": patient_id,
            "member_id": member_id
        }

        appointment_rows.append(appointment_row)
        patient_rows.append(patient_row)

    appointments_df = pd.DataFrame(appointment_rows)
    patients_df = pd.DataFrame(patient_rows)

    return patients_df, appointments_df


# ============================================================
# Last Check History Generation
# ============================================================


def generate_last_check_history(
    patients_df: pd.DataFrame,
    payer_master: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate eligibility check history records.
    """

    history_rows = []

    all_patient_ids = patients_df["patient_id"].tolist()

    # Roughly 70% coverage
    history_patients = random.sample(all_patient_ids, 175)

    payer_codes = payer_master["payer_code"].tolist()

    # Patients with changed member IDs
    changed_member_patients = set(random.sample(history_patients, 15))

    for patient_id in history_patients:

        patient_row = patients_df[
            patients_df["patient_id"] == patient_id
        ].iloc[0]

        current_member_id = patient_row["member_id"]

        # Change member ID for some patients
        if patient_id in changed_member_patients:
            history_member_id = random_member_id()
        else:
            history_member_id = current_member_id

        # Sometimes still empty
        if random.random() < 0.10:
            history_member_id = ""

        days_ago = random.randint(0, 60)

        last_check_date = (
            datetime.today() - timedelta(days=days_ago)
        ).date()

        result = random.choices(
            ["Active", "Inactive", "Unknown"],
            weights=[70, 20, 10],
            k=1
        )[0]

        row = {
            "patient_id": patient_id,
            "payer_code": random.choice(payer_codes),
            "member_id": history_member_id,
            "last_check_date": last_check_date,
            "result": result
        }

        history_rows.append(row)

    last_check_df = pd.DataFrame(history_rows)

    return last_check_df


# ============================================================
# Main
# ============================================================


def main():
    """
    Main orchestration function.
    """

    ensure_data_dir()

    # Generate all datasets
    payer_master = generate_payer_master()

    patients_df, appointments_df = generate_patients_and_appointments(
        payer_master
    )

    last_check_df = generate_last_check_history(
        patients_df,
        payer_master
    )

    # Save CSV files
    payer_master.to_csv(
        os.path.join(DATA_DIR, "payer_master.csv"),
        index=False
    )

    appointments_df.to_csv(
        os.path.join(DATA_DIR, "appointments.csv"),
        index=False
    )

    last_check_df.to_csv(
        os.path.join(DATA_DIR, "last_check_history.csv"),
        index=False
    )

    # Print summary
    print("\nSynthetic data generation complete.\n")

    print(f"payer_master.csv rows: {len(payer_master)}")
    print(f"appointments.csv rows: {len(appointments_df)}")
    print(f"last_check_history.csv rows: {len(last_check_df)}")

    print(f"\nFiles saved to ./{DATA_DIR}/")


if __name__ == "__main__":
    main()