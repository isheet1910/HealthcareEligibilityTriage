"""
rules.py

Healthcare eligibility triage rule engine.

This module:
1. Loads normalized appointment data
2. Applies business rules in strict order
3. Assigns:
    - OK
    - Re-check needed
    - Unknown
4. Saves final output CSV

------------------------------------------------------------
RULE ORDER (FIRST MATCH WINS)
------------------------------------------------------------

RULE 0 - Unknown insurance
RULE 1 - No history or stale check (>30 days)
RULE 2 - Payer changed
RULE 3 - Member ID mismatch
RULE 4 - High-turnover payer stale (>14 days)
RULE 5 - Low confidence normalization

------------------------------------------------------------
RUN
------------------------------------------------------------

python src/rules.py
"""

from typing import Dict, Tuple, Optional

import pandas as pd

# ============================================================
# Configuration
# ============================================================

LOW_CONFIDENCE_THRESHOLD = 0.70
STALE_CHECK_DAYS = 30
HIGH_TURNOVER_DAYS = 14

# ============================================================
# Helper Functions
# ============================================================


def get_latest_history_row(
    patient_id: str,
    history_df: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Return latest eligibility history row for patient.
    """

    patient_history = history_df[
        history_df["patient_id"] == patient_id
    ].copy()

    if patient_history.empty:
        return None

    patient_history["last_check_date"] = pd.to_datetime(
        patient_history["last_check_date"],
        errors="coerce"
    )

    patient_history = patient_history.sort_values(
        by="last_check_date",
        ascending=False
    )

    return patient_history.iloc[0]


def get_high_turnover_flag(
    payer_code: str,
    payer_master_df: pd.DataFrame
) -> bool:
    """
    Lookup high_turnover flag from payer master.
    """

    payer_match = payer_master_df[
        payer_master_df["payer_code"] == payer_code
    ]

    if payer_match.empty:
        return False

    value = payer_match.iloc[0].get(
        "high_turnover",
        False
    )

    # Handle string values safely
    if isinstance(value, str):
        return value.strip().lower() in [
            "true",
            "1",
            "yes",
            "y"
        ]

    return bool(value)


# ============================================================
# Rule Engine
# ============================================================


def evaluate_rules(
    row: pd.Series,
    payer_master_df: pd.DataFrame,
    history_df: pd.DataFrame
) -> Tuple[str, str]:
    """
    Apply business rules in exact order.
    Returns:
        (status, reason)
    """

    today = pd.Timestamp.today().normalize()

    # --------------------------------------------------------
    # Extract fields safely
    # --------------------------------------------------------

    patient_id = row.get("patient_id")

    payer_code = row.get("payer_code")
    normalized_payer = row.get("normalized_payer")

    confidence = row.get("confidence", 0.0)
    method = str(row.get("method", "")).strip().lower()

    appointment_member_id = row.get("member_id")

    # ========================================================
    # RULE 0 - Unknown insurance
    # ========================================================

    if (
        method == "ambiguous"
        or pd.isna(payer_code)
        or payer_code is None
        or pd.isna(normalized_payer)
    ):

        return (
            "Unknown",
            "Insurance on file could not be identified"
        )

    # ========================================================
    # Get latest history row
    # ========================================================

    latest_history = get_latest_history_row(
        patient_id,
        history_df
    )

    # ========================================================
    # RULE 1 - No history or stale check
    # ========================================================

    if latest_history is None:

        return (
            "Re-check needed",
            "No eligibility check on record for this patient"
        )

    last_check_date = pd.to_datetime(
        latest_history.get("last_check_date"),
        errors="coerce"
    )

    if pd.isna(last_check_date):

        return (
            "Re-check needed",
            "Last eligibility check date is invalid"
        )

    days_ago = (today - last_check_date.normalize()).days

    if days_ago > STALE_CHECK_DAYS:

        return (
            "Re-check needed",
            f"Last eligibility check was {days_ago} days ago (limit: {STALE_CHECK_DAYS} days)"
        )

    # ========================================================
    # RULE 2 - Payer changed
    # ========================================================

    history_payer_code = latest_history.get(
        "payer_code"
    )

    current_payer_code = payer_code

    history_payer_exists = (
        not pd.isna(history_payer_code)
        and str(history_payer_code).strip() != ""
    )

    current_payer_exists = (
        not pd.isna(current_payer_code)
        and str(current_payer_code).strip() != ""
    )

    if (
        history_payer_exists
        and current_payer_exists
        and str(history_payer_code).strip()
        != str(current_payer_code).strip()
    ):

        return (
            "Re-check needed",
            f"Insurance changed: was {history_payer_code}, now {current_payer_code}"
        )

    # ========================================================
    # RULE 3 - Member ID mismatch
    # ========================================================

    history_member_id = latest_history.get(
        "member_id"
    )

    appointment_has_member_id = (
        not pd.isna(appointment_member_id)
        and str(appointment_member_id).strip() != ""
    )

    history_has_member_id = (
        not pd.isna(history_member_id)
        and str(history_member_id).strip() != ""
    )

    if (
        appointment_has_member_id
        and history_has_member_id
        and str(appointment_member_id).strip()
        != str(history_member_id).strip()
    ):

        return (
            "Re-check needed",
            "Member ID on file does not match last verified member ID"
        )

    # ========================================================
    # RULE 4 - High turnover payer stale
    # ========================================================

    is_high_turnover = get_high_turnover_flag(
        payer_code,
        payer_master_df
    )

    if (
        is_high_turnover
        and days_ago > HIGH_TURNOVER_DAYS
    ):

        return (
            "Re-check needed",
            f"High-turnover payer: last check was {days_ago} days ago (limit: {HIGH_TURNOVER_DAYS} days)"
        )

    # ========================================================
    # RULE 5 - Low confidence
    # ========================================================

    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0

    if confidence < LOW_CONFIDENCE_THRESHOLD:

        return (
            "Re-check needed",
            f"Low confidence insurance match ({confidence:.0%}) - manual verification needed"
        )

    # ========================================================
    # ALL RULES PASSED
    # ========================================================

    return (
        "OK",
        "Eligibility verified within required timeframe"
    )


# ============================================================
# Batch Rule Application
# ============================================================


def apply_rules(
    normalized_df: pd.DataFrame,
    payer_master_df: pd.DataFrame,
    history_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply rules to all appointments.
    """

    results = []

    for _, row in normalized_df.iterrows():

        status, reason = evaluate_rules(
            row,
            payer_master_df,
            history_df
        )

        results.append({
            "status": status,
            "reason": reason
        })

    results_df = pd.DataFrame(results)

    final_df = pd.concat(
        [
            normalized_df.reset_index(drop=True),
            results_df
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Print Summary
    # --------------------------------------------------------

    print("\n===== STATUS SUMMARY =====\n")
    print(final_df["status"].value_counts())

    return final_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nLoading datasets...")

    # --------------------------------------------------------
    # Load Files
    # --------------------------------------------------------

    normalized_df = pd.read_csv(
        "data/c2p1/appointments_normalized.csv"
    )

    payer_master_df = pd.read_csv(
        "data/c2/payer_master.csv"
    )

    history_df = pd.read_csv(
        "data/c2/last_check_history.csv"
    )

    print("\nRunning rule engine...\n")

    # --------------------------------------------------------
    # Apply Rules
    # --------------------------------------------------------

    final_df = apply_rules(
        normalized_df,
        payer_master_df,
        history_df
    )

    # --------------------------------------------------------
    # Save Output
    # --------------------------------------------------------

    output_path = (
        "data/c2p1/appointments_final.csv"
    )

    final_df.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Display Sample Results
    # --------------------------------------------------------

    print("\n===== SAMPLE RESULTS =====\n")

    display_columns = [
        "patient_name",
        "provider_name",
        "insurance_on_file",
        "normalized_payer",
        "status",
        "reason"
    ]

    print(
        final_df[display_columns]
        .head(20)
        .to_string(index=False)
    )

    print("\nRule engine complete.")
    print(f"Saved output to: {output_path}")