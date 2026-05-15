"""
rules.py
========
Eligibility triage rule engine.

Takes normalized appointments + payer master + history.
Applies 5 business rules in strict order (first match wins). after checking if it is   Unknown / unresolvable insurance   → Unknown
Returns the appointments DataFrame with two new columns: status, reason.

Rules:
    Rule 1   No history, or last check > 30d    → Re-check needed
    Rule 2   Payer changed since last check      → Re-check needed
    Rule 3   Member ID mismatch                  → Re-check needed
    Rule 4   High-turnover payer + check > 14d  → Re-check needed
    Rule 5   Low confidence match (< 80%)        → Re-check needed
    Default  all clear                           → OK

No Pydantic here — the rule engine works entirely with plain pandas
DataFrames and Python types. Pydantic lives in normalize.py where it
validates data coming IN from the LLM. Here the data is already clean.

Run:
    python src/rules.py
"""

from typing import Dict, Optional, Tuple

import pandas as pd

# ============================================================
# Configuration
# ============================================================

STALE_DAYS_DEFAULT   = 30    # Rule 1 threshold
STALE_DAYS_HIGH_TURN = 14    # Rule 4 threshold
LOW_CONF_THRESHOLD   = 0.80  # Rule 5 threshold

# ============================================================
# Pre-computation helpers
# Called once before the row loop — not once per row.
# ============================================================


def _build_history_index(history_df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Build {patient_id: most_recent_history_row} for O(1) lookups.

    Dates are parsed ONCE here for the whole DataFrame rather than
    calling pd.to_datetime 250 times inside the row loop.
    """
    df = history_df.copy() #access all the data from lastcheckhistory file build disctionary once and no longer need to access the file for each row in the loop
    df["last_check_date"] = pd.to_datetime(df["last_check_date"], errors="coerce") #parse the date and convert it to datetime format
    df = df.sort_values("last_check_date", ascending=False) #sort the data by date, with the most recent check first
    df = df.drop_duplicates(subset="patient_id", keep="first") #drop the duplicates
    return {row["patient_id"]: row for _, row in df.iterrows()}


def _build_turnover_map(payer_master_df: pd.DataFrame) -> Dict[str, bool]:
    """
    Build {payer_code: high_turnover} for O(1) lookups.
    """
    return dict(
        zip(
            payer_master_df["payer_code"],
            payer_master_df["high_turnover"].astype(bool),
        )
    )


# ============================================================
# Null safety helper
# ============================================================


def _is_blank(value) -> bool:
    """True if value is NaN, None, empty string, or 'nan'/'None' strings."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in ("", "nan", "none")


# ============================================================
# Core rule evaluation — one row → (status, reason)
# ============================================================


def _evaluate_row(
    row: pd.Series,
    history_index: Dict[str, pd.Series],
    turnover_map: Dict[str, bool],
    today: pd.Timestamp,
) -> Tuple[str, str]:
    """
    Apply all 5 rules to one appointment row.
    Returns (status, reason). First rule that matches wins.
    """

    patient_id = row.get("patient_id")
    payer_code = row.get("payer_code")
    method     = str(row.get("method", "")).strip().lower()
    confidence = float(row.get("confidence", 0.0))
    member_id  = row.get("member_id")

    # ----------------------------------------------------------
    # first check for Unknown insurance
    # Can't run any other check if we don't know the payer. basically it is unkown and not reliable so we need to do manual review
    # ----------------------------------------------------------
    if method == "ambiguous" or _is_blank(payer_code):
        return (
            "Unknown",
            "Insurance on file could not be identified — manual review required",
        )

    # ----------------------------------------------------------
    # Look up this patient's most recent history row.
    # Shared by Rules 1–4 so we only fetch it once.
    # ----------------------------------------------------------
    history: Optional[pd.Series] = history_index.get(patient_id)

    # ----------------------------------------------------------
    # Rule 1a – No history record at all
    # ----------------------------------------------------------
    if history is None:
        return (
            "Re-check needed",
            "No eligibility check on record for this patient",
        )

    last_check_date = history["last_check_date"]
    days_ago: Optional[int] = (
        int((today - last_check_date).days)
        if pd.notna(last_check_date)
        else None
    )

    # ----------------------------------------------------------
    # Rule 1b – History exists but is stale (> 30 days)
    # ----------------------------------------------------------
    if days_ago is not None and days_ago > STALE_DAYS_DEFAULT:
        return (
            "Re-check needed",
            f"Last eligibility check was {days_ago} days ago "
            f"(limit: {STALE_DAYS_DEFAULT} days)",
        )

    # ----------------------------------------------------------
    # Rule 2 – Payer changed since last check
    # ----------------------------------------------------------
    history_payer = history.get("payer_code")

    if (
        not _is_blank(history_payer)
        and not _is_blank(payer_code)
        and str(payer_code).strip() != str(history_payer).strip()
    ):
        return (
            "Re-check needed",
            f"Insurance changed: was {history_payer}, now {payer_code}",
        )

    # ----------------------------------------------------------
    # Rule 3 – Member ID mismatch
    # Only fires if BOTH sides have an ID (can't compare blanks).
    # ----------------------------------------------------------
    history_member = history.get("member_id")

    if (
        not _is_blank(member_id)
        and not _is_blank(history_member)
        and str(member_id).strip() != str(history_member).strip()
    ):
        return (
            "Re-check needed",
            "Member ID on file does not match last verified member ID",
        )

    # ----------------------------------------------------------
    # Rule 4 – High-turnover payer + check older than 14 days
    # ----------------------------------------------------------
    is_high_turnover = turnover_map.get(str(payer_code).strip(), False)

    if (
        is_high_turnover
        and days_ago is not None
        and days_ago > STALE_DAYS_HIGH_TURN
    ):
        return (
            "Re-check needed",
            f"High-turnover payer: last check was {days_ago} days ago "
            f"(limit: {STALE_DAYS_HIGH_TURN} days)",
        )

    # ----------------------------------------------------------
    # Rule 5 – Low confidence in insurance match
    # ----------------------------------------------------------
    if confidence < LOW_CONF_THRESHOLD:
        pct = int(confidence * 100)
        return (
            "Re-check needed",
            f"Low confidence insurance match ({pct}%) — manual verification needed",
        )

    # ----------------------------------------------------------
    # All rules passed — eligibility is current
    # ----------------------------------------------------------
    return (
        "OK",
        f"Eligibility verified {days_ago} days ago — payer and member ID match",
    )


# ============================================================
# Main to apply rules to the whole DataFrame
# ============================================================


def apply_rules(
    normalized_df: pd.DataFrame,
    payer_master_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply eligibility triage rules to every appointment row.

    Parameters
    ----------
    normalized_df   : Output from normalize.py
    payer_master_df : payer_master.csv
    history_df      : last_check_history.csv

    Returns
    -------
    normalized_df with two new columns: status, reason
    """
    today = pd.Timestamp.today().normalize()

    history_index = _build_history_index(history_df)
    turnover_map  = _build_turnover_map(payer_master_df)

    statuses, reasons = [], []

    for _, row in normalized_df.iterrows():
        status, reason = _evaluate_row(
            row, history_index, turnover_map, today
        )
        statuses.append(status)
        reasons.append(reason)

    result_df = normalized_df.copy()
    result_df["status"] = statuses
    result_df["reason"] = reasons

    return result_df


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    import os

    DATA_DIR      = "data"
    NORMALIZED    = os.path.join(DATA_DIR, "appointments_normalized.csv")
    PAYER_MASTER  = os.path.join(DATA_DIR, "payer_master.csv")
    HISTORY       = os.path.join(DATA_DIR, "last_check_history.csv")
    OUTPUT        = os.path.join(DATA_DIR, "appointments_final.csv")

    print("\nLoading data...")
    normalized_df   = pd.read_csv(NORMALIZED)
    payer_master_df = pd.read_csv(PAYER_MASTER)
    history_df      = pd.read_csv(HISTORY)

    print(f"  appointments_normalized : {len(normalized_df)} rows")
    print(f"  payer_master            : {len(payer_master_df)} rows")
    print(f"  last_check_history      : {len(history_df)} rows")

    print("\nApplying triage rules...")
    final_df = apply_rules(normalized_df, payer_master_df, history_df)

    final_df.to_csv(OUTPUT, index=False)
    print(f"Saved → {OUTPUT}")

    print("\n===== STATUS SUMMARY =====")
    counts = final_df["status"].value_counts()
    total  = len(final_df)
    for status, count in counts.items():
        pct = round(count / total * 100, 1)
        print(f"  {status:<20} {count:>4}  ({pct}%)")
    print(f"  {'─' * 30}")
    print(f"  {'TOTAL':<20} {total:>4}")

    # print("\n===== SAMPLE RESULTS =====")
    # print(
    #     final_df[["patient_name", "insurance_on_file", "status", "reason"]]
    #     .head(20)
    #     .to_string(index=False, max_colwidth=55)
    # )