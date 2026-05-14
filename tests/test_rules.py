import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.rules import (
    _evaluate_row,
    _build_history_index,
    _build_turnover_map,
)


def build_today():
    return pd.Timestamp.today().normalize()


# ------------------------------------------------------------
# Rule 1 — Stale check (>30 days)
# ------------------------------------------------------------
def test_rule_1_stale_check_triggers_recheck():
    appt = pd.Series({
        "patient_id": "P001",
        "payer_code": "AETNA_COMM",
        "method": "exact",
        "confidence": 0.95,
        "member_id": "ABC123",
    })

    payer_master_df = pd.DataFrame({
        "payer_code": ["AETNA_COMM"],
        "high_turnover": [False],
    })

    history_df = pd.DataFrame({
        "patient_id": ["P001"],
        "payer_code": ["AETNA_COMM"],
        "member_id": ["ABC123"],
        "last_check_date": [
            (datetime.today() - timedelta(days=40)).strftime("%Y-%m-%d")
        ],
    })

    history_index = _build_history_index(history_df)
    turnover_map = _build_turnover_map(payer_master_df)

    status, reason = _evaluate_row(
        appt, history_index, turnover_map, build_today()
    )

    assert status == "Re-check needed"
    assert "40 days ago" in reason


# ------------------------------------------------------------
# Rule 2 — Payer changed
# ------------------------------------------------------------
def test_rule_2_payer_changed():
    appt = pd.Series({
        "patient_id": "P002",
        "payer_code": "UHC_COMM",
        "method": "exact",
        "confidence": 0.95,
        "member_id": "XYZ111",
    })

    payer_master_df = pd.DataFrame({
        "payer_code": ["UHC_COMM"],
        "high_turnover": [False],
    })

    history_df = pd.DataFrame({
        "patient_id": ["P002"],
        "payer_code": ["AETNA_COMM"],
        "member_id": ["XYZ111"],
        "last_check_date": [
            (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        ],
    })

    history_index = _build_history_index(history_df)
    turnover_map = _build_turnover_map(payer_master_df)

    status, reason = _evaluate_row(
        appt, history_index, turnover_map, build_today()
    )

    assert status == "Re-check needed"
    assert "was AETNA_COMM" in reason


# ------------------------------------------------------------
# Rule 3 — Member ID mismatch
# ------------------------------------------------------------
def test_rule_3_member_id_mismatch():
    appt = pd.Series({
        "patient_id": "P003",
        "payer_code": "AETNA_COMM",
        "method": "exact",
        "confidence": 0.95,
        "member_id": "NEW123",
    })

    payer_master_df = pd.DataFrame({
        "payer_code": ["AETNA_COMM"],
        "high_turnover": [False],
    })

    history_df = pd.DataFrame({
        "patient_id": ["P003"],
        "payer_code": ["AETNA_COMM"],
        "member_id": ["OLD999"],
        "last_check_date": [
            (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        ],
    })

    history_index = _build_history_index(history_df)
    turnover_map = _build_turnover_map(payer_master_df)

    status, reason = _evaluate_row(
        appt, history_index, turnover_map, build_today()
    )

    assert status == "Re-check needed"
    assert "member id" in reason.lower()   # FIXED


# ------------------------------------------------------------
# Rule 0 — Unknown insurance
# ------------------------------------------------------------
def test_unknown_insurance_returns_unknown():
    appt = pd.Series({
        "patient_id": "P004",
        "payer_code": None,
        "method": "ambiguous",
        "confidence": 0.20,
        "member_id": "",
    })

    payer_master_df = pd.DataFrame({
        "payer_code": [],
        "high_turnover": [],
    })

    # FIXED: Provide empty DataFrame with correct columns
    history_df = pd.DataFrame({
        "patient_id": [],
        "payer_code": [],
        "member_id": [],
        "last_check_date": []
    })

    history_index = _build_history_index(history_df)
    turnover_map = _build_turnover_map(payer_master_df)

    status, reason = _evaluate_row(
        appt, history_index, turnover_map, build_today()
    )

    assert status == "Unknown"
    assert "could not be identified" in reason.lower()


# ------------------------------------------------------------
# OK case — everything matches
# ------------------------------------------------------------
def test_ok_case():
    appt = pd.Series({
        "patient_id": "P005",
        "payer_code": "AETNA_COMM",
        "method": "exact",
        "confidence": 0.95,
        "member_id": "M123",
    })

    payer_master_df = pd.DataFrame({
        "payer_code": ["AETNA_COMM"],
        "high_turnover": [False],
    })

    history_df = pd.DataFrame({
        "patient_id": ["P005"],
        "payer_code": ["AETNA_COMM"],
        "member_id": ["M123"],
        "last_check_date": [
            (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")
        ],
    })

    history_index = _build_history_index(history_df)
    turnover_map = _build_turnover_map(payer_master_df)

    status, reason = _evaluate_row(
        appt, history_index, turnover_map, build_today()
    )

    assert status == "OK"
    assert "verified" in reason.lower()
