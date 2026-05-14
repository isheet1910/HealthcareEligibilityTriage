"""
normalize.py

Production-style payer normalization pipeline.

Architecture:
clean_text
    ↓
exact_match
    ↓
alias_match
    ↓
fuzzy_match
    ↓
low confidence?
    ↓
LLM cache lookup
    ↓
OpenRouter LLM fallback
    ↓
cache result
    ↓
return normalized payer

------------------------------------------------------------
INSTALLATION
------------------------------------------------------------

pip install pandas rapidfuzz openai python-dotenv

------------------------------------------------------------
ENVIRONMENT VARIABLES
------------------------------------------------------------

Create a .env file in your repo root:

OPENROUTER_API_KEY=your_api_key_here

Get free API key:
https://openrouter.ai

------------------------------------------------------------
RUN
------------------------------------------------------------

python src/normalize.py
"""

import os
import re
from typing import Dict, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from rapidfuzz import fuzz, process

# ============================================================
# Environment Setup
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "Missing OPENROUTER_API_KEY in .env file"
    )

# OpenRouter client
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ============================================================
# Configuration
# ============================================================

LOW_CONFIDENCE_THRESHOLD = 0.75

# Cache to avoid repeated LLM calls
LLM_CACHE = {}

# Ambiguous values
AMBIGUOUS_VALUES = {
    "",
    "??",
    "unknown",
    "self pay",
    "self-pay",
    "cash pay",
    "see notes",
    "medicare or medicaid",
    "bcbs or aetna",
    "bcbs/uhc",
    "none",
    "nan",
}

# ============================================================
# Alias Dictionary
# ============================================================

PAYER_ALIASES = {

    # --------------------------------------------------------
    # BCBS Illinois
    # --------------------------------------------------------
    "bcbsil": "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il": "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il ppo": "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il hmo": "Blue Cross Blue Shield of Illinois HMO",
    "blue cross illinois":
        "Blue Cross Blue Shield of Illinois PPO",
    "blue cross blue shield illinois":
        "Blue Cross Blue Shield of Illinois PPO",
    "blue cross blue shield of illinois":
        "Blue Cross Blue Shield of Illinois PPO",
    "blue cros blu sheild il":
        "Blue Cross Blue Shield of Illinois PPO",

    # --------------------------------------------------------
    # BCBS Texas
    # --------------------------------------------------------
    "bcbs texas":
        "Blue Cross Blue Shield of Texas HMO",
    "bcbs tx":
        "Blue Cross Blue Shield of Texas HMO",
    "bcbs tx hmo":
        "Blue Cross Blue Shield of Texas HMO",

    # --------------------------------------------------------
    # UnitedHealthcare
    # --------------------------------------------------------
    "uhc":
        "UnitedHealthcare Commercial",
    "uhc choice plus":
        "UnitedHealthcare Choice Plus",
    "united healthcare":
        "UnitedHealthcare Commercial",
    "united health care":
        "UnitedHealthcare Commercial",
    "unitedhealthcare":
        "UnitedHealthcare Commercial",

    # --------------------------------------------------------
    # Aetna
    # --------------------------------------------------------
    "aetna":
        "Aetna Commercial",
    "aetna ppo":
        "Aetna PPO",
    "atena":
        "Aetna Commercial",
    "atena ppo":
        "Aetna PPO",

    # --------------------------------------------------------
    # Cigna
    # --------------------------------------------------------
    "cigna":
        "Cigna Commercial",
    "cingna":
        "Cigna Commercial",

    # --------------------------------------------------------
    # Humana
    # --------------------------------------------------------
    "humana":
        "Humana Commercial",
    "humanna":
        "Humana Commercial",

    # --------------------------------------------------------
    # Medicaid / Managed Care
    # --------------------------------------------------------
    "molina":
        "Molina Healthcare",
    "wellcare":
        "WellCare Medicaid",
    "ambetter":
        "Ambetter",
    "centene":
        "Centene Medicaid",
    "caresource":
        "CareSource Medicaid",
    "amerihealth":
        "AmeriHealth Medicaid",

    # --------------------------------------------------------
    # Medicare / Medicaid
    # --------------------------------------------------------
    "medicare":
        "Traditional Medicare Part A",
    "medicaid illinois":
        "Illinois Medicaid",
    "medicaid texas":
        "Texas Medicaid",

    # --------------------------------------------------------
    # Other Commercial
    # --------------------------------------------------------
    "anthem":
        "Anthem Commercial",
    "oscar":
        "Oscar Health",
    "kaiser":
        "Kaiser Permanente",
    "health net":
        "Health Net Commercial",
    "tricare":
        "Tricare",
}

# ============================================================
# Text Cleaning
# ============================================================


def clean_text(text: str) -> str:
    """
    Clean insurance text before matching.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Remove embedded member IDs
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"id[:\s\-]*\w+", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Exact Matching
# ============================================================


def exact_match(
    cleaned_text: str,
    payer_master_df: pd.DataFrame
) -> Optional[Dict]:

    # --------------------------------------------------------
    # Ambiguous values
    # --------------------------------------------------------
    if cleaned_text in AMBIGUOUS_VALUES:

        return {
            "normalized_payer": None,
            "payer_code": None,
            "confidence": 0.10,
            "method": "ambiguous"
        }

    # --------------------------------------------------------
    # Alias matching
    # --------------------------------------------------------
    if cleaned_text in PAYER_ALIASES:

        canonical_name = PAYER_ALIASES[cleaned_text]

        matched = payer_master_df[
            payer_master_df["canonical_name"] == canonical_name
        ]

        if not matched.empty:

            row = matched.iloc[0]

            return {
                "normalized_payer": row["canonical_name"],
                "payer_code": row["payer_code"],
                "confidence": 0.99,
                "method": "alias_exact"
            }

    # --------------------------------------------------------
    # Exact canonical matching
    # --------------------------------------------------------
    for _, row in payer_master_df.iterrows():

        canonical_clean = clean_text(
            row["canonical_name"]
        )

        if cleaned_text == canonical_clean:

            return {
                "normalized_payer": row["canonical_name"],
                "payer_code": row["payer_code"],
                "confidence": 1.00,
                "method": "exact"
            }

    return None


# ============================================================
# Fuzzy Matching
# ============================================================


def fuzzy_match(
    cleaned_text: str,
    payer_master_df: pd.DataFrame,
    threshold: int = 75
) -> Dict:

    canonical_names = payer_master_df[
        "canonical_name"
    ].tolist()

    cleaned_lookup = {
        clean_text(name): name
        for name in canonical_names
    }

    cleaned_choices = list(cleaned_lookup.keys())

    match = process.extractOne(
        cleaned_text,
        cleaned_choices,
        scorer=fuzz.token_sort_ratio
    )

    # No match found
    if not match:

        return {
            "normalized_payer": None,
            "payer_code": None,
            "confidence": 0.0,
            "method": "no_match"
        }

    matched_clean_name, score, _ = match

    confidence = round(score / 100, 2)

    # Below threshold
    if score < threshold:

        return {
            "normalized_payer": None,
            "payer_code": None,
            "confidence": confidence,
            "method": "low_confidence"
        }

    canonical_name = cleaned_lookup[
        matched_clean_name
    ]

    matched_row = payer_master_df[
        payer_master_df["canonical_name"] == canonical_name
    ].iloc[0]

    return {
        "normalized_payer":
            matched_row["canonical_name"],
        "payer_code":
            matched_row["payer_code"],
        "confidence":
            confidence,
        "method":
            "fuzzy"
    }


# ============================================================
# LLM Fallback
# ============================================================


def llm_match_fallback(
    original_text: str,
    payer_master_df: pd.DataFrame
) -> Dict:
    """
    Use OpenRouter free LLM for low-confidence cases.
    Hardened against: empty choices, None choices, missing content.
    """

    cleaned = clean_text(original_text)

    # --------------------------------------------------------
    # Cache lookup
    # --------------------------------------------------------
    if cleaned in LLM_CACHE:
        return LLM_CACHE[cleaned]

    try:
        payer_names = payer_master_df["canonical_name"].tolist()
        payer_list_text = "\n".join([f"- {name}" for name in payer_names])

        system_prompt = """
You normalize messy insurance payer names.

Rules:
- Choose ONLY from the provided payer names.
- If unclear, return UNKNOWN.
- Return ONLY the payer name, nothing else.
"""

        user_prompt = f"""
Messy payer:
{original_text}

Valid payers:
{payer_list_text}
"""

        response = client.chat.completions.create(
            # model="google/gemma-3-27b-it:free",
            # model="meta-llama/llama-3.1-8b-instruct:free",
            # model="mistralai/mistral-7b-instruct:free",
            # model="nvidia/nemotron-nano-9b-v2:free",
            # model="qwen/qwen2.5-7b-instruct:free",
            # model="openrouter/free",
            model="openrouter/free",
            temperature=0,
            max_tokens=30,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ]
        )

        # --------------------------------------------------------
        # Guard 1: choices is None or empty list
        # --------------------------------------------------------
        if not response.choices:
            result = {
                "normalized_payer": None,
                "payer_code":       None,
                "confidence":       0.25,
                "method":           "llm_empty_response"
            }
            LLM_CACHE[cleaned] = result
            return result

        # --------------------------------------------------------
        # Guard 2: message or content is None
        # --------------------------------------------------------
        message = response.choices[0].message
        content = ""

        if message is not None and message.content is not None:
            content = str(message.content).strip()

        if not content:
            result = {
                "normalized_payer": None,
                "payer_code":       None,
                "confidence":       0.25,
                "method":           "llm_empty_response"
            }
            LLM_CACHE[cleaned] = result
            return result

        # --------------------------------------------------------
        # Guard 3: model said UNKNOWN
        # --------------------------------------------------------
        if content.upper() == "UNKNOWN":
            result = {
                "normalized_payer": None,
                "payer_code":       None,
                "confidence":       0.30,
                "method":           "llm_unknown"
            }
            LLM_CACHE[cleaned] = result
            return result

        # --------------------------------------------------------
        # Match returned name against payer master
        # --------------------------------------------------------
        matched_rows = payer_master_df[
            payer_master_df["canonical_name"].str.lower() == content.lower()
        ]

        if matched_rows.empty:
            result = {
                "normalized_payer": None,
                "payer_code":       None,
                "confidence":       0.20,
                "method":           "llm_invalid"
            }
            LLM_CACHE[cleaned] = result
            return result

        row = matched_rows.iloc[0]
        result = {
            "normalized_payer": row["canonical_name"],
            "payer_code":       row["payer_code"],
            "confidence":       0.78,
            "method":           "llm"
        }
        LLM_CACHE[cleaned] = result
        return result

    except Exception as e:
        print(f"LLM fallback error: {e}")
        result = {
            "normalized_payer": None,
            "payer_code":       None,
            "confidence":       0.0,
            "method":           "llm_error"
        }
        LLM_CACHE[cleaned] = result
        return result


# ============================================================
# Main Pipeline
# ============================================================


def normalize_payer(
    insurance_value: str,
    payer_master_df: pd.DataFrame
) -> Dict:
    """
    Main normalization pipeline.
    """

    # --------------------------------------------------------
    # Step 1: Clean text
    # --------------------------------------------------------
    cleaned_text = clean_text(
        insurance_value
    )

    # --------------------------------------------------------
    # Step 2: Deterministic matching
    # --------------------------------------------------------
    exact_result = exact_match(
        cleaned_text,
        payer_master_df
    )

    if exact_result:
        return exact_result

    # --------------------------------------------------------
    # Step 3: Fuzzy matching
    # --------------------------------------------------------
    fuzzy_result = fuzzy_match(
        cleaned_text,
        payer_master_df
    )

    # --------------------------------------------------------
    # Step 4: LLM fallback
    # --------------------------------------------------------
    if fuzzy_result["confidence"] < LOW_CONFIDENCE_THRESHOLD:

        return llm_match_fallback(
            insurance_value,
            payer_master_df
        )

    return fuzzy_result


# ============================================================
# Batch Processing
# ============================================================


def normalize_appointments(
    appointments_df: pd.DataFrame,
    payer_master_df: pd.DataFrame
) -> pd.DataFrame:

    results = []

    for insurance_value in appointments_df[
        "insurance_on_file"
    ]:

        result = normalize_payer(
            insurance_value,
            payer_master_df
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    final_df = pd.concat(
        [
            appointments_df.reset_index(drop=True),
            results_df
        ],
        axis=1
    )

    return final_df


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\nLoading datasets...")

    appointments_df = pd.read_csv(
        "data/c2/appointments.csv"
    )

    payer_master_df = pd.read_csv(
        "data/c2/payer_master.csv"
    )

    print("\nRunning normalization pipeline...\n")

    normalized_df = normalize_appointments(
        appointments_df,
        payer_master_df
    )

    # --------------------------------------------------------
    # Sample Test Cases
    # --------------------------------------------------------
    sample_values = [
        "BCBS IL PPO",
        "BCBSIL #X123",
        "Blue Cross Illinois",
        "Blue Cros Blu Sheild IL",
        "UHC",
        "United Health Care",
        "Atena PPO",
        "Humanna",
        "Cingna",
        "self-pay",
        "??",
        "cash pay",
        "BCBS or Aetna?",
        "Aetna / ID: 12345"
    ]

    print("\n===== SAMPLE RESULTS =====\n")

    for value in sample_values:

        result = normalize_payer(
            value,
            payer_master_df
        )

        print(f"Input: {value}")
        print(f"Normalized Payer: {result['normalized_payer']}")
        print(f"Payer Code: {result['payer_code']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Method: {result['method']}")
        print("-" * 60)

    # --------------------------------------------------------
    # Save Output
    # --------------------------------------------------------
    output_path = (
        "data/c2p1/appointments_normalized.csv"
    )

    normalized_df.to_csv(
        output_path,
        index=False
    )

    print("\nNormalization complete.")
    print(f"Saved output to: {output_path}")

    print("\n===== PIPELINE SUMMARY =====")
    print(f"Total appointments: {len(normalized_df)}")
    print(f"Method breakdown:")
    print(normalized_df["method"].value_counts())
    print(f"\nConfidence breakdown:")
    print(f"  High (>=0.75): {(normalized_df['confidence'] >= 0.75).sum()}")
    print(f"  Low (<0.75):   {(normalized_df['confidence'] < 0.75).sum()}")
    print(f"  Unknown:       {normalized_df['payer_code'].isna().sum()}")