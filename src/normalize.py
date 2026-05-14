"""
normalize.py
============
Payer normalization pipeline.

Pipeline (first match wins):
    clean_text()
        → ambiguous check       → method: ambiguous,    confidence: 0.10
        → alias dict lookup     → method: alias_exact,  confidence: 0.99
        → exact canonical match → method: exact,        confidence: 1.00
        → fuzzy match (>= 75)   → method: fuzzy,        confidence: score/100
        → LLM fallback          → method: llm,          confidence: 0.78
        → unresolved            → method: llm_unknown,  confidence: 0.0–0.30

Pydantic is used to validate every result coming OUT of this module.
This means if a bug produces a confidence of 1.5, or a method string
that isn't in our allowed list, it's caught immediately at write time
rather than silently corrupting the rule engine downstream.

LLM results are cached to data/llm_cache.json so repeated runs
cost zero extra API calls for already-seen inputs.

Run:
    python src/normalize.py
"""

import json
import os
import re
from typing import Dict, List, Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz, process

# ============================================================
# Environment
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# ============================================================
# LLM usage controls
# ============================================================

ENABLE_LLM = True          # can be toggled off in tests / offline
MAX_LLM_CALLS = 30         # safety cap per run
LLM_CALL_COUNT = 0         # incremented in _llm_fallback


# ============================================================
# Configuration
# ============================================================

DATA_DIR           = "data"
LLM_CACHE_PATH     = os.path.join(DATA_DIR, "llm_cache.json")
LLM_MODEL          = "openrouter/free"
LOW_CONF_THRESHOLD = 0.75
FUZZY_THRESHOLD    = 75

# ============================================================
# Pydantic model — validates every normalization result
#
# WHY PYDANTIC HERE:
# The normalizer feeds directly into the rule engine.
# If a bad confidence value (e.g. 1.5) or a typo in method
# (e.g. "Alias_Exact") slips through, the rule engine silently
# makes wrong decisions. Pydantic catches these at the boundary
# between the two modules — right when the result is created.
# ============================================================

NormMethod = Literal[
    "ambiguous",
    "alias_exact",
    "exact",
    "fuzzy",
    "llm",
    "llm_unknown",
    "llm_invalid",
    "llm_empty_response",
    "llm_error",
    "no_match",
    "low_confidence",
]


class NormResult(BaseModel):
    """
    Validated output for a single insurance_on_file normalization.

    Fields
    ------
    normalized_payer : Human-readable canonical name, or None if unresolved.
    payer_code       : Short code used for rule comparisons, or None.
    confidence       : Float in [0, 1]. The rule engine uses 0.70 as its threshold.
    method           : How the match was made. Used in the UI pipeline summary.
    """

    normalized_payer: Optional[str] = None
    payer_code:       Optional[str] = None
    confidence:       float         = Field(ge=0.0, le=1.0)
    method:           NormMethod

    @field_validator("payer_code", "normalized_payer", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        """Treat empty strings and 'nan'/'None' strings as None."""
        if v is None:
            return None
        if str(v).strip().lower() in ("", "nan", "none"):
            return None
        return v

    def to_dict(self) -> Dict:
        return self.model_dump()


# ============================================================
# Disk-backed LLM cache
# ============================================================

def _load_cache() -> Dict:
    if os.path.exists(LLM_CACHE_PATH):
        with open(LLM_CACHE_PATH) as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LLM_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


LLM_CACHE: Dict = _load_cache()

# ============================================================
# Ambiguous values — no LLM wasted on these
# ============================================================

AMBIGUOUS_VALUES = {
    "",
    "??",
    "n/a",
    "nan",
    "none",
    "unknown",
    "self pay",
    "self-pay",
    "cash pay",
    "see notes",
    "bcbs or aetna",
    "bcbs aetna",
    "bcbs/uhc",
    "medicare or medicaid",
}

# ============================================================
# Alias dictionary
# Keys   = cleaned text (lowercase, no punctuation)
# Values = canonical_name exactly as it appears in payer_master.csv
# ============================================================

PAYER_ALIASES: Dict[str, str] = {

    # BCBS Illinois
    "bcbsil":                               "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il":                              "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il ppo":                          "Blue Cross Blue Shield of Illinois PPO",
    "bcbs il hmo":                          "Blue Cross Blue Shield of Illinois HMO",
    "blue cross illinois":                  "Blue Cross Blue Shield of Illinois PPO",
    "blue cross blue shield illinois":      "Blue Cross Blue Shield of Illinois PPO",
    "blue cross blue shield of illinois":   "Blue Cross Blue Shield of Illinois PPO",
    "blue cros blu sheild il":              "Blue Cross Blue Shield of Illinois PPO",
    "hcsc":                                 "Blue Cross Blue Shield of Illinois PPO",

    # BCBS Texas
    "bcbs texas":                           "Blue Cross Blue Shield of Texas HMO",
    "bcbs tx":                              "Blue Cross Blue Shield of Texas HMO",
    "bcbs tx hmo":                          "Blue Cross Blue Shield of Texas HMO",
    "blue cross blue shield of texas":      "Blue Cross Blue Shield of Texas HMO",

    # BCBS Florida
    "bcbs florida":                         "Florida Blue (BCBS Florida)",
    "bcbs fl":                              "Florida Blue (BCBS Florida)",
    "blue cross blue shield of florida":    "Florida Blue (BCBS Florida)",

    # UnitedHealthcare
    "uhc":                                  "UnitedHealthcare Commercial",
    "uhc choice plus":                      "UnitedHealthcare Choice Plus",
    "united healthcare":                    "UnitedHealthcare Commercial",
    "united health care":                   "UnitedHealthcare Commercial",
    "unitedhealthcare":                     "UnitedHealthcare Commercial",

    # Aetna
    "aetna":                                "Aetna Commercial PPO",
    "aetna ppo":                            "Aetna Commercial PPO",
    "atena":                                "Aetna Commercial PPO",
    "atena ppo":                            "Aetna Commercial PPO",
    "aetna hmo":                            "Aetna HMO",
    "cvs aetna":                            "CVS Health / Aetna",

    # Cigna
    "cigna":                                "Cigna Commercial PPO",
    "cingna":                               "Cigna Commercial PPO",
    "cigna hmo":                            "Cigna HMO",

    # Humana
    "humana":                               "Humana Commercial PPO",
    "humanna":                              "Humana Commercial PPO",
    "humana medicare advantage":            "Humana Medicare Advantage",

    # Medicare
    "medicare":                             "Medicare Traditional Fee-for-Service",
    "medicare part a":                      "Medicare Traditional Fee-for-Service",
    "medicare part b":                      "Medicare Part B",
    "uhc medicare advantage":               "UnitedHealthcare Medicare Advantage",
    "unitedhealthcare medicare advantage":  "UnitedHealthcare Medicare Advantage",
    "aetna medicare advantage":             "Aetna Medicare Advantage",

    # Medicaid managed care
    "medicaid illinois":                    "Medicaid Illinois Managed Care",
    "medicaid il":                          "Medicaid Illinois Managed Care",
    "medicaid texas":                       "Medicaid Texas Managed Care",
    "medicaid tx":                          "Medicaid Texas Managed Care",
    "molina":                               "Molina Healthcare",
    "molina helthcare":                     "Molina Healthcare",
    "molina healthcare":                    "Molina Healthcare",
    "wellcare":                             "WellCare Health Plans",
    "well care":                            "WellCare Health Plans",
    "ambetter":                             "Ambetter from Centene",
    "centene":                              "Centene Corporation",
    "centene ambetter":                     "Ambetter from Centene",
    "caresource":                           "CareSource",
    "meridian health plan":                 "Meridian Health Plan",
    "amerigroup":                           "Amerigroup Medicaid",

    # Other commercial
    "anthem":                               "Anthem Blue Cross",
    "anthem blue cross":                    "Anthem Blue Cross",
    "oscar":                                "Oscar Health",
    "oscar health":                         "Oscar Health",
    "kaiser":                               "Kaiser Permanente",
    "kaiser permanente":                    "Kaiser Permanente",
    "tricare":                              "Tricare",
    "magellan":                             "Magellan Health",
    "magellan health":                      "Magellan Health",
    "amerihealth":                          "AmeriHealth Commercial",
    "health net":                           "Health Net Commercial",
    "fidelis care":                         "Fidelis Care",
    "fidelis":                              "Fidelis Care",
}

# ============================================================
# Text cleaning
# ============================================================


def clean_text(text: str) -> str:
    """
    Normalise raw insurance_on_file text before matching.
    Strips member IDs, punctuation, and extra whitespace.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"#\w+", "", text)                  # #MBR12345
    text = re.sub(r"\bid[\s:\-]*\w+", "", text)       # ID: 987654
    text = re.sub(r"\bmbr[\s\-]*\w+", "", text)       # MBR-48302
    text = re.sub(r"\b\d{4,}\w*\b", "", text)         # bare numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # punctuation → space
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# Matching stages (internal)
# ============================================================


def _deterministic(
    cleaned: str,
    payer_master_df: pd.DataFrame,
) -> Optional[NormResult]:
    """Ambiguous check → alias lookup → exact canonical match."""

    if cleaned in AMBIGUOUS_VALUES:
        return NormResult(confidence=0.10, method="ambiguous")

    if cleaned in PAYER_ALIASES:
        canonical = PAYER_ALIASES[cleaned]
        row = payer_master_df[payer_master_df["canonical_name"] == canonical]
        if not row.empty:
            return NormResult(
                normalized_payer=row.iloc[0]["canonical_name"],
                payer_code=row.iloc[0]["payer_code"],
                confidence=0.99,
                method="alias_exact",
            )

    for _, row in payer_master_df.iterrows():
        if cleaned == clean_text(row["canonical_name"]):
            return NormResult(
                normalized_payer=row["canonical_name"],
                payer_code=row["payer_code"],
                confidence=1.00,
                method="exact",
            )

    return None


def _fuzzy(cleaned: str, payer_master_df: pd.DataFrame) -> NormResult:
    """RapidFuzz token_sort_ratio match against cleaned canonical names."""

    cleaned_lookup = {
        clean_text(name): name
        for name in payer_master_df["canonical_name"]
    }

    match = process.extractOne(
        cleaned,
        list(cleaned_lookup.keys()),
        scorer=fuzz.token_sort_ratio,
    )

    if not match:
        return NormResult(confidence=0.0, method="no_match")

    matched_key, score, _ = match
    confidence = round(score / 100, 2)

    if score < FUZZY_THRESHOLD:
        return NormResult(confidence=confidence, method="low_confidence")

    canonical = cleaned_lookup[matched_key]
    row = payer_master_df[
        payer_master_df["canonical_name"] == canonical
    ].iloc[0]

    return NormResult(
        normalized_payer=row["canonical_name"],
        payer_code=row["payer_code"],
        confidence=confidence,
        method="fuzzy",
    )


def _llm_fallback(
    original: str,
    payer_master_df: pd.DataFrame,
) -> NormResult:
    """
    LLM fallback for inputs that couldn't be resolved deterministically.
    Results are cached to disk so each unique string is only sent once.
    """

    global LLM_CALL_COUNT

    cleaned = clean_text(original)

    if cleaned in LLM_CACHE:
        print(f"  [CACHE HIT] '{original}'")   
        cached = LLM_CACHE[cleaned]
        return NormResult(**cached)
    
    # LLM disabled or cap reached → don't call, just mark low confidence
    if not ENABLE_LLM or LLM_CALL_COUNT >= MAX_LLM_CALLS:
        print(f"  [LLM SKIP] '{original}' (ENABLE_LLM={ENABLE_LLM}, calls={LLM_CALL_COUNT})")
        return NormResult(confidence=0.0, method="llm_error")

    print(f"  [LLM CALL] '{original}'")
    LLM_CALL_COUNT += 1

    def _cache(result: NormResult) -> NormResult:
        LLM_CACHE[cleaned] = result.to_dict()
        _save_cache(LLM_CACHE)
        return result

    try:
        payer_list = "\n".join(
            f"- {n}" for n in payer_master_df["canonical_name"]
        )

        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_tokens=30,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You normalize messy insurance payer names.\n"
                        "Rules:\n"
                        "- Choose ONLY from the provided payer list.\n"
                        "- If unclear, return UNKNOWN.\n"
                        "- Return ONLY the exact payer name. Nothing else."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Messy payer:\n{original}\n\n"
                        f"Valid payers:\n{payer_list}"
                    ),
                },
            ],
        )

        if not response.choices:
            return _cache(NormResult(confidence=0.25, method="llm_empty_response"))

        message = response.choices[0].message
        content = (
            str(message.content).strip()
            if (message and message.content)
            else ""
        )

        if not content:
            return _cache(NormResult(confidence=0.25, method="llm_empty_response"))

        if content.upper() == "UNKNOWN":
            return _cache(NormResult(confidence=0.30, method="llm_unknown"))

        rows = payer_master_df[
            payer_master_df["canonical_name"].str.lower() == content.lower()
        ]

        if rows.empty:
            return _cache(NormResult(confidence=0.20, method="llm_invalid"))

        row = rows.iloc[0]
        return _cache(
            NormResult(
                normalized_payer=row["canonical_name"],
                payer_code=row["payer_code"],
                confidence=0.78,
                method="llm",
            )
        )

    except Exception as e:
        print(f"LLM fallback error: {e}")
        return _cache(NormResult(confidence=0.0, method="llm_error"))


# ============================================================
# Public API
# ============================================================


def normalize_payer(
    insurance_value: str,
    payer_master_df: pd.DataFrame,
) -> NormResult:
    """
    Normalize a single raw insurance_on_file string.
    Returns a validated NormResult.
    """
    cleaned = clean_text(insurance_value)

    result = _deterministic(cleaned, payer_master_df)
    if result:
        return result

    result = _fuzzy(cleaned, payer_master_df)
    if result.confidence >= LOW_CONF_THRESHOLD:
        return result

    return _llm_fallback(insurance_value, payer_master_df)


def normalize_appointments(
    appointments_df: pd.DataFrame,
    payer_master_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run normalize_payer() over every row.
    Returns appointments_df with 4 new columns:
        normalized_payer, payer_code, confidence, method
    """
    results: List[Dict] = [
        normalize_payer(v, payer_master_df).to_dict()
        for v in appointments_df["insurance_on_file"]
    ]

    print("\n[PYDANTIC TEST] Trying to create NormResult with confidence=999...")
    try:
        NormResult(confidence=999, method="exact")   # ← ADD THIS BLOCK
        print("  ERROR: Pydantic did NOT catch it!")
    except Exception as e:
        print(f"  PASS: Pydantic caught it → {e}")

    return pd.concat(
        [appointments_df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    APPOINTMENTS_PATH = os.path.join(DATA_DIR, "appointments.csv")
    PAYER_MASTER_PATH = os.path.join(DATA_DIR, "payer_master.csv")
    OUTPUT_PATH       = os.path.join(DATA_DIR, "appointments_normalized.csv")

    print("\nLoading datasets...")
    appointments_df = pd.read_csv(APPOINTMENTS_PATH)
    payer_master_df = pd.read_csv(PAYER_MASTER_PATH)
    print(f"  appointments : {len(appointments_df)} rows")
    print(f"  payer master : {len(payer_master_df)} rows")
    print(f"  LLM cache    : {len(LLM_CACHE)} entries loaded from disk")

    print("\nRunning normalization pipeline...")
    normalized_df = normalize_appointments(appointments_df, payer_master_df)
    normalized_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")

    print("\n===== PIPELINE SUMMARY =====")
    print(f"Total rows      : {len(normalized_df)}")
    print("\nMethod breakdown:")
    print(normalized_df["method"].value_counts().to_string())
    print(f"\nHigh confidence (>=0.75) : {(normalized_df['confidence'] >= 0.75).sum()}")
    print(f"Low confidence  (<0.75)  : {(normalized_df['confidence'] < 0.75).sum()}")
    print(f"Unresolved               : {normalized_df['payer_code'].isna().sum()}")
    print(f"\nLLM calls this run : {LLM_CALL_COUNT}")
    print(f"LLM cache size     : {len(LLM_CACHE)}")
