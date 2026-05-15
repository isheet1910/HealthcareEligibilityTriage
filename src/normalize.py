"""
normalize.py
============
Payer normalization pipeline.

Pipeline (first match wins):
    clean_text()
        → ambiguous check       → method: ambiguous,    confidence: 0.10  basically hard code for ?? na none unkwonw etc
        → alias dict lookup     → method: alias_exact,  confidence: 0.99
        → exact canonical match → method: exact,        confidence: 1.00
        → fuzzy match (>= 75)   → method: fuzzy,        confidence: score/100
        → LLM fallback          → method: llm,          confidence: 0.8
        → unresolved            → method: llm_unknown,  confidence: 0.0-0.30

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
import traceback

import pandas as pd
from dotenv import load_dotenv #store API keys in env file
from openai import OpenAI #used openai as its easier and compatible with openrouter as well
from pydantic import BaseModel, Field, field_validator
from rapidfuzz import fuzz, process

# ============================================================
# Environment
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") #load key in .env file
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY in .env file")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1", #changed from openai to openrouter for cost
)

# ============================================================
# LLM usage controls
# ============================================================

ENABLE_LLM = True          # can be toggled off in tests / offline
MAX_LLM_CALLS = 30         # safety cap per run I think limit for day is 50
LLM_CALL_COUNT = 0         # incremented in _llm_fallback to keep track 


# ============================================================
# Configuration
# ============================================================

DATA_DIR           = "data"
LLM_CACHE_PATH     = os.path.join(DATA_DIR, "llm_cache.json")
LLM_MODEL          = "openrouter/free"
LOW_CONF_THRESHOLD = 0.80 #important factor in deciding when to call llm
FUZZY_THRESHOLD    = 80 #threshold for fuzzy matching

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
] #method must be one of these values, otherwise pydantic will raise an error


class NormResult(BaseModel):
    """
    Validated output for a single insurance_on_file normalization.

    Fields
    ------
    normalized_payer : Human-readable canonical name, or None if unresolved.
    payer_code       : Short code used for rule comparisons, or None.
    confidence       : Float in [0, 1]. The rule engine uses 0.80 as its threshold.
    method           : How the match was made. Used in the UI pipeline summary.
    """

    normalized_payer: Optional[str] = None
    payer_code:       Optional[str] = None
    confidence:       float         = Field(ge=0.0, le=1.0) #confidence must be between 0 and 1, otherwise pydantic will raise an error
    method:           NormMethod #method must be one of the defined values, otherwise pydantic will raise an error

    @field_validator("payer_code", "normalized_payer", mode="before") #treat empty strings as None
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
#if cache file exists, load it into a dictionary. If not, return an empty dictionary


def _save_cache(cache: Dict) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LLM_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
# Save the given cache dictionary to disk as JSON. Creates data directory if it doesn't exist.

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
    "medicare or medicaid"
} #hardcode for these ambiguors data

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
    text = re.sub(r"#\w+", "", text)                  # #MBR12345 #to remove unnecesaary member id whicih is not useful 
    text = re.sub(r"\bid[\s:\-]*\w+", "", text)       # ID: 987654 #remove id: or id- from field
    text = re.sub(r"\bmbr[\s\-]*\w+", "", text)       # MBR-48302 remove mbr- or mbr followed by id
    text = re.sub(r"\b\d{4,}\w*\b", "", text)         # bare numbers which have been added like 4 digitds fo not mbr id 
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # punctuation → space remove ./-etc and replace with space
    text = re.sub(r"\s+", " ", text).strip()          #if there are multiple spaces then remove them 

    return text


# ============================================================
# Matching stages (internal)
# ============================================================


def _deterministic( #for the first 3 stages: ambiguous check, alias dict lookup, exact canonical match
    cleaned: str,
    payer_master_df: pd.DataFrame,
) -> Optional[NormResult]:
    """Ambiguous check → alias lookup → exact canonical match."""

    if cleaned in AMBIGUOUS_VALUES: #if the cleaned text is in the hardcoded ambiguous values set, return a low confidence result with method "ambiguous"
        return NormResult(confidence=0.10, method="ambiguous")

    if cleaned in PAYER_ALIASES: #if the cleaned text is in the alias dictionary, return the corresponding canonical name and payer code with high confidence and method "alias_exact"
        canonical = PAYER_ALIASES[cleaned]
        row = payer_master_df[payer_master_df["canonical_name"] == canonical]
        if not row.empty:
            return NormResult(
                normalized_payer=row.iloc[0]["canonical_name"],
                payer_code=row.iloc[0]["payer_code"],
                confidence=0.99,
                method="alias_exact",
            ) #basically we got an exact alias match

    for _, row in payer_master_df.iterrows():
        if cleaned == clean_text(row["canonical_name"]):
            return NormResult(
                normalized_payer=row["canonical_name"],
                payer_code=row["payer_code"],
                confidence=1.00,
                method="exact",
            ) #look through all 40 paymaster rows one by one and if you find the match then return it with confidence 1 and method exact

    return None


def _fuzzy(cleaned: str, payer_master_df: pd.DataFrame) -> NormResult:
    """RapidFuzz token_sort_ratio match against cleaned canonical names."""
    #basically we use rapidfuzz library which uses fuzzy macthmaking to score from 0 to 100 on how close the strings are to what we want 
    # token_sort_ratio is a method which uses alphabetic logic which can be similar to our use case and hecne I used it like alphabets will be similar or closer when people type so

    cleaned_lookup = {
        clean_text(name): name
        for name in payer_master_df["canonical_name"]
    } #make a mapping dictionary where keys are cleaned canonical names and values are original canonical names from the payer master dataframe

    match = process.extractOne(
        cleaned,
        list(cleaned_lookup.keys()),
        scorer=fuzz.token_sort_ratio,
    ) #esxtract one basically compares to all cleaned canonical and returns the one which is the closet to the cleaned input along with the score and the index of the match in the list
 
    if not match:
        return NormResult(confidence=0.0, method="no_match")
    #basically if there is no match at all then return confidence 0 and method no_match and we can send it to LLM fallback

    matched_key, score, _ = match
    confidence = round(score / 100, 2) #output of fuzzy match is rfom 0 to 100 and we need confidenec between 0 adn 1 so

    if score < FUZZY_THRESHOLD:
        return NormResult(confidence=confidence, method="low_confidence")

    # Take the fuzzy-matched payer name, convert it to the official canonical name, then fetch that payer’s row from the master table.
    canonical = cleaned_lookup[matched_key] #look up the original canonical name using the matched cleaned key
    row = payer_master_df[
        payer_master_df["canonical_name"] == canonical
    ].iloc[0] # find the row in the payer master dataframe where the canonical name matches the one we found and get payer details

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

    # Check cache first — if we've already asked the LLM about this input, reuse that result
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

    # Helper function to save LLM result to cache and return it in one step
    def _cache(result: NormResult) -> NormResult:
        LLM_CACHE[cleaned] = result.to_dict()
        _save_cache(LLM_CACHE)
        return result

    try:
        payer_list = "\n".join(
            f"- {n}" for n in payer_master_df["canonical_name"]
        ) #list of all valid payer namews to provide to the LLM in the prompt basically take the entire list into a string

        #make the api call to the LLM with a system prompt that instructs it to normalize messy insurance payer names according to specific rules, and a user prompt that provides the messy payer and the list of valid payers. The LLM is expected to return only the exact payer name if it can find a match, or "UNKNOWN" if it's unclear, or an empty response if it can't find anything. The response is then processed to determine the final normalization result.
        response = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0, #deterministic output no risk with randomness as can cause future issues no creativity
            max_tokens=30, #only want final names short results no long paragraphs
            messages=[
                {
                    "role": "system", #system tells AI how to behave what to do
                    "content": (
                        "You normalize messy insurance payer names.\n"
                        "Rules:\n"
                        "- Choose ONLY from the provided payer list.\n"
                        "- If unclear, return UNKNOWN.\n"
                        "- Return ONLY the exact payer name. Nothing else."
                    ),
                },
                {
                    "role": "user", #give the actual user input and the list of valid payers to choose from
                    "content": (
                        f"Messy payer:\n{original}\n\n"
                        f"Valid payers:\n{payer_list}"
                    ),
                },
            ],
        )

        if not response.choices: #if api fails or not response then what to do, return low confidence with method llm_empty_response
            return _cache(NormResult(confidence=0.25, method="llm_empty_response")) #always return after caching the ouput

        message = response.choices[0].message #extract the message
        content = (
            str(message.content).strip() #remove extra spaces lines etc from the response
            if (message and message.content) #if both are true access the content otherwise return empty string to avoid errors
            else ""
        )
        print(f"    [LLM RESPONSE] {content}")

        if not content: #check if return is blank test
            return _cache(NormResult(confidence=0.25, method="llm_empty_response"))

        if content.upper() == "UNKNOWN": #if LLM says unknown then return low confidence with method llm_unknown
            return _cache(NormResult(confidence=0.30, method="llm_unknown"))

        rows = payer_master_df[
            payer_master_df["canonical_name"].str.lower() == content.lower()
        ] #check for hallucinations as the answer given by the llm should be in the payer master dataframe if not then it is a hallucination and we should return low confidence with method llm_invalid

        if rows.empty: #if the row is empty then it means the LLM response is not valid and we should return low confidence with method llm_invalid
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
        print("\n========== LLM ERROR ==========")
        traceback.print_exc()
        print("================================\n")
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
        # ------------------------------------------------------------
    # CLEAN CASE BYPASS (from simulator)
    # Format: CLEAN::<payer_code>
    # #implemented a clean bypass which consistem of clean values but later changed but code works as it checks if the insurance value starts with CLEAN:: and if it does then it extracts the payer code from the string and looks it up in the payer master dataframe to get the canonical name and returns a NormResult with confidence 1 and method exact 
    # without doing any of the other steps, this is useful for testing and for bypassing the cleaning steps when we already have clean data 
    # ------------------------------------------------------------
    if isinstance(insurance_value, str) and insurance_value.startswith("CLEAN::"): #skip all the rows which have been clean by default and marked byt simulator
        payer_code = insurance_value.replace("CLEAN::", "").strip()
        row = payer_master_df[payer_master_df["payer_code"] == payer_code]

        if not row.empty:
            canonical = row.iloc[0]["canonical_name"]
            return NormResult(
                normalized_payer=canonical,
                payer_code=payer_code,
                confidence=1.0,
                method="exact",
            )
        else:
            # Should never happen, but safe fallback
            return NormResult(
                normalized_payer=None,
                payer_code=None,
                confidence=0.0,
                method="llm_invalid",
            )
        
    #Clean the text, try deterministic matching. If it worked, return immediately.
    cleaned = clean_text(insurance_value)

    result = _deterministic(cleaned, payer_master_df)
    if result:
        return result

    # If deterministic matching failed, try fuzzy matching. If confidence is high enough, return that.
    result = _fuzzy(cleaned, payer_master_df)
    if result.confidence >= LOW_CONF_THRESHOLD:
        return result

    return _llm_fallback(insurance_value, payer_master_df) #if both fail callLLM fallback


def normalize_appointments(
    appointments_df: pd.DataFrame,
    payer_master_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run normalize_payer() over every row.
    Returns appointments_df with 4 new columns:
        normalized_payer, payer_code, confidence, method
    """
    #for every line in insurance_on_file call the normalize_payer function and store the results in a list of dictionaries
    results: List[Dict] = [
        normalize_payer(v, payer_master_df).to_dict()
        for v in appointments_df["insurance_on_file"]
    ]
    
    #below is to check if pydantic test is working for code if we try to create a NormResult with confidence 999 which is invalid then it should raise an error and we can catch it to confirm that pydantic is working as expected and it should not allow us to create a NormResult with confidence greater than 1 or method that is not in the defined list
    #test code for pydantic nothing more
    print("\n[PYDANTIC TEST] Trying to create NormResult with confidence=999...") #should not work as our confidence must be within 0 and 1 
    try:
        NormResult(confidence=999, method="exact")   # ← ADD THIS BLOCK
        print("  ERROR: Pydantic did NOT catch it!")
    except Exception as e:
        print(f"  PASS: Pydantic caught it → {e}")

    return pd.concat(
        [appointments_df.reset_index(drop=True), pd.DataFrame(results)],
        axis=1,
    ) #join the twi data frames side by side column wise and add the 4 normalized payer code and cofidence and  method to teh dataframe file


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
