# **README — Insurance Eligibility Triage**

A complete, end‑to‑end implementation of the healthcare eligibility triage workflow described in the take‑home project.
The system simulates messy real‑world appointment data, normalizes payer names using deterministic logic + LLM fallback, applies ordered business rules, and presents results in a Streamlit dashboard.

## 1. Project Overview
Every morning, an eligibility coordinator needs to know:

Which of today’s scheduled patients require an insurance eligibility re‑check, and why?

This project builds a reproducible pipeline that:

1. Simulates realistic messy healthcare data
2. Normalizes payer names using deterministic logic first, then LLM fallback
3. Applies the five business rules in strict order
4. Surfaces results in a Streamlit dashboard with three buckets: <br/>
- 🔴 Re‑check needed
- 🟢 OK
- 🟡 Unknown

The entire pipeline is deterministic (SEED=42), reproducible, and runs in seconds.

## 2. How to Run

### Install dependencies
```
pip install -r requirements.txt
```

### Activate the environment
```
 .\venv\Scripts\activate
```
### Run the full pipeline

```
python src/run_all.py
```

This will run all four steps in sequence and launches the dashboard:

- Generate synthetic data <br/>
- Normalize payer names <br/>
- Apply eligibility rules <br/>
- Write data/appointments_final.csv <br/>
- Launch the Streamlit dashboard <br/>

### To Run the dashboard manually

```
streamlit run app.py
```

Requires OPENROUTER_API_KEY in a .env file at the repo root.
Get a free key at https://openrouter.ai — the free tier is sufficient for this project.

## 3. Architecture

### 3.1 Data Simulator

The simulator generates three CSVs:

- payer_master.csv (~40 rows) <br/>
- appointments.csv (250 rows) <br/>
- last_check_history.csv (~175 rows) <br/>

The simulator intentionally includes: <br/>
- Typos <br/>
- Abbreviations <br/>
- Member‑ID‑stuffed strings <br/>
- Missing values <br/>
- Ambiguous values <br/>
- High‑turnover payers <br/>
- Payer mismatches <br/>
- Member ID mismatches <br/>
- Fresh vs stale last‑check dates <br/>



The insurance_on_file column intentionally contains the full spectrum of real front-desk messiness: clean names, abbreviations (BCBSIL), plan tiers (BCBS IL PPO), embedded member IDs (BCBSIL #MBR12345),
typos (Humanna, Blue Cros Blu Sheild IL), missing values, and genuinely ambiguous entries (self-pay, ??, BCBS / Aetna).
About 25% of patients are seeded as "clean OK" cases: their insurance_on_file is set to the exact canonical name from payer_master.csv, and their history record uses the matching
payer_code with a fresh check date (1–10 days ago). This ensures the dashboard always has a populated OK bucket for demonstration. The other 75% exercise the normalization pipeline and rule engine fully.
The history table deliberately seeds mismatches to exercise the rules:

15 patients have a different member_id than their appointment → fires Rule 3
20 patients have a different payer_code → fires Rule 2

### 3.2 Payer Normalization Pipeline

Normalization follows a deterministic‑first strategy:
1. Clean text
2. Ambiguous check
3. Alias dictionary
4. Exact canonical match
5. Fuzzy match (RapidFuzz)
6. LLM fallback (OpenRouter)
7. Pydantic validation

### LLM usage
* Only used when deterministic logic fails <br/>
* Cached to avoid repeated calls <br/>
* Hard cap of 30 calls <br/>
* Total cost: $0 (OpenRouter free tier) <br/>

Why deterministic first: The alias dictionary and exact match handle 84% of rows in the test run with zero API calls and instant results.
Fuzzy handles another 5%. The LLM sees only the genuinely hard cases typically 10–15 unique strings per run — and those results are cached to data/llm_cache.json so subsequent runs cost nothing.
Pydantic validation: Every normalizer output is a NormalizationResult model with confidence: float = Field(ge=0.0, le=1.0) and a Literal type for method. If the LLM returns a confidence value outside [0, 1]
or an unrecognised method string, Pydantic raises at the boundary rather than letting bad data propagate to the rule engine.

### 3.3 Rule Engine

Implements the five business rules in strict order:
1. No history OR last check > 30 days
2. Payer changed
3. Member ID mismatch
4. High‑turnover payer + last check > 14 days
5. Low confidence (<0.80)

If none match → OK  
If payer cannot be identified → Unknown

### Confidence threshold (Rule 5) & Operational Reasoning

One of the key judgment calls in this project is Rule 5, which triggers a re‑check when the payer normalization confidence falls below a threshold. The spec intentionally leaves this open, and Girish’s feedback clarified the operational context needed to make a defensible choice.

Why the threshold matters less than the reasoning

In this pipeline, the deterministic stages i.e.  cleaning, alias dictionary, exact match, and fuzzy match — already resolve the majority of payer names with high confidence. Only a small minority of rows fall through to the LLM fallback. Because of that, the exact numeric threshold is less important than the logic behind it. What matters is demonstrating an understanding of the workflow’s asymmetry and choosing a threshold that aligns with real operational risk.

The operational asymmetry

I chose 0.80, based on the operational asymmetry:
- False positive (extra re‑check): 5–10 minutes
- False negative (missed eligibility change): denied claim, patient friction, revenue loss

The cost of a false positive (re‑checking when it wasn’t needed) is small:
- 5–10 minutes of coordinator time
- A quick payer portal lookup or phone call

The cost of a false negative (missing an eligibility change) is much larger:
- Denied claim
- Patient frustration at the front desk
- Potential loss of visit revenue

Because false negatives are far more costly, the system leans toward more re‑checks, not fewer.

###Chosen threshold: 0.80

I selected 0.80 as the confidence threshold for Rule 5.

This value reflects the operational asymmetry:
- Anything below 80% confidence means the system is not reliably sure which payer the patient has
- If we don’t know the payer, we can’t even begin to verify eligibility
- That uncertainty alone is enough to justify a re‑check

This threshold also aligns with the fuzzy‑matching behavior: most fuzzy matches above 0.75 are strong, while those below 0.70 tend to be unreliable or ambiguous.

### 3.4 Streamlit Dashboard

The dashboard shows:
- Total appointments
- Re‑check / OK / Unknown counts
- Filters (provider, status, confidence)
- Sortable tables
- Downloadable CSV
- Diagnostics on normalization methods

The coordinator can immediately see:
- Who needs a re‑check
- Why
- How confident the system is
- How the payer was matched

## 4. Final Output Distribution

After tuning the simulator and normalization pipeline, the final distribution is:
- Re‑check needed: 133 (53%)
- OK: 56 (22%)
- Unknown: 61 (24%)

This matches real operational patterns:
- Majority require re‑check
- A healthy OK bucket
- A realistic Unknown bucket due to messy front‑desk data

## 5. Tests

Located in tests/test_rules.py.

Covers:
- Stale check
- Payer changed
- Member ID mismatch
- Unknown insurance
- OK case

All tests pass:

```
5 passed in 1.02s
```


## 6. LLM Cost Accounting

- Model: OpenRouter free tier
- Total LLM calls per run: 0–14 (cached)
- Total cost: $0.00

Deterministic logic resolves ~75% of rows.

## 7. What I’d Improve With More Time or in production

- Coordinator feedback on false positives
- Add Dockerfile for reproducible deployment
- Add historical trend dashboard
- Add a feature store for alias learning
- Add confidence calibration using real operational data
- Add unit tests for normalization
- Add CI/CD with GitHub Actions


## 8. What Changed From the Spec

- Added CLEAN:: tags to guarantee OK cases
- Added Pydantic validation to catch malformed LLM outputs
- Added LLM caching to stay under cost limits
- Added diagnostics section in the UI
- Added member‑ID‑stuffed strings to simulate real front‑desk behavior

## 9. Hours Log 
| **Phase** |	**Hours**|
| :------: | :----------: |
| Data simulator |	4 |
| Normalization pipeline |	5 |
| Rule engine	| 2 |
| Streamlit UI	| 1 |
| Debugging + tuning |	5 |
| Tests	| 1 |
| README |	2 |
| :----: | :-------: |
| **Total**	| **20 hours** |


## 10. Tools & Agents Used

I used AI coding assistants (ChatGpt + Claude Code + Copilot) for:
- Scaffolding the simulator
- Drafting normalization logic
- Streamlit layout
- Test scaffolding

I manually reviewed and rewrote:
- All business logic
- All rule ordering
- All LLM fallback behavior
- All confidence thresholds
- All simulator realism logic
- All CLEAN:: alignment logic

Agents were used as junior pair programmers, not as black boxes.

## 11. Closing Notes

This project demonstrates:
- Deterministic‑first design
- Safe, bounded LLM usage
- Realistic data simulation
- Clear rule ordering
- Operationally grounded decisions
- A maintainable, readable codebase

It reflects how I would build an internal tool for a real healthcare operations team.
