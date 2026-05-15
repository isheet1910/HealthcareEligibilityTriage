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

###Install dependencies

pip install -r requirements.txt

###Run the full pipeline

python src/run_all.py

This will:

- Generate synthetic data <br/>
- Normalize payer names <br/>
- Apply eligibility rules <br/>
- Write data/appointments_final.csv <br/>
- Launch the Streamlit dashboard <br/>

### To Run the dashboard manually

streamlit run app.py

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

#### CLEAN:: tags (key design choice)

To guarantee a realistic OK bucket, ~25% of appointments are marked:

CLEAN::<payer_code>

These bypass normalization entirely and ensure: <br/>
- payer_code matches history <br/>
- member_id matches <br/>
- last_check_date < 10 days <br/>

This produces a stable OK bucket (~20%).

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

### CLEAN:: bypass
If the simulator emits CLEAN::AETNA_COMM, normalization returns: <br/>
- payer_code = AETNA_COMM <br/>
- confidence = 1.0 <br/>
- method = exact <br/>
 
No fuzzy, no alias, no LLM.

### 3.3 Rule Engine

Implements the five business rules in strict order:
1. No history OR last check > 30 days
2. Payer changed
3. Member ID mismatch
4. High‑turnover payer + last check > 14 days
5. Low confidence (<0.70)

If none match → OK  
If payer cannot be identified → Unknown

### Confidence threshold (Rule 5)

I chose 0.70, based on the operational asymmetry:
- False positive (extra re‑check): 5–10 minutes
- False negative (missed eligibility change): denied claim, patient friction, revenue loss

Because false negatives are far more costly, the system leans toward more re‑checks, not fewer.

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


5 passed in 1.02s


## 6. LLM Cost Accounting

- Model: OpenRouter free tier
- Total LLM calls per run: 0–14 (cached)
- Total cost: $0.00

Deterministic logic resolves ~75% of rows.

## 7. What I’d Improve With More Time

- Add a feature store for alias learning
- Add confidence calibration using real operational data
- Add unit tests for normalization
- Add CI/CD with GitHub Actions
- Add Dockerfile for reproducible deployment
- Add historical trend dashboard

## 8. What Changed From the Spec

- Added CLEAN:: tags to guarantee OK cases
- Added Pydantic validation to catch malformed LLM outputs
- Added LLM caching to stay under cost limits
- Added diagnostics section in the UI
- Added member‑ID‑stuffed strings to simulate real front‑desk behavior

## 9. Hours Log (Honest)
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
