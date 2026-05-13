# Take-Home Project — Insurance Eligibility Triage

**Time-box:** 2-3 days of focused work, spread across up to one week
**Stack:** Your call within the constraints below

---

## Why we're doing this

I want to see how you think and how you ship, on a problem shaped like the work you'd do here. Take-home over whiteboard because the work is more real, you can use the tools you'd actually use, and I get a better signal than I would from any 90-minute live coding session. The goal is not a polished product — it's evidence of judgment, taste, and real Python/LLM fluency.

You will simulate the dataset yourself. That's deliberate. Building a realistic enough fixture is part of the test.

---

## The problem

You're building an internal tool for a healthcare operations team. Every morning, an eligibility coordinator wants to know: *which of today's scheduled patient appointments need an insurance eligibility re-check before the patient arrives, and why?*

The inputs to your tool:

- **Today's appointments** — a CSV/Excel export with patient name, date of birth, appointment time, provider, and a free-text "insurance on file" field captured by the front desk.
- **Contracted-payer master list** — the canonical list of insurance plans your organization is in-network with.
- **Last-check history** — when each patient last had eligibility verified, with which payer, and what the result was.

The output: a Streamlit (or comparable) dashboard the coordinator opens in the morning, showing today's appointments grouped into three buckets — **Re-check needed**, **OK**, **Unknown** — sortable by appointment time, with the reasoning for each row visible.

---

## What you'll build

1. A **data simulator** that produces the three input datasets described in the next section. Commit the simulator. The data should be regeneratable from a seed.
2. A **payer normalization step** that maps the messy free-text "insurance on file" field to a canonical payer in the master list. Some entries will match cleanly. Some won't. Use whatever combination of deterministic logic and LLM calls you think is right. I'm interested in the choice.
3. A **rule engine** that applies the business rules in the section below to each appointment and produces a status (Re-check / OK / Unknown) plus a reason.
4. A **Streamlit UI** that loads the data, runs the pipeline, shows results in the three buckets, lets the user filter by provider and by status, and lets them download a cleaned export.
5. A **README** written for me, not for another engineer. What you built, how to run it, what you'd do differently with another week, and what's in the code that you're least proud of.

---

## Data you'll simulate

You decide field types and values; what matters is that the data has the right shape of messiness. The realism of your simulator is itself part of what I'm evaluating.

**Appointments (~250 rows):**
- Patient ID, name, DOB, appointment datetime, provider name.
- A free-text **`insurance_on_file`** column. This is the important one. It should contain a realistic mix: clean payer names, abbreviations ("BCBSIL"), full long names ("Blue Cross Blue Shield of Illinois"), names with plan tier ("BCBS IL PPO"), name-with-member-ID stuffed in ("BCBSIL #XYZ123"), typos, missing values, and 5-10% genuinely ambiguous entries (multiple payers mentioned, "self-pay", "??"). Mix at least 25 different real-world payer name variations.
- A `member_id` column that is sometimes present, sometimes empty, occasionally different from the member ID on the last-check history (test the "member ID changed" rule).

**Payer master (~40 rows):**
- Canonical payer name, payer code, plan type (Commercial / Medicare / Medicaid / Medicare Advantage), and a `high_turnover` boolean flag. Mark Medicaid managed care plans and a couple of BCBS plans as high-turnover.

**Last-check history (~150 rows, covering ~70% of patients):**
- Patient ID, payer code at last check, member ID at last check, last-check date (spread across the last 60 days), result (Active / Inactive / Unknown).

---

## The business rules

A re-check is needed if **any** of the following are true. Apply them in order; first match wins, and the reason should be specific.

1. There is no last-check record for this patient, OR the last check is more than **30 days old**.
2. The normalized payer for today's appointment is different from the payer at last check.
3. The member ID on file is present and different from the member ID at last check.
4. The payer is flagged `high_turnover` AND the last check is more than **14 days old**.
5. The payer normalization has confidence below some threshold you choose. Decide what that threshold is and defend it in the README.

If none match: status is **OK**.
If the patient's `insurance_on_file` couldn't be normalized at all: status is **Unknown** and the reason should make that clear.

---

## Technical constraints

- **Python 3.10+.** Whatever web framework you want — Streamlit is the obvious choice for the UI but I'd accept FastAPI + a small front-end if that's where your strength is.
- **LLM:** OpenAI, Anthropic, or Azure OpenAI is fine. Use whatever you have access to. Stay under **$5** in total LLM spend. Show your token / cost accounting in the README.
- **No PHI.** All data is synthetic. Don't pull real patient or payer data from anywhere.
- **Repo:** GitHub or a zip. Either works. Include a `requirements.txt` or `pyproject.toml` and a one-command run path.
- **Tests:** at least one test, even if small. I want to see what you choose to test.

---

## Deliverables

1. A working repo (or zip) that I can clone, install, and run.
2. The data simulator script and the generated CSVs/Excel files committed alongside.
3. The Streamlit (or equivalent) app.
4. A README that covers: what you built, how to run it, the architecture choices you made and why, what you'd build differently with more time, your token/cost accounting, and an honest hours log.
5. A short separate note (a paragraph in the README is fine) on what changed in the spec as you implemented it. Real specs always change. I want to see what you'd push back on.

---

## Using a coding agent is encouraged

Use Claude Code, Cursor, Copilot, Codex, or whatever coding agent you're fluent with. We use them aggressively here — pretending you didn't would be the wrong signal.

A few things I'd want you to do regardless:

- Note in the README which agent(s) you used and roughly for what — scaffolding, the LLM call wiring, the rule engine, the Streamlit UI, tests, the data simulator. I'm not policing it, I just want to see how you partition human-driven versus agent-driven work.
- Read every line you commit. If I ask you in the walkthrough why a particular function exists or why a Pydantic field is `Optional`, "the agent wrote that" isn't an answer that goes anywhere good. Treat the agent like a junior pair, not a black box.
- The judgment calls — what to deterministic-vs-LLM, what the confidence threshold should be, what your data fixture's messiness profile is, what to test — are the points where I'm reading *you*, not the agent. Spend your thinking time there.

Working fast with agents is a strength. Shipping code you don't understand is the failure mode I'm checking for.

---

## What I'll be looking at

I'm partly transparent about this on purpose, because I'd rather you optimize for the things that matter than for what looks impressive.

- **Where you used the LLM and where you didn't.** The strongest candidates use deterministic logic first and fall back to the LLM only where ambiguity actually requires reasoning. Calling the LLM 250 times in a loop is the wrong answer.
- **Structured outputs.** If your LLM call returns a string you parse with regex, that's a flag. Use Pydantic or a JSON schema and validate.
- **What happens when the LLM returns garbage.** Retries, fallbacks, default values — show me you've thought about it.
- **The data simulator.** Does it produce the right *shape* of messiness, or is it a clean fixture pretending to be messy?
- **Idiomatic Python.** Pandas, type hints, list comprehensions, Pydantic models. Not Java translated to Python.
- **The README.** Does it tell me a story I can follow without you in the room? If I had to maintain this code in six months without you, would the README help?

---

## Time-box, honesty, and pace

Two to three days of actual focused work. You decide how to spread that across the week. Log your hours per phase (data sim, normalization, rule engine, UI, README) and put the log in the README. If you spent eight hours on the LLM call and 30 minutes on the UI, tell me — I'd rather see honesty than a fictional even split.

If you finish in less than the time-box, that's fine. Don't pad. Use the leftover time to write the README well.

If you hit a problem you can't solve, document the problem in the README and move on. Acknowledging what's broken is more useful to me than hiding it.

---

## Logistics

Send me the repo link or the zip when you're done. I'll review it within 2 business days and we'll set up the half-day onsite to walk through it together.

If you have questions while you're working, email me. I'd rather answer one specific question than have you guess at intent.

Girish Srinivasan
