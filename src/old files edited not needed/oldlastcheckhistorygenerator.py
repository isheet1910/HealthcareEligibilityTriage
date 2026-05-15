# ============================================================
# 3. Last-Check History (~175 rows, ~70% of patients)
# ============================================================


# def generate_last_check_history(patients_df, appointments_df, payer_master, clean_patients):
#     """
#     Build eligibility check history covering ~70% of patients.

#     Deliberately seeded mismatches to exercise triage rules:
#       - 15 patients have a DIFFERENT member_id than appointments.csv
#         → triggers Rule 3 "member ID changed"
#       - 20 patients have a DIFFERENT payer_code than what is on file
#         → triggers Rule 2 "payer changed"

#     last_check_date is spread across the last 40 days (per spec),
#     skewed so a realistic mix falls inside and outside the 30-day window.
#     """

#     today      = datetime.today().date()
#     payer_codes = payer_master["payer_code"].tolist()

#     all_patients   = patients_df["patient_id"].tolist()
#     n_with_history = int(len(all_patients) * 0.70)  

#     patients_with_history = random.sample(all_patients, n_with_history)

#     # Pre-select mismatch patients (non-overlapping sets)
#     changed_member_patients = set(random.sample(patients_with_history, 5))
#     remaining               = [p for p in patients_with_history if p not in changed_member_patients]
#     changed_payer_patients  = set(random.sample(remaining, 10))

#     appt_member_id = dict(zip(patients_df["patient_id"], patients_df["member_id"]))

#     rows = []

#     for patient_id in patients_with_history:

#         # if patient_id in clean_patients:
#         #     payer_code = payer_master.sample(1)["payer_code"].iloc[0]
#         #     member_id = appt_member_id[patient_id]
#         #     days_ago = random.randint(1, 10)
#         #     last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

#         if patient_id in clean_patients:
#     # Extract the CLEAN payer_code from appointments_df
#             clean_value = appointments_df.loc[appointments_df["patient_id"] == patient_id, "insurance_on_file"].iloc[0]
#             payer_code = clean_value.replace("CLEAN::", "").strip()
#             member_id = appt_member_id[patient_id]

#     # Fresh check → OK
#             days_ago = random.randint(1, 10)
#             last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")


#         else:
#             days_ago = random.randint(0, 25)
#             last_check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

#             if random.random() < 0.70:
#                 payer_code = payer_master.sample(1)["payer_code"].iloc[0]
#             else:
#                 payer_code = random.choice(payer_codes)

#             original = appt_member_id.get(patient_id, "")
#             if patient_id in changed_member_patients and original:
#                 new_id = _random_member_id()
#                 while new_id == original:
#                     new_id = _random_member_id()
#                 member_id = new_id
#             else:
#                 member_id = original

#         result = random.choices(
#             ["Active", "Inactive", "Unknown"],
#             weights=[0.70, 0.20, 0.10],
#             k=1,
#         )[0]

#         rows.append({
#             "patient_id":     patient_id,
#             "payer_code":     payer_code,
#             "member_id":      member_id,
#             "last_check_date": last_check_date,
#             "result":         result,
#         })

#     return pd.DataFrame(rows)