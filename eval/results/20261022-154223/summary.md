# Evaluation 20261022-154223

Model `gemini-3.5-flash-lite` · layer `4fe9ae6e6482` · today = 2026-10-22 · 1 runs per question · 2026-10-22T15:42:23 to 2026-10-22T15:44:23

| Metric | Value |
|---|---|
| Correct runs (all questions) | 62% of 8 |
| Correct runs (answerable questions) | 50% |
| Questions correct in every run | 62% |
| Questions with the same result in every run | 100% |
| Queries that ran (answerable questions) | 100% |
| Right rows but a column missing | 1 runs |
| Unanswerable questions refused | 100% |
| Answerable questions wrongly refused | 0% |
| Runs that gave a wrong answer | 25% |
| Runs within 60s | 100% |
| Seconds per run (mean / median / p95 / max) | 13.65 / 14.74 / 19.95 / 19.95 |
| Model calls | 25 |

| # | Level | Question | Verdicts per run | Correct | Consistent | Median s |
|---|---|---|---|---|---|---|
| q1 | Easy | List the active shops in Istanbul with their shop number, district and sales manager. | partial | 0/1 | yes | 9.65 |
| q2 | Easy | How many phone campaigns are available for sale, by device? | match_extra_columns | 1/1 | yes | 19.08 |
| q3 | Medium | Which customer segment had the highest number of unique customers assigned in June 2025? | match | 1/1 | yes | 11.81 |
| q4 | Medium | Compare monthly gross adds and disconnections for July and August 2026, showing the month-over-month change in each. | match | 1/1 | yes | 18.08 |
| q5 | Hard | As of 28 August 2026, how many subscribers in the mobile base used a physical SIM and how many used an eSIM? | mismatch | 0/1 | yes | 19.95 |
| q6 | Hard | For every disconnection in August 2026, which tariff was the subscription on at the moment of the event? Count disconnections per tariff. | mismatch | 0/1 | yes | 17.67 |
| r1 | Refusal | What was the average hotel room price in Paris last year? | correct_refusal | 1/1 | yes | 3.79 |
| r2 | Refusal | How much revenue did flight ticket sales bring in during August 2026? | correct_refusal | 1/1 | yes | 9.14 |

## Notes

- **q6**: The reference query joins the tariff list on its id without picking one version, and 5 tariffs have more than one version in that list, so 130 disconnections are counted twice (937 in total for 807 events). Rosetta counts each disconnection once, so this question is expected to be reported as a mismatch.

Every run (query, rows, time, verdict and reason) is in `runs.jsonl`; expected rows in `expected.json`.
