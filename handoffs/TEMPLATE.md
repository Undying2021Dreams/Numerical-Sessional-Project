# RUN REPORT — Milestone M<n>: <title>

> Fill in every section. Write for a reader who cannot see the repository.
> Report numbers, not adjectives. "Tests pass" is not a result.
> If something is unverified, guessed, or shaky, say so plainly — that is the single
> most useful thing this document can contain.

---

## 1. One-paragraph summary

What was built in this milestone, in plain prose, in 4-6 sentences.

## 2. Files added or changed

| Path | Lines | Purpose |
|---|---|---|

## 3. What each component does, mathematically

For each significant module, state which equation, lemma, or algorithm from the paper it
implements, and how. Name the paper object explicitly, e.g. "eq. (3) of Lemma 6",
"eq. (26), IRGNM step", "Remark 18, biexponential f". If a component corresponds to
nothing in the paper, say that and explain why it exists.

## 4. Acceptance criteria — results

Reproduce each acceptance criterion from PLAN.md for this milestone, verbatim, followed
by the measured outcome as a NUMBER.

| Criterion (from PLAN.md) | Target | Measured | Pass? |
|---|---|---|---|

## 5. Key numerical results

Tables and numbers a reviewer can sanity-check by hand or against the paper. Include:
- errors, residuals, condition numbers, convergence orders, iteration counts, runtimes
- for any Monte Carlo study: n realisations, mean, std, and number of divergent runs

## 6. Figures produced

| File | What it shows | What the reader should look for |
|---|---|---|

## 7. Assumptions made this milestone

Everything not dictated by the paper or by PLAN.md. One line of justification each.
Cross-reference DECISIONS.md entries.

| Assumption | Why | Risk if wrong |
|---|---|---|

## 8. Failures, divergences, and things that did not work

Be complete. Include experiments abandoned, tolerances that were hard to hit, settings
where the solver diverged, and anything that only works with a suspiciously specific
parameter value.

## 9. Track A / Track B audit

- Confirm `tests/test_no_library_solvers.py` passes, and list exactly which library
  functions are used and where (they must all be under `tests/` or a marked benchmark).
- List every Track A routine written so far and the Track B reference it was checked
  against.

## 10. Mutation check

Pick one core routine. Deliberately introduce one wrong sign or dropped term. Report
which tests went red. If none did, say so — that means the tests are decorative and need
strengthening before proceeding.

| Mutation applied | Tests that failed | Verdict |
|---|---|---|

## 11. Reproduction commands

Exact commands that regenerate everything in this report, with seeds.

```
```

## 12. Uncertainties and open questions for the reviewer

The most important section. What are you unsure about? Where might the implementation
diverge from the paper's intent? What judgement calls would you like a second opinion on?
Number them so they can be answered individually.

1.
2.
3.

## 13. Proposed next step

What M<n+1> should start with, and anything from this milestone that should be revisited
first.