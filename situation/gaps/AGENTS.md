# Gaps

A Gap records an absence, concern, or uncertainty encountered during repository
work, including minor or tentative concerns. It preserves what prompted the
concern and what remains unresolved. Recording it establishes that the concern
was raised, not that a suspected defect exists or that a remedy is necessary.

## File naming

```
G-<six digits>-<kebab-case-name>.md
```

## Required headings

- `State` — lifecycle state
- `Gap` — the absence, concern, or uncertainty at its actual level of certainty
- `Relevance` — the repository behavior or work in which it arose
- `Evidence` — what prompted it, distinguishing observation from interpretation
- `Impact` — known or possible consequences; an unresolved effect is valid
- `Resolution` — current resolution, or explicitly none
- `References` — optional retained depth

## States

- `open` — recorded and unresolved
- `addressing` — linked Candidate, Promise, or Plan is actively resolving it
- `closed` — cited evidence or a Decision settles the absence or uncertainty
- `accepted` — a Decision explicitly tolerates it
- `superseded` — replaced by a more accurate Gap

## Relationship rules

- `addressing` links the Candidate/Promise/Plan acting on it.
- `closed` links the Promise/Oracle/Witness, commit, or Decision settling it.
- `accepted` links the Decision accepting it.
- `superseded` links the replacement Gap.
- A Promise Residual may link a Gap when excluded behavior remains relevant.

## Incidental reporting

Record concerns when noticed, particularly when uncertainty begins to pull work
beyond its assigned Promise or Oracle. Check relevant existing Gaps through
ordinary discovery: cite an already captured concern, add observations to the
same unresolved question, or create a distinct Gap with links to related ones.
Shared terminology alone does not establish that two concerns are the same.
Create a new Gap in state `open` with Resolution explicitly `none`.

Link available Promises, Oracles, Witnesses, and source references. State
observed sequences separately from suspected interactions or causal chains.
The basis may be tentative or pedantic; reporting needs no additional
experiment to establish the concern's importance.

Reporting preserves knowledge; qualification, Decisions, Candidates, and
resolution belong to separately assigned work. Continue the current
Promise/Oracle/Witness assignment. An open Gap alone neither invalidates a
passing Oracle nor changes a failing, invalid, or blocked Witness into PASS.

## Closure repair

A Gap is an observation, and recording one assigns nothing. When the change
under review affects behavior the repository claims and that behavior has no
local Promise, Oracle, or Witness lineage, establishing the missing records
is repair work inside the assigned closure: write the promise and oracle
honestly — retrospectively when the behavior predates the records — rather
than discharging the obligation as an incidental Gap. Lineage missing for
behavior the change does not affect remains an ordinary Gap observation for
separately assigned work.

## Additive observations

After a closing checkpoint, earlier Gap statements and observations remain
unchanged. Later work may append attributed observations under Evidence,
Impact, or References, identifying the originating work, run, or head and
linking available records. Differing interpretations may coexist as unresolved
observations; an addendum does not adjudicate them or reopen a prior disposition.

Reporting leaves State and Resolution unchanged. Separately assigned
disposition work may update those two sections through a forward commit with
the evidence or Decision required by the relationship rules.

## Rules

- Surface concerns encountered in the assigned work; this is incidental
  reporting, not an assignment to hunt for Gaps or reconcile the Gap register.
- Prefer an admitted Gap over invented certainty or unassigned remediation.
- A Gap preserves uncertainty without narrowing a Promise or creating a
  Promise, Candidate, Decision, or remedy.
- Risk is assessed when needed from Gap plus Promise/Oracle/Witness/Decision
  state; it is not duplicated here.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
