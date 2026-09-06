# Gaps

A Gap records a bounded, repository-relevant absence: a capability, evidence,
decision, implementation, or instrument that is missing relative to behavior or
work this repository already claims or intends. It does not enumerate unlimited
possibility space and does not commit the repository to closing it.

## File naming

```
G-<six digits>-<kebab-case-name>.md
```

## Required headings

- `State` — lifecycle state
- `Gap` — the absence, stated directly
- `Relevance` — existing record or repository purpose that makes it matter
- `Evidence` — what establishes the absence
- `Impact` — what the absence blocks or weakens
- `Resolution` — current resolution, or explicitly none
- `References` — optional retained depth

## States

- `open` — evidenced and unresolved
- `addressing` — linked Candidate, Promise, or Plan is actively closing it
- `closed` — cited evidence proves the absence no longer exists
- `accepted` — a Decision explicitly tolerates it
- `superseded` — replaced by a more accurate Gap

## Relationship rules

- `addressing` links the Candidate/Promise/Plan acting on it.
- `closed` links the Promise/Oracle/Witness or commit proving closure.
- `accepted` links the Decision accepting it.
- `superseded` links the replacement Gap.
- A Promise Residual may link a Gap when excluded behavior remains relevant.

## Rules

- No evidence and no bounded relevance means no Gap.
- Prefer an admitted Gap over invented certainty.
- A Gap is not a Promise, Candidate, Decision, or derived risk narrative.
- Risk is assessed when needed from Gap plus Promise/Oracle/Witness/Decision
  state; it is not duplicated here.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
