# Candidates

A Candidate is an evidence-derived possible behavior or approach that may become
a Promise. It is not committed behavior, repository law, or planned outcome.

## File naming

```
C-<six digits>-<kebab-case-name>.md
```

## Required headings

- `State` — lifecycle state
- `Candidate` — the possible behavior/approach
- `Origin` — Gap, Witness, Oracle, Decision, user report, or other evidence
- `Why consider it` — bounded repository relevance
- `Qualification questions` — what must be resolved before promotion
- `Candidate approaches` — alternatives within this candidate, when any
- `Disposition` — resulting links, or explicitly none

## States

- `proposed` — evidence supports consideration
- `qualifying` — a Plan is evaluating it
- `promoted` — selected into Promise + Oracle by a Decision
- `rejected` — Decision records why
- `merged` — folded into another Candidate
- `superseded` — replaced by a more accurate Candidate

## Promotion transaction

Promotion is one atomic commit:

```text
Candidate -> promoted
Decision created
Promise created
Oracle created
Plan updated
linked Gap -> addressing, when applicable
```

Promotion requires precise falsifiable behavior, Scope, Residual, Oracle inputs,
Pass/Fail conditions, and the Decision selecting it. If these cannot be stated,
the Candidate remains `qualifying`.

## Relationship rules

- `promoted` links Decision, Promise, and Oracle.
- `rejected` links Decision.
- `merged` links destination Candidate.
- `superseded` links replacement Candidate.
- Every Candidate has an evidence-bearing Origin. Unbounded brainstorming does
  not become repository knowledge.

## Qualification plans

Plans contain Candidates being qualified and Promises being implemented or
assured. Qualification is ordinary Plan work over Candidates.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
