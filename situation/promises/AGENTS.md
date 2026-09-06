# Promises

A promise states falsifiable behavior the repository claims, intends, or is
attempting to provide. It does not prove itself and does not contain design
rationale.

## File naming

```
P-<six digits>-<kebab-case-name>.md
```

Witnesses for a promise live under `witnesses/P-<same six digits>/`.

## Required headings

- `State` — the lifecycle state
- `Promise` — the behavior, stated directly
- `Scope` — what the promise covers
- `Oracle` — link to the judgment rule
- `State evidence` — links justifying the current state
- `Residual` — what the promise deliberately does not assure
- `References` — optional links into `references/`

## States

```
hypothesis      stated, no feasibility evidence
qualifying      feasibility or evidence assessment in progress
qualified       evidence shows implementation is feasible
implementing    implementation work is active
implemented     code exists; assurance not yet complete
assuring        oracle being applied, witnesses being collected
assured         named oracle passed on a named witness; residual recorded
refuted         evidence shows the promise cannot or should not hold
withdrawn       intentionally abandoned without refutation
superseded      replaced by another promise; link the successor
```

A promise need not visit every state. Simple work may move
`hypothesis → implementing → implemented → assured`.

## State evidence

Every state transition cites its cause:

- `qualified` cites a qualifying witness;
- `implemented` cites the implementation commit or PR;
- `assured` cites the oracle and a passing witness;
- `refuted` cites a failing witness and a decision;
- `superseded` cites the replacement promise and a decision.

State never rests on uncited judgment.

## Assurance coverage

A promise may enter `assured` only when its cited PASS witnesses cover every
behavior asserted in the Promise section. Any behavior not exercised by the
witnesses must be named explicitly in Residual as outside the assurance.
Residual cannot silently narrow the behavior marked invariant by `assured`.

## Rules

- A promise states behavior, not implementation detail.
- Design rationale belongs in a decision, linked from References if needed.
- Every assured promise is invariant behavior; changing it requires
  supersession, a decision, a replacement oracle, and new witnesses.
- Refuted and withdrawn promises are retained; they are evidence.
- A Promise promoted from a Candidate links that Candidate and the selecting
  Decision. Direct feature work may create a Promise without a Candidate.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
