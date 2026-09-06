# Decisions

A decision records why a choice collapsed. It preserves rationale so the
same path is not relitigated without new evidence. Decisions are append-only:
supersede, never edit.

## File naming

```
D-<six digits>-<kebab-case-name>.md
```

## Required headings

- `Status` — `accepted`, `superseded`, or `reversed`
- `Date`
- `Context` — the situation that forced a choice
- `Evidence` — links to witnesses and oracles that informed the choice
- `Decision` — what was chosen, stated directly
- `Why` — the reason the choice collapsed this way
- `Rejected alternatives` — what was not chosen and why
- `Consequences` — what follows from the decision
- `Revisit when` — the condition under which this decision may be reopened

Superseded decisions additionally link `Supersedes` and `Superseded by`.

## Relationship to invariants

A decision states the why. An invariant states the resulting rule. When a
decision produces a binding rule, write the invariant and link this decision
as its Basis. Decisions do not appear in the root `AGENTS.md` directly;
invariants do.

## Relationship to promises

When a promise is refuted or superseded, a decision records why. The promise
links the decision in its state evidence.

During BACKPORT, a donor's selected provider, rejected alternative, or stated
revisit condition is a collapsed choice and requires a Decision even when the
corresponding Promise remains `hypothesis` for lack of feasibility evidence.

A Candidate promotion/rejection Decision links the Candidate and every Promise
and Oracle created by promotion. Promotion is incomplete unless all records and
state changes land atomically.

## Rules

- One decision per file, immutable from the first closing checkpoint that
  follows its creation or change. Until then, on the open pull request, it may
  be corrected in place by a forward commit.
- Supersession replaces a decision with a new record; both remain.
- A decision without evidence is a preference, not a decision; say which it
  is.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
