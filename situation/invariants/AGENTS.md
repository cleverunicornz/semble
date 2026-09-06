# Invariants

An invariant states a binding repository rule. It is the rule itself, not
the reason the rule exists.

## File naming

```
I-<six digits>-<kebab-case-name>.md
```

## Required headings

- `Priority` — `critical` or `standard`
- `Invariant` — the rule, stated directly
- `Basis` — optional links to supporting records

## Priority

`critical` invariants must appear in the repository root `AGENTS.md`.
Violation before retrieval could corrupt data, break authority boundaries,
invalidate assured behavior, compromise security, or cause irreversible
architecture drift.

`standard` invariants are binding but discoverable from this folder or from
the relevant promise before changing that surface.

## Basis

Some invariants are axioms adopted without adjudication; they omit Basis.
Some invariants derive from a decision; the Basis section links the decision
that explains why:

```
## Basis

- [D-000014](situation/decisions/D-000014-postgres-authority.md)
```

The decision states the why. The invariant states the rule. An agent that
needs the full rationale follows the link.

## Rules

- State the rule positively and directly; do not enumerate negative space.
- One invariant per file.
- An invariant does not restate what an assured promise already covers
  unless the rule must be resident in the root `AGENTS.md`.
- Changing a critical invariant requires a decision unless it is an axiom
  being restated.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
