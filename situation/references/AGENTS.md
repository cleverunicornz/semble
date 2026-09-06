# References

References hold retained depth: research, long-form material, rejected
paths, historical context, and anything a record points to that is too large
to live inline.

## Structure

References are organized by the record that owns them:

```
references/P-000001/
references/O-000003/
references/D-000014/
```

A reference is a child of the record that links it.

## Rules

- A reference must be linked from at least one record; unowned files here
  are misplaced.
- References are depth, not law. They never override a promise, oracle,
  decision, or invariant.
- Git is the canonical store for historical donor bytes. Copy donor material
  here only when active human/agent retrieval needs an in-tree reference; any
  copy must be owned by and linked from a record.
- Any content type is acceptable: markdown, images, data, diagrams.

## Procedures

A repository procedure is a Reference owned by the Invariant that requires it
or the Promise it satisfies. Guidance that fits in a few lines and applies to
every session belongs in the root `AGENTS.md` blocks instead. Organization-wide rules, such as fork synchronization and contribution, live
in the root organization block; tool procedures are named skills supplied
outside the repository. A repository never restates either.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.

## Donor provenance

Git is the donor receipt. BACKPORT's opening checkpoint identifies the exact
trigger tree. Do not copy donor snapshots or create donor registries.

On DELTA, `git diff <last closing checkpoint>..<trigger head>` is the complete
review surface. An empty diff means there is no closure work; a nonempty diff
bounds it. Historical donor bytes remain available through `git show`.
