# Plans

A plan is a thin container that groups Candidates being qualified and Promises
being implemented or assured. It does not restate those records and does not
carry design detail.

## File naming

```
plans/<lifecycle>/PLAN-<six digits>-<kebab-case-name>.md
```

Lifecycle directories: `draft/`, `active/`, `done/`, `abandoned/`. The
directory is the lifecycle state; the file does not repeat it.

## Required headings

- `Candidates` — optional links to possibilities being qualified
- `Promises` — optional links to behavior being implemented or assured
- `Dependencies` — ordering between promises, when any exists
- `Completion` — the target states that constitute plan completion

## Rules

- A plan contains at least one Candidate or Promise, dependencies, and a
  completion condition. Nothing else.
- Candidate qualification plans complete only when every Candidate is
  promoted, rejected, merged, or superseded; promoted Candidates link the
  resulting Decision, Promise, and Oracle.
- Completion is written as a target condition ("completes when...") and is
  defined by promise states, never as a present-tense claim that target states
  already hold.
- Moving a plan between lifecycle directories is an explicit commit that
  explains the transition in its message.
- A done plan's promises must have reached the states its Completion
  section requires.

## Reference discipline

Reference discipline is defined in `situation/AGENTS.md`.
