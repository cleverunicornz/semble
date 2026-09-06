# Situation System

All repository knowledge records live under `situation/`. This file explains
the structure, identifier rules, and relationships between namespaces.

## Structure

- `invariants/` — binding repository rules, explicit and citable
- `promises/` — falsifiable behavior claims with a lifecycle state
- `oracles/` — judgment rules that decide whether a promise holds
- `witnesses/` — immutable observations from real runs
- `decisions/` — append-only records of why a choice collapsed
- `gaps/` — bounded repository-relevant absences
- `candidates/` — evidence-derived possibilities, not commitments
- `plans/` — thin containers grouping candidates and promises into work
- `references/` — retained depth linked from records
- `context.md` — repository identity, phase, current/intended state, and
  closure state

## Identifier rules

Every record class carries a repository-scoped numeric identifier:

```
I-000001  invariant
P-000001  promise
O-000001  oracle
W-000001  witness
D-000001  decision
G-000001  gap
C-000001  candidate
PLAN-000001  plan
```

Identifiers are minimum six decimal digits, zero-padded, monotonically
allocated, never reused, never renumbered. Expansion beyond six digits is
allowed.

## Reference discipline

Every reference carries a visibility class so a reader never has to guess
whether a reference is broken or access-controlled. This section and the next
state that discipline once for every namespace.

- A file **inside this repository** is referenced by its repository-root-
  relative path: `situation/promises/P-000001-stable-identity.md`. Never a
  path that walks above the repository, never a machine-local path.
- Historical bytes from this repository are referenced as
  `<commit>:<repository-root-relative-path>` and resolved with `git show`.
  This is the canonical form when a BACKPORT later rewrites the live file.
- A file **in an external public repository** is referenced only by its
  full public URL, including the exact path to the file.
- A file **in an external private repository** is referenced by its
  coordinate — `Private: owner/repo@<ref>#<path>` — with a short access
  note. Never by an unauthenticated URL, never undeclared.
- Never reference a local clone of a public project as if other machines
  could access it. If the material matters, it is either committed into
  this repository (usually under `references/`) or linked by public URL.

## Private reference behavior

A declared-private reference that cannot be fetched is a normal, expected
state — not an error, not a missing file, and never grounds to stop work,
remove the reference, or invent its contents. Access to private
repositories is environment-dependent: a failed API call does not mean a
credentialed clone will not work. Agents use whatever access method their
environment provides; when the material is unreachable, they record the
need and proceed with available evidence. Content from a private reference
is never copied into a public document.

An undeclared reference that cannot be resolved is a defect. A declared
private reference that cannot be resolved is expected.

## Quantitative and provenance discipline

Every numeric claim names the exact repository-root-relative path or public
source it was counted from. Counts are re-derived from that source rather
than copied from another record. Words such as `donor`, `legacy`, `current`,
or `generated` always resolve to a named path, tree, commit, or URL.

## Relationships

```
Promise -> Oracle -> Witness -> disposition in the Promise
Gap -> Candidate -> Decision -> promoted Promise + Oracle
Decision -> Invariant (basis)
Decision -> Decision (supersedes)
Plan -> Candidate and Promise set
Reference -> owning record
```

A promise states behavior. An oracle states the judgment rule. A witness
retains one observation. The promise's state records the current disposition
after applying the oracle to available witnesses. A decision explains why a
path was accepted or rejected; an invariant states the binding rule that
results.

## Gaps, Candidates, and the learning loop

A Gap records a bounded absence relevant to existing repository behavior or
work. A Candidate records an evidence-derived possible response. Candidates are
not commitments. Plans qualify Candidates and implement/assure Promises. A
Decision promotes or rejects a Candidate; promotion creates the falsifiable
Promise and Oracle atomically.

```text
Promise -> implementation -> Oracle -> Witness -> disposition
        -> Gap -> Candidates -> Plan -> Decision
        -> promoted Promise + Oracle -> implementation
```

Risk is assessed when needed from Promise state, Oracle maturity, Witness
results, open Gaps, qualifying Candidates, Decisions, and dependencies. It is
not stored as a duplicate record class.

## Assured promises are invariant

Every promise in state `assured` is invariant behavior. Changing it requires
a new promise superseding the old one, a decision explaining why, a
replacement oracle, and new witnesses. The old record remains immutable
history.

## Repository phase

`context.md` states the repository's current phase: `INITIAL`, `PLANNING`,
`IMPLEMENTATION`, or `EVOLUTION`. Absence of implementation source is a
current-phase fact; it never implies the repository's purpose is
documentation. Future-facing records are valid when clearly represented as
intended rather than implemented.

## Bedrock operation

Repository phase and closure operation are separate classifications:

- `INITIALIZE` — no substantive donor or implementation; install the
  substrate and minimal orientation without inventing behavior.
- `BACKPORT` — substantive existing documentation/code, no completed Bedrock
  adoption; establish records from existing behavior and preserve already
  collapsed choices as Decisions.
- `DELTA` — a completed adoption exists; inspect only the pull-request diff
  and records it directly affects. Never re-derive unchanged donors.

Every run records its operation in its opening checkpoint.

DELTA reviews `git diff <last closing checkpoint>..<trigger head>`. That diff is
the complete review surface; an empty diff means there is no closure work. No
parallel donor registry or copied donor snapshot exists.

## Runs

A run performs one closure on one pull request branch. It is bounded by two
orchestrator commits: an opening checkpoint with subject
`bedrock: open closure <run-id>`, and a closing checkpoint with subject
`bedrock: complete closure <run-id>`. Interior commits are completed units of
agent work, pushed as they happen; corrections are new forward commits. Only the
checkpoints are prescribed; interior commit count and shape are not.

Checkpoint metadata is one contiguous final Git trailer block with no blank
lines between trailer lines, so `git interpret-trailers --parse` reads it. Agent
transcripts are archived outside the repository, and both checkpoints carry the
archive URI in a `Bedrock-Transcript` trailer alongside their other trailers.

Run reports — closer summary, validator docket, corrector summary — are pull
request comments. They are never repository files.

A failed run is never resumed. An opening checkpoint with no closing checkpoint
marks a failed closure: the pull request is closed with a pointer to its rerun,
and the rerun starts on a new branch from the head admitted before the failed
run. The failed branch and its comments remain the record.

A record is immutable from the first closing checkpoint that follows its
creation or change. Until then, on the open pull request, it may be corrected
in place by a forward commit.

## Closure state

`context.md` carries a `## Closure state` section written only by the
orchestrator's checkpoint commits, with exactly these lines:

```text
- Current run: `<run-id>` (open)
- Last completed closure: run `<run-id>`, opened at `<opening-commit-sha>`
- Transcript: `<uri>`
```

With no run open the first line is `- Current run: none`. Before the first
completed closure the second line is `- Last completed closure: none` and the
third is `- Transcript: none`.

The opening checkpoint sets `Current run`. The closing checkpoint sets
`Current run` to none and updates the other two lines. The most recent commit
with subject `bedrock: complete closure <run-id>` is the DELTA base.

## Repository ownership

- `OWNED` — normal repository; the operational trunk is its default branch.
- `UPSTREAM_FORK` — repository has an external upstream authority. Record the
  upstream public URL and identify the repository as a fork.

Every run records ownership and, for forks, the upstream coordinate in
`context.md` and the repository block of root `AGENTS.md`. Upstream
synchronization and contribution are separate operations outside Bedrock that
follow the organization's fork rules in the root organization block; Bedrock
does not restate them.

## README lifecycle

README is human-facing orientation and is always considered after records and
root `AGENTS.md` stabilize:

- `INITIALIZE` — minimal purpose and pointers; no invented behavior.
- `BACKPORT` — replace dense canonical detail with human orientation and
  pointers into `situation/`; Git retains historical donor bytes.
- `DELTA` — update only when the delta changes human-facing purpose, usage,
  setup, or capabilities; otherwise leave it unchanged.

README never overrides records under `situation/`.

During a Bedrock closure, upstream-owned files remain untouched by the
knowledge projection; fork orientation lives only in the root `AGENTS.md`
blocks and `situation/`. This restriction governs Bedrock alignment, not
ordinary product changes made through the fork's working trunk.

## Documentation classification

Repository-operational knowledge — architecture explanations, maintainer or
contributor procedure, plans, rationale, setup/status prose, and agent guidance
— is represented under `situation/`. In an owned repository it does not remain
as a competing documentation authority; in an upstream fork, upstream-owned
files remain as they are.

Documentation that is functionally part of the product remains in its native
path: website/help content, API/schema inputs, generated-code inputs, build or
test fixtures, release/legal material, and other files whose removal changes a
runtime, build, test, release, or delivered documentation artifact.

On `UPSTREAM_FORK`, Bedrock does not remove or rewrite upstream-owned
documentation to impose repository-operational orientation.

## Root AGENTS.md blocks

Root `AGENTS.md` carries three tagged blocks in this order:

- `bedrock-protocol` — protocol-owned; the published root protocol block,
  installed byte-for-byte and immutable to agents.
- `bedrock-organization` — organization-owned; synchronized by the closure
  automation and immutable to agents. It is optional; adopters whose automation
  supplies none carry the other two blocks.
- `bedrock-repository` — repository-owned; written by the closer in the shape
  given by the repository block template published with the protocol release
  and reproduced in the closure automation.

Agents edit only the repository block.

## AGENTS.md placement

AGENTS.md files may exist only at the repository root, `situation/`, and the
protocol namespace roots listed in Structure. Any other AGENTS.md is competing
operating authority: internalize relevant rules into situation records and
remove it.
