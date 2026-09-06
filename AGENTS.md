<bedrock-protocol>
## Repository knowledge protocol

This repository operates under the Bedrock knowledge protocol. These policies
are repository law: follow them as written; do not readjudicate them during
ordinary work.

Before changing code, behavior, architecture, repository policy, documentation,
or planned work:

1. Read `situation/AGENTS.md`.
2. Read the relevant situation records, related open Gaps, and qualifying
   Candidates.
3. Read the nested `AGENTS.md` governing every situation namespace you will
   modify.
4. Update affected records in the same work as the repository change.
5. Treat `situation/` as canonical repository knowledge. README is human-facing
   orientation; neither README, comments, plans, nor pretrained assumptions
   override situation records.

The record classes are:

- **Promises** state falsifiable behavior and carry lifecycle state.
- **Oracles** define how Promises are judged.
- **Witnesses** retain immutable observations from actual runs.
- **Decisions** preserve why choices were selected or rejected.
- **Invariants** state binding repository rules.
- **Gaps** record bounded, repository-relevant absences.
- **Candidates** record evidence-derived possibilities, not commitments.
- **Plans** group Candidates and Promises into work without restating them.
- **References** retain supporting depth.

Git is the run's append-only event log. A run performs one closure on one pull
request branch, bounded by an opening checkpoint commit and a closing checkpoint
commit on that branch. Agents commit and push completed units of work promptly;
corrections are new forward commits. Published history is never amended,
rebased, reset, or force-pushed. Only opening and closing checkpoints define the
run container; interior commit count and shape are not prescribed.

Every trunk change lands through a pull request. Passing branch protection and
satisfying review requirements make a pull request mergeable; they do not
authorize an agent to merge it. An agent opens or updates a pull request and
leaves it open unless the active task explicitly authorizes that agent to merge
that exact pull request.

Run reports — closer summary, validator docket, corrector summary — are pull
request comments, never repository files. Agent transcripts are archived outside
the repository; both checkpoint commits carry the archive URI in a
`Bedrock-Transcript` trailer alongside their other trailers. The checkpoint
commits are the only writers of the closure state in `situation/context.md`.

A failed run is never resumed. An opening checkpoint with no closing checkpoint
marks a failed closure: the pull request is closed with a pointer to its rerun,
and the rerun starts on a new branch from the head admitted before the failed
run. The failed branch and its comments remain the record.

A record is immutable from the first closing checkpoint that follows its
creation or change. Until then, on the open pull request, it may be corrected
in place by a forward commit.

Every assured Promise is invariant behavior. Changing it requires a superseding
Promise, a Decision explaining the change, a replacement Oracle, and new
Witnesses.

Gaps record what relevant capability, evidence, decision, implementation, or
instrument is absent. Candidates are possible responses derived from evidence.
A Candidate becomes behavior only through a Decision that promotes it into a
falsifiable Promise with an Oracle. Plans qualify Candidates and implement or
assure Promises.

The learning loop is:

```text
Promise -> implementation -> Oracle -> Witness -> disposition
        -> Gap -> Candidates -> Plan -> Decision
        -> promoted Promise + Oracle -> implementation
```

Repository files are referenced by repository-root-relative path. Historical
repository bytes use `<commit>:<path>`. External public files use full URLs.
External private files use declared `Private: owner/repo@<ref>#<path>`
coordinates; inability to fetch a declared-private reference is expected and
never grounds to stop, remove it, or invent its contents.

When repository orientation identifies an upstream fork, upstream
synchronization and contribution follow the organization's fork rules in the
root organization block. Bedrock records ownership and the upstream coordinate
and performs neither.

Root `AGENTS.md` carries three tagged blocks in this order: the protocol block
`bedrock-protocol`, the organization block `bedrock-organization`, and the
repository block `bedrock-repository`. The organization block is synchronized
by the closure automation and is optional; adopters whose automation supplies
none carry the other two blocks.

This protocol block is protocol-owned; the organization block is
organization-owned. Agents must not edit any byte inside either. Agents edit only the repository block, which holds all repository-specific
orientation in the shape given by the repository block template published with
the protocol release and reproduced in the closure automation.
</bedrock-protocol>

<bedrock-repository>
Legacy repository guidance at `9d2d3e9ba4dc4f986b0aad3f2d51fff7b0dce1ec:AGENTS.md` is BACKPORT donor
material, not current instruction. Repository-specific orientation will be
authored during this closure.
</bedrock-repository>
