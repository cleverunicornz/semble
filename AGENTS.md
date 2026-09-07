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

<bedrock-organization>
## Clever Unicorn operating axioms

These axioms apply to every managed repository. This block is
organization-owned and synchronized by Bedrock; agents do not edit it.
Repository-specific orientation belongs in the repository block that follows.

### How we work

- Code is a possibility space: a surface does everything it can do, not what
  its author meant. A green test proves a behavior exists, never that nothing
  else happens. Declare what was not collapsed.
- A gap has three suspects: the code, the requirement, or the instrument.
  Interrogate in the open before displacing any of them.
- Situate before acting; re-situate after. Interrogate a contradiction before
  displacing what it contradicts.
- Completion is behavior at the promised boundary. Nothing delivered is a
  stub, placeholder, or deferred branch.
- Make it first, prove it after: build the slice, prove it, fix, continue.
- Verify by regenerating from source, never by reading the claim. A gate claim
  cites a CI run URL, never a local attestation.
- Predeclare criteria before the run that answers them and judge only against
  them. A refuted hypothesis is a successful experiment; mixed outcomes stay
  mixed.
- The dependency boundary is the assurance boundary. Pinned versions are
  assured versions and bumps are deliberate acts. A missing capability at a
  consumed boundary is blocking: stop and escalate rather than work around it.
- Public-first: before building inside, ask why it cannot be a public crate or
  repository.
- Decisions are append-only: supersede, never edit.

### Git and workflows

- Force push does not exist. Nothing pushed is deleted. One writer per ref.
  Every trunk change lands through a pull request.
- Passing branch protection and satisfying review requirements make a pull
  request mergeable; they do not authorize an agent to merge it. An agent
  opens or updates a pull request and leaves it open unless the active task
  explicitly authorizes that agent to merge that exact pull request.
- An owned repository uses `main` as its default working trunk.
- Pull requests are orchestrated. Workflows trigger on `pull_request` with
  `types: [opened, reopened, ready_for_review]`, on explicit dispatch, or on a
  Bedrock request. Branches carry no `push` trigger; `push` to main exists only
  for release and deployment witnesses. CI runs once when a pull request opens
  and once on its final head by dispatch before merge.
- Linux and platform-neutral jobs run on the owned automation fleet through
  logical labels; WarpBuild only for native macOS and Windows artifacts. Fork
  pull requests never reach the fleet. A missing host tool is a P0 defect,
  never a hidden substitute. CI runs the real suite.
- One fixed toolchain per repository with canonical task names.
- A pull request that carried a Bedrock closure merges with a merge commit,
  never a squash or rebase, so its checkpoint commits stay reachable from the
  trunk.

- Every pull request into a Bedrock-enrolled trunk requires a completed
  Bedrock closure before it can merge. Request the closure with the exact
  Integrity phrase when the pull request is ready for it, and never merge
  before the closing checkpoint; branch protection enforces the same gate
  through the Bedrock review.
- All internal reach rides the tailnet. Public-IP access is break-glass only.

### Forks

- A fork of an upstream repository keeps `main` as its upstream trunk and
  works on `internal/main`, its default working trunk. Every work branch is
  `internal/<name>` or `upstream/<name>`; the organization rulesets admit no
  other name.
- Both `main` and `internal/main` accept updates only through pull requests.
- Advancing `main`, including an upstream sync, requires approval from a human
  fork maintainer. This review gate is independent of authority to execute the
  merge.
- `internal/main` has no standing human-review requirement. Bedrock runs there
  and nowhere else.
- Contributions travel outward only: cherry-pick from `internal/main` onto an
  `upstream/<name>` branch cut from `main`, open the pull request into `main`,
  and from `main` open the pull request to the parent. Bedrock-owned files are
  never cherry-picked; keep record changes in separate commits from code so a
  cherry-pick stays clean.
- During a Bedrock closure, upstream-owned files are not edited, removed, or
  rewritten merely to impose fork orientation; that orientation lives only in
  the root blocks and `situation/`. This does not restrict ordinary product
  work through `internal/main`.

### Tools and knowledge

- Inside a repository, semantic search comes first: `semantic_index_status`,
  then `semantic_search`, then exact reads. Grep, glob, and broad reads follow
  semantic results. Outside the repository tree, use `rg` and exact paths.
- Tool-specific skills live at the harness user level and install with their
  plugin. Managed repositories carry no skills directory. A repository procedure
  is a Reference owned by the Invariant that requires it or the Promise it
  satisfies. Fleet-wide procedures are named plugin skills invoked by exact
  name and never restated.
- Scratch work lives under `/Volumes/code/temp/` on the home server and under
  the job home on a fleet runner. Remove it when the task completes.
- Bedrock run transcripts are archived outside the repository at
  `s3://cvu-automation-runs-uk/bedrock/<owner>/<repo>/pr-<number>/<run-id>/` on OVH
  Object Storage in the UK region. Each closure's opening and closing
  checkpoint commits carry that URI in a `Bedrock-Transcript` trailer.
</bedrock-organization>

<bedrock-repository>
## semble

- Identity: This repository is the Clever Unicorn fork of the Python `semble` code-search package and produces a library, CLI, and context-bound MCP server for coding agents.
- Ownership: `UPSTREAM_FORK` — public upstream: https://github.com/MinishLab/semble. Synchronization and contribution follow the organization's fork rules in the root organization block.
- Phase and implementation map: `situation/context.md`.
- Critical invariants: none
- Verification: Unassured — no assured witness route is presently recorded for fork gate claims; [G-000002](situation/gaps/G-000002-no-assured-fork-gate-route.md) retains the absence and [C-000002](situation/candidates/C-000002-qualify-fork-gate-assurance.md) proposes qualification without promoting it.
- Tool priority: organization defaults
- Donor boundary: `3f93c19ba404dd06c09e6fbbcb9740ded5d6bc1f` (the BACKPORT opening checkpoint; trigger head `fe65ccf3197446ee06ad3ab7d023b798219a2b0d`).
</bedrock-repository>
