<bedrock-repository>
## semble

- Identity: This repository is the Clever Unicorn fork of the Python `semble` code-search package and produces a library, CLI, and context-bound MCP server for coding agents.
- Ownership: `UPSTREAM_FORK` — public upstream: https://github.com/MinishLab/semble. Synchronization and contribution follow the organization's fork rules in the organization layer (`git-etiquette` skill).
- Phase and implementation map: `situation/context.md`.
- Critical invariants: [I-000002](situation/invariants/I-000002-public-secure-material-boundary.md) — This repository is public. Secure material — our own known vulnerabilities from the security vault, their details, exploitability and affected code paths — is never represented in it: not in code, comments, commits, branches, issues, pull requests, reviews or comments. Public CVE and CWE references are fine. A fix lands normally, with a Promise that describes the property the code keeps, never the vulnerability.
- Verification: Unassured — no assured witness route is presently recorded for fork gate claims; [G-000002](situation/gaps/G-000002-no-assured-fork-gate-route.md) retains the absence and [C-000002](situation/candidates/C-000002-qualify-fork-gate-assurance.md) proposes qualification without promoting it.
- Tool priority: organization defaults
- Donor boundary: `3f93c19ba404dd06c09e6fbbcb9740ded5d6bc1f` (the BACKPORT opening checkpoint; trigger head `fe65ccf3197446ee06ad3ab7d023b798219a2b0d`).
</bedrock-repository>
