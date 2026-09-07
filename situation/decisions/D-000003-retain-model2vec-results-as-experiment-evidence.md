## Status

accepted

## Date

2026-09-07

## Context

The new Model2Vec smoke and full contrastive paths produced retained run evidence, including a smoke candidate that regressed on its held-out metrics and a full candidate with resolved synthetic-corpus gains but no statistically resolved real-repository result sufficient for production selection.

## Evidence

- [W-000001](situation/witnesses/P-000003/W-000001-qwen-rust-smoke.md) retains the smoke observation whose receipt reports regression on every held-out retrieval metric.
- [W-000002](situation/witnesses/P-000004/W-000002-full-rust-contrastive-run.md) retains the full CPU observation and its experiment-only boundary.
- [W-000003](situation/witnesses/P-000004/W-000003-full-rust-contrastive-reproduction.md) retains the subsequent reproduction and its non-deployment boundary.
- [P-000003](situation/promises/P-000003-model2vec-training-smoke.md), [O-000003](situation/oracles/O-000003-model2vec-training-smoke.md), [P-000004](situation/promises/P-000004-full-rust-contrastive-evaluation.md), and [O-000004](situation/oracles/O-000004-full-rust-contrastive-evaluation.md) bound the experiment mechanics and their judgments.

## Decision

Retain the smoke and full contrastive artifacts as experiment evidence and do not select or deploy any Model2Vec candidate represented by those observations.

## Why

Completion of an offline training run and improvement on one synthetic corpus do not establish the production retrieval behavior required to replace a Semble model. The retained results themselves delimit that conclusion, and preserving both regression and mixed evidence keeps a later decision falsifiable.

## Rejected alternatives

- Promote the Qwen-distilled smoke candidate. Rejected because its retained held-out metrics regressed and its receipt explicitly rejects deployment.
- Promote the full raw or post-SIF candidate from the synthetic-corpus result alone. Rejected because the retained real-repository evidence does not provide the required production qualification.
- Remove the unsuccessful or mixed receipts. Rejected because failed and mixed observations are decision evidence.

## Consequences

- [I-000002](situation/invariants/I-000002-model2vec-results-require-recorded-promotion.md) makes the non-promotion boundary binding.
- [P-000003](situation/promises/P-000003-model2vec-training-smoke.md) and [P-000004](situation/promises/P-000004-full-rust-contrastive-evaluation.md) remain `implemented`, not assured or deployed behavior.
- The retained receipts remain available as evidence for any later candidate qualification.

## Revisit when

A replacement candidate has a separately recorded Promise, Oracle, and PASS Witness covering an explicitly chosen production-retrieval qualification scope, and a new Decision evaluates that evidence.
