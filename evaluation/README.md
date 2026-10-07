# Price-prompt evaluation

The original replay is compared with a fresh-memory control built from each ticker's eligible history. The source is pinned inside the reproduction script, so it does not run the current code as its baseline.

From the repository root, in the pinned offline-check environment described in the main README:

```sh
python tests/reproduce_original_prompts.py . /tmp/trader-original-prompts
python tests/check_memory_replay.py
```

The original run covers 537 decision dates and 1,611 agent opportunities. It creates 1,609 price prompts, compared with 1,602 in the control. There are 14 prompt-or-call mismatches, including eight prompts with future rows and seven premature mid-term calls. These categories overlap; do not add them. The repaired replay check covers the same 537 dates and produces 1,602 prompts, all matching its eligible-prefix control.

[summary.json](original-prompts/summary.json) includes the source hashes, scope and example prompts. [decisions.csv](original-prompts/decisions.csv) contains one row per agent opportunity, including cases where no model call would occur. Both were reproduced byte-for-byte with pandas 2.3.1. The original script reads the pinned source and fixture through Git; a shallow clone must contain that commit.

Only price-input construction is compared. Model responses are substituted and semantic search returns no results. No fills, returns, causal trading impact, or historical news availability are measured. Fixture prices and historical outputs are outside the source-code license described in the main README.
