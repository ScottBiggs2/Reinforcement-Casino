# Agent flow v2 — staged, NOT active

Two agent definitions adapted from `burtenshaw/multiautoresearch`
(`post-training/.pi/agents/`, commit "add post-training and inference", 2026-05-02).
The local snapshot in `docs/multiautoresearch_agents/` predates that commit, so these
two agents were missing from it.

**These files are deliberately not in `.claude/agents/`.** Dropping them there activates
them. Do not do that before the NeurIPS #29841 rebuttal closes on 2026-08-03 — changing
tooling mid-sprint is how you lose a day.

**To activate afterwards:** `cp docs/agent_flow_v2/rc-*.md .claude/agents/`

---

## Why these two

The four existing agents (`rc-researcher`, `rc-planner`, `rc-reviewer`,
`rc-cluster-reporter`) are all **read-only** — none has `Write` or `Edit`. That leaves
two jobs with no owner, and both were done by hand during the rebuttal cycle at real
cost:

| gap | what it cost |
|---|---|
| no memory keeper | Seven confident claims were asserted and later disproved. The "Retracted" section of `VERIFIED_FINDINGS.md` and `docs/run_logs/DO_NOT_REPEAT.md` were written *after* the damage, not as the work happened. Two of the seven had already propagated into three documents before anyone checked the arithmetic. |
| no experiment worker | Every sbatch was hand-written. Five landmines shipped in one script (colliding output dirs, shared wandb ids, a `--requeue` that cannot fire, a smoke test that skipped the path it was meant to validate, a wall-clock metric wrong in our own favour). All were caught by review, none by process. |

Upstream had already encoded the fixes:

- `memory-keeper` is given `read, grep, find, ls, edit, write` and **explicitly no
  `bash`** — it records, it cannot run experiments, so it cannot contaminate its own
  ledger.
- `experiment-worker`'s contract is *one coherent change → local smoke → then exactly
  one paid run*. The LoRA smoke is precisely what caught the silent-gradient failure
  here; upstream makes that a rule rather than an improvisation.
- `research/do-not-repeat.md` and `research/results.tsv` are standing artifacts, not
  something remembered at the end.

## What was changed in porting

The upstream files target `.pi`/OpenCode and a NanoChat + HF Jobs project. Claude Code
agents only read `name` / `description` / `tools` frontmatter, so `defaultContext`,
`inheritSkills` and `maxSubagentDepth` are dropped. Every command was replaced with the
Slurm/Discovery equivalent. **The structure is what transfers; none of the text does.**

## Already live (safe, data-only)

- `docs/run_logs/results.tsv` — one sortable row per measurement, backfilled with this
  cycle's results. Previously these were scattered across six markdown files with no
  way to sort or diff them.
- `docs/run_logs/DO_NOT_REPEAT.md` — seeded with seven disproved claims, six
  experiments that must not be re-run as designed, and five infrastructure traps.
