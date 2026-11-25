# Calico — working agreement for agents

Read `docs/architecture.md` first. It is the map; this file is the rules. Name things with the words in its Vocabulary table (story, location, view, occasion, taste…) in code, docs and replies.


## Commands

uv manages everything (Python 3.12, `uv.lock`). Never `pip install` into the project.

```bash
make check        # format check + lint + strict types + tests with coverage. Must pass before you finish.
make fix          # auto-format and auto-fix lint
make test         # tests only (fast; ML tests skip without the extra)
uv sync              # everything, including PyTorch + open_clip (the default `ml` group)
uv run pytest -m ml       # real-CLIP tests (need the extra)
uv run pytest -m browser  # drive the web UI in Chrome via Playwright (needs `make ui`)
make ui-check        # UI: Biome lint, tsc, Vitest
make api-types       # after ANY change to an API response model: regenerate the UI's types
```

## Rules

1. **Tests first.** Write the failing test, watch it fail, then make it pass. Every behaviour has a test; every bug fix starts with a test that reproduces it.
2. **Tasks and models are pure.** They receive arrays in a `TaskContext` /`FeatureView` and return values. No filesystem, database, network or global state. I/O lives in `catalog/`, `features/` (decode only), `runs.py` and the interfaces.
3. **One way to do each thing.** Selection logic lives in `tasks/` and `selection/` only. The CLI and web call `Engine.select`; neither ranks anything itself.
4. **Explain every pick.** Scores are built from `Evidence` (value in [0, 1] × weight, weights sum to 1). Record per-photo signals with `ctx.trace.record` so `calico runs explain` can answer "why not this one".
5. **Never fail silently.** If an input is ignored or a fallback is taken (no text encoder, an empty filter), call `ctx.trace.note(...)`. Errors a user can fix are `CalicoError` subclasses whose message says how to fix them.
6. **Version what you compute.** Bump an extractor's or model's `version` when its output changes. Cached features are keyed by it.
7. **Deterministic.** Seeded randomness only; ties break by path.
8. **The archive is read-only.** Nothing writes under a collection root.
9. **No personal conventions in code.** Folder naming, vocabulary and thresholds are configuration (`calico.toml`, taxonomies, task options).
10. **Match the surrounding style.** Typed (`mypy --strict`), small modules, no commented-out code, no `print` outside `cli.py`. New code carries no rationale in comments or docstrings: why goes in a decision record (`docs/decisions/`, if present locally), the pull request, or the improvement loop's ledger.

## Working here

* Start every session from the newest `main` that passed CI: `git fetch`, then check that `gh run list --branch main --workflow ci --status success --limit 1 --json headSha -q '.[0].headSha'` equals `git rev-parse origin/main`. If it does not, look at that commit's own run: `gh run list --branch main --commit "$(git rev-parse origin/main)" --workflow ci --limit 1 --json status,conclusion`. If it is still running (or has no run yet), it has not failed: wait for it, or branch from the commit the first command printed. If it failed or was cancelled, tell the owner which commit it was and branch from the commit the first command printed. Commit each finished step and push it. CI runs `make check` on GitHub's Linux and macOS runners for every branch push and every pull request; Dependabot's branches run it through their pull requests only.
* `make setup` switches on the repository's git hooks (`scripts/git-hooks/`). They refuse private material, photos, data files, real home or volume paths and content keys. Never get around them (`--no-verify`, `git add -f`, changing `core.hooksPath`); if one refuses a file, the file does not belong in the repository.
* Without the `ml` group, ML tests skip.
* **Improving picks.** No plan, gate or handoff depends on new ratings or on a rating or review session: changes are judged on the data Calico already holds. Ratings given in ordinary use still train Yours.
* **Private material, if present locally.** The owner's checkout also holds private material at the paths the tools expect (`.private/`, `.claude/`, every `docs/` file but `architecture.md`, `GLOSSARY.md`, the improvement loop's guides and data under `claude/improvement-loop/`, `tests/private/`), kept in a separate repository (`.private.git`; `make private-status`, `make private-save`). The public repository ignores all of it: never copy any of it, or a name, path or place from it, into a tracked file, a commit message or a pull request. When it is present, read `claude/improvement-loop/RUNBOOK.md` before working on the loop; evaluation and training run on a copy of the calico home in a sandbox (`claude/improvement-loop/loop`). Reading the archive, and installing a newly trained Yours into the real home, are for the user: give the exact command.

## Reviews

* **Plans.** Before adding a data source, changing a default in `src/calico/` or writing a decision record, have the plan reviewed adversarially (the `plan-critic` agent, or `plan-critic-stats` when it touches a measurement or a verdict, if present locally) and keep the review with the decision or the pull request.
* **Numbers.** The `stats-reviewer` agent (if present locally) audits a measured number before it is adopted or written down.
* **Fanouts.** Any fanout of read-only agents is led by the `coordinator` agent (if present locally), at most 20 agents at once.
* **Pull requests.** Before opening one, run `/code-review` and the `code-reviewer` agent (if present locally). CodeRabbit reviews it on GitHub; answer every comment, by a fix or a reason, before asking the owner to merge (the `pr-cycle` skill, if present locally).

## Testing

* Build test data with `tests/factories.py` (`scene`, `save_jpeg`, `photo`, `view`, `context`, `FakeModels`). Do not hand-roll fixtures that already exist there.
* Every test runs with an isolated `$CALICO_HOME` (autouse fixture in `conftest.py`).
* Tests needing PyTorch are marked `@pytest.mark.ml` and skip without the extra. CI runs them, and fails when more tests skip than its workflow's `CALICO_MAX_SKIPPED`: a new skip on Linux or macOS means raising that number.
* When an ML extractor's output changes on purpose (and its `version` is bumped), write its golden output again on macOS: `CALICO_WRITE_GOLDEN=1 uv run pytest -m ml tests/features/test_golden.py`.
* No test may read the real archive or the user's calico home.

## Diagnosing (look before you change)

```bash
uv run calico sandbox /tmp/cal --force      # synthetic archive + isolated home
export CALICO_HOME=/tmp/cal/home            # everything below now uses the sandbox
uv run calico doctor                        # environment, coverage, failures
uv run calico locations --json              # location names and likely duplicates
uv run calico -vv select best /tmp/cal/archive --json > /tmp/r.json
uv run calico runs explain <run-id> <file>  # every signal for one photo, picked or not
CALICO_SCREENSHOTS=/tmp/shots uv run pytest -m browser   # look at the UI
```

Never point experiments at the user's real calico home or archive.

## Files you make

Nothing for this project goes outside this checkout (not the Desktop or Downloads). A file for the user to open (a `--sheet`, an `eval --json` report, a screenshot) goes in `out/` (git-ignored), named by date and what it measured; so does the output of a command you give the user to run. Your own working files go in your scratchpad. What must outlast the session goes in the docs, never only in `out/`.

## Where things go

| Change | File(s) |
|---|---|
| New menu entry | `src/calico/tasks/<name>.py` + entry point + `tests/tasks/test_<name>.py` |
| New vocabulary | `src/calico/models/taxonomies/<name>.toml` |
| New trained model | `src/calico/models/<name>.py` + entry point |
| New image feature | `src/calico/features/<name>.py` + entry point |
| Shared ranking helper | `src/calico/selection/` |
| New light, colour or mood word for measuring | `src/calico/looks.toml` (used by the opt-in `parts-looks` scorer) |
| New way of reading "Describe it" | `src/calico/criteria.py` (add to `SCORERS`) + test; measure with `calico eval criteria` |
| New model for reading "Describe it" | An `Embedder` extractor (see `features/clip.py`: `match_probability` if it can tell a match) + entry point; compare with `calico eval criteria --embedder NAME`; switch with `describe_embedder` |
| A track's options (its champion) | `src/calico/tracks/<track>.toml`, kept equal to the loop's `tracks/<track>/champion.toml` |
| A new or changed domain word | `docs/architecture.md`'s Vocabulary table (and `GLOSSARY.md`, if present locally) |
| A design decision, with its evidence | The pull request, and `docs/decisions/NNNN-title.md` if present locally; `docs/architecture.md` quoting a constant's value → `MANIFEST` in `tests/test_docs.py` |
| UI screen or component | `frontend/src/pages/`, `frontend/src/components/` + a Vitest test; logic goes in `frontend/src/lib/` (pure, tested) |

UI rules: run node only as `uv run npm …` inside `frontend/`; never hand-edit `frontend/src/api/schema.ts` or `fingerprint.ts` (generated); every user-visible string comes from the API or `frontend/src/lib/format.ts`; keep it keyboard-accessible and working at 375px.

# Pull Requests

- **Pull requests: only for finished work of these kinds**: a complete feature, a completed fix for an error in the repo, or completed work on an existing feature, such as an improvement to something already merged.
- **No pull request for a step along the way.** Keep intermediate work on its branch until the feature or fix is done.
- **No pull request for a small fix, planning note, or docs update** Commit these on the branch of related open work, so they goes in with that pull request. If no related work is open, leave it on its own branch and tell the user where it is.
- When work qualifies, open the pull request and say which kind above it is. If it's unclear whether it qualifies, suggest a pull request instead of opening one.
