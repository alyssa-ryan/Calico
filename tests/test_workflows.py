"""CI runs ``make check`` on GitHub's own Linux and macOS runners, for every push to main and
every pull request, and nothing else: no self-hosted runner and no deploy (decision 0002)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"


def triggers(text: str) -> dict[str, str]:
    """The events under a workflow's top-level ``on:``, each with its own lines."""
    block = re.search(r"^on:\n((?:[ \t]+.*\n|\n)*)", text, re.MULTILINE)
    assert block, "no on: block"
    parts = re.split(r"^  ([a-z_]+):", block.group(1), flags=re.MULTILINE)[1:]
    return dict(zip(parts[::2], parts[1::2], strict=True))


def jobs(text: str) -> dict[str, str]:
    body = text[text.index("\njobs:\n") :]
    parts = re.split(r"^  ([a-z][a-z-]*):\n", body, flags=re.MULTILINE)[1:]
    return dict(zip(parts[::2], parts[1::2], strict=True))


def test_ci_is_the_only_workflow() -> None:
    assert [w.name for w in WORKFLOWS.glob("*.y*ml")] == ["ci.yml"]


def test_ci_runs_for_pushes_to_every_branch_but_dependabots_and_for_pull_requests() -> None:
    on = triggers(CI.read_text())
    assert set(on) == {"push", "pull_request"}
    assert re.search(r'branches: \["\*\*", "!dependabot/\*\*"\]', on["push"])


def test_ci_runs_only_on_githubs_own_runners_linux_and_macos() -> None:
    text = CI.read_text()
    assert "self-hosted" not in text
    assert sorted(re.findall(r"runs-on: (.+)", text)) == ["macos-latest", "ubuntu-latest"]


def test_every_action_is_pinned_to_a_commit() -> None:
    uses = re.findall(r"uses: (\S+)", CI.read_text())
    assert uses
    for action in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", action), action


def test_ci_reads_the_repository_only_and_cannot_run_for_hours() -> None:
    """Without a permissions block the token may write; without a timeout a hung job holds
    the runner for GitHub's six hours."""
    text = CI.read_text()
    assert re.search(r"^permissions:\n  contents: read$", text, re.MULTILINE)
    for name, job in jobs(text).items():
        assert re.search(r"^    timeout-minutes: \d+$", job, re.MULTILINE), name
        assert "persist-credentials: false" in job, name


def test_each_check_states_how_many_tests_it_expects_to_skip() -> None:
    """Linux skips the four macOS-only path tests and the two that need PyTorch missing;
    macOS only the two (tests/skip_limit.py)."""
    found = {
        re.search(r"^    name: (.+)$", job, re.MULTILINE).group(1): job  # type: ignore[union-attr]
        for job in jobs(CI.read_text()).values()
    }
    assert set(found) == {"Linux check", "macOS check"}
    for name, expected in (("Linux check", 6), ("macOS check", 2)):
        assert re.search(rf"CALICO_MAX_SKIPPED: {expected}$", found[name], re.MULTILINE), name


def test_tests_keep_away_from_any_real_calico_home() -> None:
    for name, job in jobs(CI.read_text()).items():
        assert "CALICO_HOME: ${{ runner.temp }}/" in job, name


def test_the_runner_context_is_used_only_inside_steps() -> None:
    """GitHub has no ``runner`` context outside a step, and rejects the whole workflow file
    that uses it there, so the workflow never runs."""
    for name, job in jobs(CI.read_text()).items():
        steps = job.find("    steps:")
        assert "${{ runner." not in job[:steps], name


def test_ci_installs_and_runs_the_ml_tests() -> None:
    """``make check`` runs ``uv run``, which installs every default group, PyTorch
    included, whatever ``uv sync`` left out; the workflow says what really runs."""
    assert "--no-group ml" not in CI.read_text()


def test_macos_runs_the_models_on_the_cpu() -> None:
    """GitHub's macOS runners report a GPU (MPS) whose results are wrong."""
    macos = jobs(CI.read_text())["macos"]
    assert re.search(r"^          CALICO_DEVICE: cpu$", macos, re.MULTILINE)


def ecosystems() -> dict[str, str]:
    text = DEPENDABOT.read_text()
    parts = re.split(r'^  - package-ecosystem: "([a-z-]+)"\n', text, flags=re.MULTILINE)[1:]
    return dict(zip(parts[::2], parts[1::2], strict=True))


def groups(block: str) -> dict[str, str]:
    body = block[block.index("    groups:\n") + len("    groups:\n") :]
    end = re.search(r"^    [a-z-]+:", body, re.MULTILINE)
    body = body[: end.start()] if end else body
    parts = re.split(r"^      ([a-z-]+):\n", body, flags=re.MULTILINE)[1:]
    return dict(zip(parts[::2], parts[1::2], strict=True))


def test_dependabot_covers_actions_python_and_the_ui_monthly_after_a_week_of_cooldown() -> None:
    found = ecosystems()
    assert set(found) == {"github-actions", "uv", "npm"}
    for name, block in found.items():
        assert re.search(r'^      interval: "monthly"$', block, re.MULTILINE), name
        assert re.search(r"^      default-days: 7$", block, re.MULTILINE), name


def test_each_ecosystem_opens_one_pull_request_per_kind_of_update() -> None:
    found = ecosystems()
    assert list(groups(found["github-actions"])) == ["actions"]
    for name, prefix in (("uv", "python"), ("npm", "ui")):
        kinds = groups(found[name])
        assert {prefix, f"{prefix}-major", f"{prefix}-security"} <= set(kinds), name
        assert 'update-types: ["minor", "patch"]' in kinds[prefix], name
        assert 'update-types: ["major"]' in kinds[f"{prefix}-major"], name
        assert "applies-to: security-updates" in kinds[f"{prefix}-security"], name
        assert 'update-types: ["minor", "patch"]' in kinds[f"{prefix}-security"], name


def test_libraries_that_change_computed_features_come_in_their_own_pull_request() -> None:
    kinds = groups(ecosystems()["uv"])
    assert list(kinds)[:2] == ["features", "features-security"]
    assert "applies-to: version-updates" in kinds["features"]
    assert "applies-to: security-updates" in kinds["features-security"]
    packages = (
        "torch",
        "open-clip-torch",
        "timm",
        "transformers",
        "rawpy",
        "pillow",
        "pillow-heif",
    )
    for package in packages:
        for group in ("features", "features-security"):
            assert f'"{package}"' in kinds[group], (group, package)


def test_majors_that_cannot_be_taken_yet_are_ignored() -> None:
    found = ecosystems()
    assert re.search(r'dependency-name: "open-clip-torch"\n\s+versions: \[">=4"\]', found["uv"])
    assert re.search(r'dependency-name: "typescript"\n\s+versions: \[">=6"\]', found["npm"])
    assert re.search(r'dependency-name: "@types/node"\n\s+versions: \[">=25"\]', found["npm"])
