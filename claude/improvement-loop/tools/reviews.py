from __future__ import annotations

from guard import Refused

REVIEWS = "claude/improvement-loop/reviews/"
CRITIC = "plan-critic-stats"
REFUSING = "DO NOT PROCEED"


def verdict_of(text: str) -> str | None:
    found = [line for line in text.splitlines() if line.startswith("Verdict:")]
    return found[-1].removeprefix("Verdict:").strip() if found else None


def refuse_unreviewed(path: str, text: str | None, *, committed: bool) -> None:
    if not path.startswith(REVIEWS) or not path.endswith(".md"):
        raise Refused(
            f"a registration names its adversarial review in `review`: a .md file under {REVIEWS}"
        )
    if text is None:
        raise Refused(f"{path} does not exist: have {CRITIC} review the plan and save it there")
    if not committed:
        raise Refused(f"{path} is not committed as it stands: `loop commit` it first")
    if f"Reviewer: {CRITIC}" not in text:
        raise Refused(f"{path} must be {CRITIC}'s review, with the line 'Reviewer: {CRITIC}'")
    verdict = verdict_of(text)
    if verdict is None or (not verdict.startswith("PROCEED") and not verdict.startswith(REFUSING)):
        raise Refused(
            f"{path} has no 'Verdict:' line (PROCEED, PROCEED WITH CHANGES or {REFUSING})"
        )
    if verdict.startswith(REFUSING):
        raise Refused(
            f"{path} says {verdict!r}: revise the plan and have it reviewed again, or ask the owner"
        )
