"""Backend diagnostic probes for --backend-info sanity checks."""

from __future__ import annotations

from dataclasses import dataclass

from src.sem_cat.utils.text_utils import contains_ascii_letters


@dataclass(frozen=True)
class ProbeResult:
    """Result of a single translation probe."""
    source: str
    output: str | None
    status: str  # "PASS", "WARN", "FAIL"
    notes: list[str]


# Probe cases for RU->EN models.
# Using explicit Unicode Russian literals.
RU_EN_PROBES: list[str] = [
    "\u0434\u043e\u043c",       # дом -> house
    "\u043a\u043e\u0448\u043a\u0430",  # кошка -> cat
    "\u0432\u043e\u0434\u0430",       # вода -> water
]


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    return text.strip().lower() if text else ""


def _run_probe(translator, source: str) -> ProbeResult:
    """Run a single translation probe and evaluate the result."""
    notes: list[str] = []

    try:
        output = translator.translate(source)
    except Exception as e:
        return ProbeResult(
            source=source,
            output=None,
            status="FAIL",
            notes=[f"Translation raised {type(e).__name__}: {e}"],
        )

    if output is None:
        return ProbeResult(
            source=source,
            output=None,
            status="FAIL",
            notes=["Output is None"],
        )

    if not output.strip():
        return ProbeResult(
            source=source,
            output=output,
            status="FAIL",
            notes=["Output is empty after stripping"],
        )

    normalized_output = _normalize(output)
    normalized_source = _normalize(source)

    # Check if output is identical to source (no translation happened)
    if normalized_output == normalized_source:
        return ProbeResult(
            source=source,
            output=output,
            status="WARN",
            notes=["Output is identical to source text (no translation)"],
        )

    # Check for ASCII letters (expected for RU->EN)
    if not contains_ascii_letters(output):
        return ProbeResult(
            source=source,
            output=output,
            status="WARN",
            notes=["Output contains no ASCII letters (unexpected for RU->EN)"],
        )

    # Check for suspiciously long output
    if len(output) > len(source) * 5:
        return ProbeResult(
            source=source,
            output=output,
            status="WARN",
            notes=[f"Output is {len(output)} chars vs {len(source)} source (suspiciously long)"],
        )

    return ProbeResult(
        source=source,
        output=output,
        status="PASS",
        notes=[],
    )


def run_backend_diagnostics(translator) -> list[ProbeResult]:
    """Run diagnostic probes against a translator.

    Returns a list of ProbeResult for each probe case.
    """
    return [_run_probe(translator, source) for source in RU_EN_PROBES]


def summarize_diagnostics(results: list[ProbeResult]) -> tuple[str, str]:
    """Summarize diagnostic results into an overall status and message.

    Returns:
        Tuple of (overall_status, summary_message).
        overall_status is one of "OK", "WARN", "FAIL".
    """
    if not results:
        return "FAIL", "No probes were run."

    statuses = [r.status for r in results]
    fail_count = statuses.count("FAIL")
    warn_count = statuses.count("WARN")
    pass_count = statuses.count("PASS")

    lines: list[str] = []
    for r in results:
        status_icon = {"PASS": "\u2713", "WARN": "\u26a0", "FAIL": "\u2717"}.get(r.status, "?")
        output_preview = (r.output[:40] + "...") if r.output and len(r.output) > 40 else (r.output or "(none)")
        lines.append(f"  {status_icon} '{r.source}' -> {output_preview!r}")
        for note in r.notes:
            lines.append(f"      {note}")

    summary = "\n".join(lines)

    if fail_count == len(results):
        return "FAIL", f"All {len(results)} probes failed.\n{summary}"

    if fail_count > 0 or warn_count > 0:
        details = []
        if pass_count:
            details.append(f"{pass_count} passed")
        if warn_count:
            details.append(f"{warn_count} warnings")
        if fail_count:
            details.append(f"{fail_count} failed")
        return "WARN", f"Probes: {', '.join(details)}.\n{summary}"

    return "OK", f"All {len(results)} probes passed.\n{summary}"
