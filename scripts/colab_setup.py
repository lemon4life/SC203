#!/usr/bin/env python3
"""UNTESTED: install DoSSR's Colab dependencies without changing Colab's PyTorch."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402

DEFAULT_REQUIREMENTS = REPO_ROOT / "external" / "DOSSR" / "requirements.txt"
DEFAULT_DOSSR_DIR = REPO_ROOT / "external" / "DOSSR"
DEFAULT_DOSSR_URL = "https://github.com/AMD-AIG-AIMA/DOSSR.git"
EXCLUDED = {"xformers", "triton"}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def scrub_output(value: str) -> str:
    """Redact common credentials if a tool unexpectedly includes one."""
    value = re.sub(r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@", r"\1[REDACTED]@", value)
    value = re.sub(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b",
        "[REDACTED]",
        value,
    )
    value = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
    value = re.sub(
        r"(?i)(token|password|authorization|api[_-]?key)(\s*[:=]\s*)[^\s,;]+",
        r"\1\2[REDACTED]",
        value,
    )
    return value


def log_output(logger, output: str, level: str = "info") -> None:
    log_method = getattr(logger, level)
    for line in scrub_output(output).splitlines():
        if line.strip():
            log_method("%s", line)


def run_command(command: list[str], logger, *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", scrub_output(repr(command)))
    try:
        return subprocess.run(command, text=True, capture_output=capture, check=False)
    except OSError as error:
        return subprocess.CompletedProcess(command, 127, "", str(error))


def resolve_requirements(path: Path, repo_url: str, logger) -> bool:
    if path.is_file():
        return True
    if path.resolve() != DEFAULT_REQUIREMENTS.resolve():
        logger.error("Requirements file does not exist: %s", path)
        return False
    if DEFAULT_DOSSR_DIR.exists():
        logger.error(
            "DoSSR directory exists but requirements.txt is missing; leaving it untouched: %s",
            DEFAULT_DOSSR_DIR,
        )
        return False
    DEFAULT_DOSSR_DIR.parent.mkdir(parents=True, exist_ok=True)
    logger.info("DoSSR clone is absent; cloning upstream source only (no weights).")
    result = run_command(["git", "clone", repo_url, str(DEFAULT_DOSSR_DIR)], logger)
    log_output(logger, result.stdout or "")
    log_output(logger, result.stderr or "", "error" if result.returncode else "info")
    if result.returncode != 0 or not path.is_file():
        logger.error("Could not obtain %s; stopping before package installation.", path)
        return False
    return True


def filtered_requirements(source: Path, destination: Path, logger) -> set[str]:
    kept: list[str] = []
    skipped: set[str] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)", line)
        if match and canonical_name(match.group(1)) in EXCLUDED:
            skipped.add(canonical_name(match.group(1)))
            continue
        kept.append(line)
    destination.write_text("\n".join(kept) + "\n", encoding="utf-8")
    logger.info("Excluded packages: %s", sorted(skipped))
    return skipped


def installed_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--dossr-repo-url", default=DEFAULT_DOSSR_URL)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--allow-preexisting-conflicts",
        action="store_true",
        help=(
            "Kaggle mode: report baseline pip-check conflicts and continue only if "
            "the DoSSR dry-run passes and preserves torch/torchvision"
        ),
    )
    args = parser.parse_args()

    logger, log_path = configure_logging("colab_setup", args.log_dir)
    log_run_info(logger, args, seed=None)
    logger.info("Log file: %s", log_path)

    torch_before = installed_version("torch")
    torchvision_before = installed_version("torchvision")
    if torch_before is None:
        logger.error("PyTorch is not installed. Refusing to resolve/install DoSSR requirements.")
        return 1
    logger.info("Preserving installed torch=%s, torchvision=%s", torch_before, torchvision_before)

    requirements_path = args.requirements.expanduser().resolve()
    if not resolve_requirements(requirements_path, args.dossr_repo_url, logger):
        return 1

    pip_check = run_command([sys.executable, "-m", "pip", "check"], logger)
    current_conflicts = (pip_check.stdout or "") + (pip_check.stderr or "")
    baseline_conflicts = {
        line.strip() for line in current_conflicts.splitlines() if line.strip()
    }
    if pip_check.returncode != 0:
        level = "warning" if args.allow_preexisting_conflicts else "error"
        getattr(logger, level)(
            "Existing environment dependency conflicts found%s:",
            "; continuing with Kaggle preflight" if args.allow_preexisting_conflicts else "; stopping before installation",
        )
        log_output(logger, current_conflicts, level)
        if not args.allow_preexisting_conflicts:
            return 1

    with tempfile.TemporaryDirectory(prefix="sc203-colab-setup-") as temp_name:
        temp_dir = Path(temp_name)
        filtered_path = temp_dir / "requirements.txt"
        constraints_path = temp_dir / "constraints.txt"
        report_path = temp_dir / "pip-report.json"
        filtered_requirements(requirements_path, filtered_path, logger)

        constraints = [f"torch=={torch_before}"]
        if torchvision_before is not None:
            constraints.append(f"torchvision=={torchvision_before}")
        constraints_path.write_text("\n".join(constraints) + "\n", encoding="utf-8")

        dry_run = run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--dry-run",
                "--report",
                str(report_path),
                "--constraint",
                str(constraints_path),
                "--requirement",
                str(filtered_path),
                "--disable-pip-version-check",
            ],
            logger,
        )
        dry_output = (dry_run.stdout or "") + (dry_run.stderr or "")
        if dry_run.returncode != 0:
            logger.error("Dependency resolver found a conflict; stopping without installing:")
            log_output(logger, dry_output, "error")
            return 1

        if not report_path.is_file():
            logger.error("pip did not produce a dry-run report; stopping without installing.")
            return 1
        report = json.loads(report_path.read_text(encoding="utf-8"))
        planned = report.get("install", [])
        protected = {"torch", "torchvision"}
        changing_protected = [
            item.get("metadata", {}).get("name", "unknown")
            for item in planned
            if canonical_name(item.get("metadata", {}).get("name", "")) in protected
        ]
        if changing_protected:
            logger.error(
                "Resolver proposes changing protected package(s) %s; stopping without installing.",
                changing_protected,
            )
            return 1

        logger.info("Dependency dry run passed; planned package changes follow.")
        log_output(logger, dry_output)
        install = run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-input",
                "--constraint",
                str(constraints_path),
                "--requirement",
                str(filtered_path),
                "--disable-pip-version-check",
            ],
            logger,
        )
        install_output = (install.stdout or "") + (install.stderr or "")
        log_output(logger, install_output, "info" if install.returncode == 0 else "error")
        if install.returncode != 0:
            logger.error("Package installation failed; see the log for pip diagnostics.")
            return 1

    torch_after = installed_version("torch")
    torchvision_after = installed_version("torchvision")
    if torch_after != torch_before or torchvision_after != torchvision_before:
        logger.error(
            "Protected package version changed unexpectedly: torch %s -> %s; torchvision %s -> %s",
            torch_before,
            torch_after,
            torchvision_before,
            torchvision_after,
        )
        return 1

    final_check = run_command([sys.executable, "-m", "pip", "check"], logger)
    final_output = (final_check.stdout or "") + (final_check.stderr or "")
    if final_check.returncode != 0:
        final_conflicts = {
            line.strip() for line in final_output.splitlines() if line.strip()
        }
        new_conflicts = final_conflicts - baseline_conflicts
        log_output(logger, final_output, "error")
        if new_conflicts:
            logger.error("New dependency conflicts appeared after setup:")
            log_output(logger, "\n".join(sorted(new_conflicts)), "error")
            return 1
        logger.warning("Only the previously reported environment conflicts remain.")

    logger.info("Setup complete. torch=%s, torchvision=%s unchanged.", torch_after, torchvision_after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
