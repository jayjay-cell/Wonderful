"""Architectural guard tests.

These do not test behaviour -- they test that the structure the design
depends on has not quietly eroded. Without them, the layering degrades
within weeks: someone imports httpx into core/ for one quick fix, and the
property that made the system testable and air-gappable is gone with no
visible symptom until it matters.

Two invariants:
  1. core/ is pure -- no LLM framework, no HTTP client, no I/O (NFR-1).
  2. No cloud SDK is imported outside providers/cloud/ (FR-F4).

Implemented by parsing imports from source rather than by importing the
modules, so a forbidden dependency is detected even if it is not installed
on this machine.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Anything that implies a framework, a network, or an external service.
FORBIDDEN_IN_CORE = {
    "langchain", "langchain_core", "langchain_anthropic", "langgraph",
    "anthropic", "openai", "httpx", "requests", "aiohttp", "urllib",
    "fastapi", "uvicorn", "sqlite3", "aiosqlite",
}

# core/ legitimately reads mission files, so yaml and pathlib are allowed
# there; everything else above is not.
ALLOWED_IN_CORE = {"yaml", "pathlib"}

CLOUD_SDKS = {
    "anthropic", "langchain_anthropic", "openai", "elevenlabs",
    "google", "google_genai", "langchain_google_genai", "groq", "langchain_groq",
}


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as err:  # pragma: no cover - a syntax error fails elsewhere
        pytest.fail(f"{path} has a syntax error: {err}")

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have no module to inspect.
            if node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _python_files(directory: str) -> list[Path]:
    """Every .py file under a package, for scanning imports."""
    root = PROJECT_ROOT / directory
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _is_type_checking_only(path: Path, module: str) -> bool:
    """Whether a module is imported only inside `if TYPE_CHECKING:`.

    Such an import never executes at runtime, so it does not create a real
    dependency -- providers/base.py uses this to reference a langchain type
    while staying importable in an air-gapped install with langchain absent.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded: set[str] = set()
    for node in ast.walk(tree):
        is_type_checking_guard = (
            isinstance(node, ast.If)
            and (
                (isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING")
                or (isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING")
            )
        )
        if not is_type_checking_guard:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Import):
                guarded.update(a.name.split(".")[0] for a in inner.names)
            elif isinstance(inner, ast.ImportFrom) and inner.module:
                guarded.add(inner.module.split(".")[0])
    return module in guarded


class TestCoreIsPure:
    """core/ must stay free of frameworks, networking and I/O (NFR-1).

    This is what makes the whole engine unit-testable with no API key and
    no network, and what lets a 30-minute session be tested in
    milliseconds.
    """

    def test_core_imports_no_framework_or_network(self) -> None:
        """core/ must stay pure: no FastAPI, no HTTP client, no provider SDK."""
        violations: list[str] = []
        for path in _python_files("core"):
            for module in _imported_modules(path) & FORBIDDEN_IN_CORE:
                if module in ALLOWED_IN_CORE:
                    continue
                if _is_type_checking_only(path, module):
                    continue
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module!r}")

        assert not violations, (
            "core/ must contain no LLM framework, HTTP client or I/O dependency.\n"
            "Found:\n  " + "\n  ".join(violations) + "\n\n"
            "Move the dependency outward: core/ holds pure logic, and tools/ or "
            "providers/ is where anything external belongs. See NFR-1."
        )

    def test_core_files_exist(self) -> None:
        """Guards against the test passing vacuously because core/ moved."""
        assert _python_files("core"), "no Python files found in core/ -- has it moved?"


class TestNoCloudSdkOutsideCloudProviders:
    """A cloud SDK may only be imported inside providers/cloud/ (FR-F4).

    This is the mechanism that makes air-gap readiness verifiable before
    deployment rather than discovered on arrival.
    """

    def test_cloud_sdks_confined(self) -> None:
        """Cloud SDKs may only be imported inside providers/cloud/."""
        violations: list[str] = []
        for directory in ("core", "agent", "tools", "delivery", "sim", "api", "obs"):
            for path in _python_files(directory):
                for module in _imported_modules(path) & CLOUD_SDKS:
                    if _is_type_checking_only(path, module):
                        continue
                    violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module!r}")

        for path in _python_files("providers"):
            if "cloud" in path.parts:
                continue
            for module in _imported_modules(path) & CLOUD_SDKS:
                if _is_type_checking_only(path, module):
                    continue
                violations.append(f"{path.relative_to(PROJECT_ROOT)} imports {module!r}")

        assert not violations, (
            "Cloud SDKs may only be imported inside providers/cloud/.\n"
            "Found:\n  " + "\n  ".join(violations) + "\n\n"
            "The deployment target has no internet. Reach external services "
            "through the protocols in providers/base.py instead. See FR-F4."
        )

    def test_cloud_provider_directory_is_the_only_exception(self) -> None:
        """Documents the intended exception, so the rule above is not
        mistaken for 'no cloud SDKs anywhere'."""
        assert (PROJECT_ROOT / "providers" / "cloud").exists()
