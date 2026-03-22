"""
Test runner with programmatic execution and custom PASS/FAIL summary.

This module provides:
- Programmatic execution of all tests
- Custom PASS/FAIL summary with detailed failure diagnostics
- Exit codes (0 for success, 1 for failure)
"""

import logging
import sys
import traceback
from dataclasses import dataclass, field
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pytest

logger = logging.getLogger(__name__)


# =============================================================================
# TEST RESULT DATA STRUCTURES
# =============================================================================

@dataclass
class FailureInfo:
    """Represents a single test failure with details."""
    test_name: str
    message: str
    context: Dict[str, str] = field(default_factory=dict)
    traceback: Optional[str] = None

    def __str__(self) -> str:
        lines = [
            f"FAILED: {self.test_name}",
            "",
            "Reason:",
            self.message,
        ]
        if self.context:
            lines.append("")
            lines.append("Context:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
        if self.traceback:
            lines.append("")
            lines.append("Traceback:")
            lines.append(self.traceback)
        return "\n".join(lines)


@dataclass
class SummaryInfo:
    """Summary of test execution results."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    errors: int = 0
    failures: List[FailureInfo] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed_tests == 0 and self.errors == 0

    def add_failure(
        self,
        test_name: str,
        message: str,
        context: Optional[Dict[str, str]] = None,
        traceback: Optional[str] = None,
    ):
        self.failures.append(FailureInfo(test_name, message, context or {}, traceback))


# =============================================================================
# TEST EXECUTION
# =============================================================================

def run_tests(
    verbose: bool = True,
    capture_logs: bool = True,
    test_paths: Optional[List[str]] = None,
) -> SummaryInfo:
    """
    Run all tests programmatically.

    Parameters
    ----------
    verbose : bool
        Enable verbose output.
    capture_logs : bool
        Capture and display log output.
    test_paths : list of str, optional
        Specific test paths to run. If None, runs all tests.

    Returns
    -------
    SummaryInfo
        Summary of test execution results.
    """
    if test_paths is None:
        test_paths = [
            "tests/test_abr_algorithm.py",
            "tests/test_streamer_unit.py",
            "tests/test_streamer_integration.py",
            "tests/test_invariants.py",
        ]

    # Create summary
    summary = SummaryInfo()

    # Capture stdout/stderr
    stdout_capture = StringIO()
    stderr_capture = StringIO()

    # Pytest arguments
    pytest_args = [
        "-v" if verbose else "-q",
        "--tb=short",
        "--color=yes",
    ]

    if capture_logs:
        pytest_args.append("--log-cli-level=INFO")

    # Add test paths
    pytest_args.extend(test_paths)

    # Run pytest with custom plugin to capture results
    class _ResultsPlugin:
        def __init__(self):
            self.summary = summary

        def pytest_runtest_logreport(self, report):
            if report.when == "call":
                if report.passed:
                    self.summary.passed_tests += 1
                elif report.failed:
                    self.summary.failed_tests += 1
                    # Extract failure info
                    self._extract_failure(report)
                elif report.skipped:
                    self.summary.skipped_tests += 1

        def _extract_failure(self, report):
            """Extract failure details from pytest report."""
            test_name = report.nodeid

            # Try to extract assertion message
            message = str(report.longrepr) if report.longrepr else "Test failed"

            # Try to extract context from test report
            context = {}
            if hasattr(report, "longrepr"):
                longrepr = report.longrepr
                if hasattr(longrepr, "reprcrash"):
                    crash = longrepr.reprcrash
                    if hasattr(crash, "message"):
                        message = crash.message
                    if hasattr(crash, "lineno"):
                        context["line"] = str(crash.lineno)
                if hasattr(longrepr, "reprtraceback"):
                    tb = longrepr.reprtraceback
                    if hasattr(tb, "reprentries"):
                        # Extract key info from traceback
                        context["traceback_entries"] = str(len(tb.reprentries))

            # Get short traceback
            tb_str = None
            if report.longreprtb:
                if isinstance(report.longreprtb, str):
                    tb_str = report.longreprtb
                elif hasattr(report.longreprtb, "longrepr"):
                    tb_str = str(report.longreprtb.longrepr)

            self.summary.add_failure(
                test_name=test_name,
                message=message,
                context=context,
                traceback=tb_str,
            )

    # Register plugin
    plugin = _ResultsPlugin()
    pytest.main(pytest_args, plugins=[plugin])

    return summary


def run_tests_simple() -> Tuple[int, str]:
    """
    Simple test runner that returns exit code and output.

    Returns
    -------
    tuple
        (exit_code, output_string)
    """
    output = StringIO()
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        sys.stdout = output
        sys.stderr = output

        # Run pytest with JSON report
        import tempfile
        import json

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_report = f.name

        pytest_args = [
            "-v",
            "--tb=short",
            f"--json-report",
            f"--json-report-file={json_report}",
            "tests/",
        ]

        try:
            exit_code = pytest.main(pytest_args)
        except SystemExit as e:
            exit_code = e.code

        # Read JSON report if available
        try:
            with open(json_report, 'r') as f:
                report_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            report_data = None

        return exit_code, output.getvalue()

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# =============================================================================
# SUMMARY REPORTING
# =============================================================================

def print_summary(summary: SummaryInfo) -> None:
    """
    Print formatted test summary.

    Parameters
    ----------
    summary : SummaryInfo
        Test execution summary.
    """
    print("=" * 60)
    if summary.all_passed:
        print("ALL TESTS PASSED")
        print("=" * 60)
        print(f"Total:   {summary.total_tests}")
        print(f"Passed:  {summary.passed_tests}")
        print(f"Skipped: {summary.skipped_tests}")
    else:
        print("TEST FAILURES DETECTED")
        print("=" * 60)
        print(f"Total:   {summary.total_tests}")
        print(f"Passed:  {summary.passed_tests}")
        print(f"Failed:  {summary.failed_tests}")
        print(f"Skipped: {summary.skipped_tests}")
        print()
        print("=" * 60)
        print("FAILURE DETAILS")
        print("=" * 60)

        for i, failure in enumerate(summary.failures, 1):
            print()
            print(f"[{i}] {failure.test_name}")
            print("-" * 40)
            print(f"Reason:")
            print(f"  {failure.message}")
            if failure.context:
                print(f"Context:")
                for key, value in failure.context.items():
                    print(f"  {key}: {value}")
            if failure.traceback:
                print(f"Traceback:")
                for line in failure.traceback.split('\n'):
                    print(f"  {line}")


def generate_report(summary: SummaryInfo) -> str:
    """
    Generate a detailed failure report.

    Parameters
    ----------
    summary : SummaryInfo
        Test execution summary.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []

    if summary.all_passed:
        lines.append("====================")
        lines.append("ALL TESTS PASSED")
        lines.append("====================")
        lines.append(f"Total:   {summary.total_tests}")
        lines.append(f"Passed:  {summary.passed_tests}")
        lines.append(f"Skipped: {summary.skipped_tests}")
    else:
        lines.append("====================")
        lines.append("TEST FAILURES DETECTED")
        lines.append("====================")
        lines.append(f"Total:   {summary.total_tests}")
        lines.append(f"Passed:  {summary.passed_tests}")
        lines.append(f"Failed:  {summary.failed_tests}")
        lines.append(f"Skipped: {summary.skipped_tests}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("FAILURE DETAILS")
        lines.append("=" * 60)

        for i, failure in enumerate(summary.failures, 1):
            lines.append("")
            lines.append(f"FAILED: {failure.test_name}")
            lines.append("")
            lines.append("Reason:")
            lines.append(failure.message)
            if failure.context:
                lines.append("")
                lines.append("Context:")
                for key, value in failure.context.items():
                    lines.append(f"  {key}: {value}")
            if failure.traceback:
                lines.append("")
                lines.append("Traceback:")
                lines.append(failure.traceback)

    return "\n".join(lines)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for running tests."""
    import argparse

    parser = argparse.ArgumentParser(description="Run ABR Streaming Test Suite")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Quiet output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Generate JSON report",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate detailed failure report",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help="Specific tests to run",
    )

    args = parser.parse_args()

    # Determine verbosity
    verbose = args.verbose and not args.quiet

    # Build pytest args
    pytest_args = []

    if verbose:
        pytest_args.append("-v")
    elif args.quiet:
        pytest_args.append("-q")

    pytest_args.extend([
        "--tb=short",
        "--color=yes",
    ])

    # Add specific tests if provided
    if args.tests:
        pytest_args.extend(args.tests)
    else:
        pytest_args.extend([
            "tests/test_abr_algorithm.py",
            "tests/test_streamer_unit.py",
            "tests/test_streamer_integration.py",
            "tests/test_invariants.py",
        ])

    # Capture output
    output = StringIO()
    stderr_output = StringIO()

    # Run tests
    exit_code = 0

    try:
        # Use pytest.main
        exit_code = pytest.main(pytest_args)
    except SystemExit as e:
        exit_code = e.code

    # Print summary
    if not verbose:
        # Print brief summary
        if exit_code == 0:
            print("=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
        else:
            print("=" * 60)
            print("TEST FAILURES DETECTED")
            print("=" * 60)
            print("Run with -v for detailed failure information")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
