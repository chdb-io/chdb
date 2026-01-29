#!/usr/bin/env python3
"""
Run pytest with JUnit XML output and determine success based on XML results.

This script handles the case where chDB's cleanup causes SIGABRT after tests
complete. It uses JUnit XML output to reliably determine if tests passed,
rather than relying on pytest's exit code (which may be corrupted by SIGABRT).

Exit codes:
  0 - All tests passed
  1 - Some tests failed
  2 - Script/setup error
"""
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def check_junit_xml(xml_path: str) -> tuple:
    """
    Parse pytest JUnit XML output to determine test success/failure.

    Returns:
        (success, message) tuple
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0

        for testsuite in root.iter('testsuite'):
            total_tests += int(testsuite.get('tests', 0))
            total_failures += int(testsuite.get('failures', 0))
            total_errors += int(testsuite.get('errors', 0))
            total_skipped += int(testsuite.get('skipped', 0))

        passed = total_tests - total_failures - total_errors - total_skipped

        if total_failures > 0 or total_errors > 0:
            msg = f"FAIL: {total_failures} failures, {total_errors} errors out of {total_tests} tests"
            return False, msg

        msg = f"PASS: {passed} passed, {total_skipped} skipped out of {total_tests} tests"
        return True, msg

    except FileNotFoundError:
        return False, f"ERROR: XML file not found: {xml_path}"
    except ET.ParseError as e:
        return False, f"ERROR: Failed to parse XML: {e}"


def main():
    # Get the datastore directory
    script_dir = Path(__file__).parent
    datastore_dir = script_dir.parent
    tests_dir = datastore_dir / "tests"

    if not tests_dir.exists():
        print(f"ERROR: Tests directory not found: {tests_dir}")
        return 2

    # Use a temp file for JUnit XML output
    with tempfile.NamedTemporaryFile(suffix='.xml', delete=False) as f:
        xml_path = f.name

    try:
        # Build pytest command with additional args from command line
        pytest_args = [
            sys.executable, '-m', 'pytest',
            str(tests_dir),
            '-v', '--tb=short',
            f'--junit-xml={xml_path}'
        ]
        # Add any extra args passed to this script
        pytest_args.extend(sys.argv[1:])

        # Run pytest - we don't care about its exit code due to SIGABRT issue
        # The subprocess will capture all output
        result = subprocess.run(
            pytest_args,
            cwd=str(datastore_dir),
        )

        # Check the XML results - this is the source of truth
        success, message = check_junit_xml(xml_path)
        print(f"\n{'='*60}")
        print(f"Test Result: {message}")
        print(f"{'='*60}")

        if result.returncode != 0 and success:
            print(f"Note: pytest exited with code {result.returncode} (likely SIGABRT during cleanup)")
            print("This is a known chDB library issue - tests themselves passed.")

        return 0 if success else 1

    finally:
        # Clean up temp file
        try:
            os.unlink(xml_path)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
