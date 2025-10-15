#!python3

import unittest
import os
import glob

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'

def check_optional_dependencies():
    has_pyarrow = False
    has_pandas = False

    try:
        import pyarrow  # type: ignore
        has_pyarrow = True
        print(f"{Colors.GREEN}PyArrow {pyarrow.__version__} is available{Colors.END}")
    except ImportError:
        print(f"{Colors.YELLOW}PyArrow not installed{Colors.END}")

    try:
        import pandas  # type: ignore
        has_pandas = True
        print(f"{Colors.GREEN}Pandas {pandas.__version__} is available{Colors.END}")
    except ImportError:
        print(f"{Colors.YELLOW}Pandas not installed{Colors.END}")

    return has_pyarrow and has_pandas

def main():
    has_pyarrow_and_pandas = check_optional_dependencies()

    BASIC_TEST_FILES = [
        'test_basic.py',
        'test_command_line.py',
        'test_conn_cursor.py',
        'test_dbapi_persistence.py',
        'test_dbapi.py',
        'test_delta_lake.py',
        'test_drop_table.py',
        'test_early_gc.py',
        'test_final_join.py',
        'test_gc.py',
        'test_insert_error_handling.py',
        'test_insert_vector.py',
        'test_issue104.py',
        'test_issue135.py',
        'test_issue229.py',
        'test_issue31.py',
        'test_issue60.py',
        'test_materialize.py',
        'test_multiple_query.py',
        'test_open_session_after_failure.py',
        'test_optional_dependencies.py',
        'test_signal_handler.py',
        'test_statistics.py',
        'test_streaming_query.py',
        'test_udf.py',
        'test_usedb.py',
        'test_query_json.py',
    ]

    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()

    if has_pyarrow_and_pandas:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All dependencies available - running all tests{Colors.END}")
        all_test_files = glob.glob('test_*.py')
        test_files_to_run = [f for f in all_test_files]
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}Some dependencies missing - running basic tests only{Colors.END}")
        test_files_to_run = BASIC_TEST_FILES.copy()

    print(f"\n{Colors.GREEN}Running test files: {', '.join(test_files_to_run)}{Colors.END}\n")

    for test_file in test_files_to_run:
        if not test_file.endswith('.py'):
            test_file += '.py'
        if os.path.exists(test_file):
            module_name = test_file[:-3].replace('/', '.')
            try:
                suite = test_loader.loadTestsFromName(module_name)
                test_suite.addTest(suite)
                print(f"{Colors.GREEN}Loaded {test_file}{Colors.END}")
            except Exception as e:
                print(f"{Colors.YELLOW}Warning: Could not load {test_file}: {e}{Colors.END}")
        else:
            print(f"{Colors.RED}Error: Test file {test_file} not found{Colors.END}")

    run_tests(test_suite)

def run_tests(test_suite):
    test_runner = unittest.TextTestRunner(verbosity=2)
    ret = test_runner.run(test_suite)

    total = ret.testsRun
    failures = len(ret.failures)
    errors = len(ret.errors)
    success = total - failures - errors

    if failures + errors == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED{Colors.END}")
        print(f"{Colors.GREEN}Success: {success}, Total: {total}{Colors.END}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✖ TEST FAILURES{Colors.END}")
        print(f"{Colors.RED}Failed: {failures}, Errors: {errors}, Success: {success}, Total: {total}{Colors.END}")

        if failures > 0:
            print(f"\n{Colors.YELLOW}Failed Tests:{Colors.END}")
            for failure in ret.failures:
                test_case, traceback = failure
                print(f"{Colors.RED}• {test_case.id()}{Colors.END}")

        if errors > 0:
            print(f"\n{Colors.YELLOW}Errored Tests:{Colors.END}")
            for error in ret.errors:
                test_case, traceback = error
                print(f"{Colors.RED}• {test_case.id()}{Colors.END}")

    # if any test fails, exit with non-zero code
    if len(ret.failures) > 0 or len(ret.errors) > 0:
        exit(1)
    else:
        exit(0)

if __name__ == '__main__':
    main()
