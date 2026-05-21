"""
Real-user-path regression tests for chdb-ds.

This subpackage exists *exclusively* to hold tests that mirror chains of
operations a real user actually wrote (Slack reports, GitHub issues,
StackOverflow questions, notebook transcripts). Every file here should
follow these rules:

1. **Verbatim mirror**: the test code should look as close as possible to
   what the user wrote, including any "wrong" or sub-optimal patterns.
   Resist the temptation to "clean it up" - the user wrote it that way
   for a reason and that reason is the test case.

2. **Mirror pandas**: each user path is run on both ``pandas`` and
   ``DataStore`` with the same code, and outputs are compared via
   :func:`datastore.tests.test_utils.assert_datastore_equals_pandas`.

3. **Long chains are the point**: most chains here are 5+ operations.
   Single-op behaviour is covered by the feature-oriented test files
   under ``datastore/tests/``; this subpackage focuses on the
   compositional space those tests do not reach.

4. **One file per user / scenario** when feasible, so that a future
   git-blame on a failing test points back to the original report.

Adding a new file is encouraged the moment a user-reported bug lands -
even before the fix. ``xfail`` markers are fine while the bug is open;
removing them is the definition of "fixed".
"""
