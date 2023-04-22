#!python3

import unittest

test_loader = unittest.TestLoader()
test_suite = test_loader.discover('./')

test_runner = unittest.TextTestRunner()
ret = test_runner.run(test_suite)

# if any test fails, exit with non-zero code
if len(ret.failures) > 0 or len(ret.errors) > 0:
    exit(1)
else:
    exit(0)
