#!python3

import unittest

# 使用 TestLoader() 加载所有测试模块
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('./')

# 使用 TextTestRunner() 运行测试套件
test_runner = unittest.TextTestRunner()
test_runner.run(test_suite)