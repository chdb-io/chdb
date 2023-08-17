#!python3

import unittest
import pandas as pd
from chdb import dataframe as cdf


class TestJoinDf(unittest.TestCase):
    def test_1df(self):
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': [b"one", b"two", b"three"]})
        cdf1 = cdf.Table(dataframe=df1)
        ret1 = cdf.query(sql="select * from __tbl1__", tbl1=cdf1)
        self.assertEqual(str(ret1), str(df1))

    def test_2df(self):
        df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ["one", "two", "three"]})
        df2 = pd.DataFrame({'c': [1, 2, 3], 'd': ["①", "②", "③"]})
        ret1 = cdf.query(sql="select * from __tbl1__ t1, __tbl2__ t2 where t1.a = t2.c",
                         tbl1=cdf.Table(dataframe=df1), tbl2=cdf.Table(dataframe=df2))
        self.assertEqual(str(ret1), str(pd.DataFrame({
            'a': [1, 2, 3], 'b': [b"one", b"two", b"three"],
            'c': [1, 2, 3], 'd': [b'\xe2\x91\xa0', b'\xe2\x91\xa1', b'\xe2\x91\xa2']})))


if __name__ == '__main__':
    unittest.main()
