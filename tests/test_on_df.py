import atexit
import io
import os.path
import time
import unittest

from chdb.dataframe.query import Table
from utils import current_dir

# 122MB parquet file
hits_0 = os.path.join(current_dir, "hits_0.parquet")
sql = """SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID)
                        FROM __table__ GROUP BY RegionID ORDER BY c DESC LIMIT 10"""

expected = """   RegionID  sum(AdvEngineID)       c  avg(ResolutionWidth)  uniqExact(UserID)
0       229             38044  426435           1612.787187              27961
1         2             12801  148193           1593.870891              10413
2       208              2673   30614           1490.615111               3073
3         1              1802   28577           1623.851699               1720
4        34               508   14329           1592.897201               1428
5        47              1041   13661           1637.851914                943
6       158                78   13294           1576.340605               1110
7         7              1166   11679           1627.319034                647
8        42               642   11547           1625.601022                956
9       184                30   10157           1614.693807                987"""

output = io.StringIO()
# run print at exit
atexit.register(lambda: print("\n" + output.getvalue()))


class TestRunOnDf(unittest.TestCase):
    def test_run_parquet(self):
        pq_table = Table(parquet_path=hits_0)
        t = time.time()
        ret = pq_table.query(sql)
        print("Run on parquet file. Time cost:", time.time() - t, "s", file=output)
        self.assertEqual(expected, str(ret))

    def test_run_parquet_buf(self):
        pq_table = Table(parquet_memoryview=memoryview(open(hits_0, 'rb').read()))
        t = time.time()
        ret = pq_table.query(sql)
        print("Run on parquet buffer. Time cost:", time.time() - t, "s", file=output)
        self.assertEqual(expected, str(ret))

    def test_run_arrow_table(self):
        import pyarrow.parquet as pq
        arrow_table = pq.read_table(hits_0)
        pq_table = Table(arrow_table=arrow_table)
        t = time.time()
        ret = pq_table.query(sql)
        print("Run on arrow table. Time cost:", time.time() - t, "s", file=output)
        self.assertEqual(expected, str(ret))

    def test_run_df(self):
        import pandas as pd
        df = pd.read_parquet(hits_0)
        pq_table = Table(dataframe=df)
        t = time.time()
        ret = pq_table.query(sql)
        print("Run on dataframe. Time cost:", time.time() - t, "s", file=output)
        self.assertEqual(expected, str(ret))


if __name__ == '__main__':
    unittest.main()
