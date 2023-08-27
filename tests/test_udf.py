#!python3

from chdb.udf import *

@to_clickhouse_udf()
def sum_udf(lhs, rhs):
    import time
    time.sleep(1)
    return int(lhs) + int(rhs)

@to_clickhouse_udf(return_type="Int32")
def mul_udf(lhs, rhs):
    return int(lhs) * int(rhs)
