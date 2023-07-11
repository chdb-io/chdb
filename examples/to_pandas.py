#!python3
import os
import pyarrow as pa
import chdb

# get current file dir
current_dir = os.path.dirname(os.path.abspath(__file__))
test_parquet = current_dir + "/../tests/data/alltypes_dictionary.parquet"

# run SQL on parquet file and return arrow format
res = chdb.query(f"select * from file('{test_parquet}', Parquet)", "Arrow")
print("\nresult from chdb:")
print(res.bytes())

def to_arrowTable(res):
    # convert arrow format to arrow table
    paTable = pa.RecordBatchFileReader(res.bytes()).read_all()
    return paTable

def to_df(res):
    # convert arrow format to arrow table
    paTable = to_arrowTable(res)
    # convert arrow table to pandas dataframe
    return paTable.to_pandas(use_threads=True)

print("\nresult from chdb to pyarrow:")
print(to_arrowTable(res))

# convert arrow table to pandas dataframe
print("\nresult from chdb to pandas:")
print(to_df(res))
