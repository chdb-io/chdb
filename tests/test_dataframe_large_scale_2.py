#!/usr/bin/env python3

import os
import unittest
import time
from urllib.request import urlretrieve
import pandas as pd
import chdb
import json
import numpy as np
from datetime import timedelta


class TestDataFrameLargeScale(unittest.TestCase):
    """Test DataFrame generation with large scale data (1M rows) and diverse data types"""

    @classmethod
    def setUpClass(cls):
        cls.parquet_file = "hits_0.parquet"
        if not os.path.exists(cls.parquet_file):
            print(f"Downloading {cls.parquet_file}...")
            url = "https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_0.parquet"
            urlretrieve(url, cls.parquet_file)
            print("Download complete!")

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.session = chdb.session.Session("./tmp")

    def tearDown(self):
        self.session.close()

    def test_pandas_chdb_dataframe_consistency(self):
        """Compare pandas and chdb DataFrame consistency when reading the same parquet file"""

        print(f"Testing DataFrame consistency between pandas and chdb for {self.parquet_file}")

        # 1. Read with pandas
        print("Reading parquet file with pandas...")
        pandas_start = time.time()
        df_pandas = pd.read_parquet(self.parquet_file)
        pandas_time = time.time() - pandas_start
        print(f"Pandas read time: {pandas_time:.3f} seconds")

        # 2. Read with chdb
        print("Reading parquet file with chdb...")
        chdb_start = time.time()
        df_chdb = self.session.query(f"""
            SELECT * FROM file('{self.parquet_file}')
        """, "DataFrame")
        chdb_time = time.time() - chdb_start
        print(f"chdb read time: {chdb_time:.3f} seconds")

        # 3. Compare basic structure
        print("\n=== Structure Comparison ===")
        pandas_rows, pandas_cols = df_pandas.shape
        chdb_rows, chdb_cols = df_chdb.shape

        print(f"Pandas DataFrame: {pandas_rows:,} rows × {pandas_cols} columns")
        print(f"chdb DataFrame:   {chdb_rows:,} rows × {chdb_cols} columns")

        # Assert row and column counts match
        self.assertEqual(pandas_rows, chdb_rows, f"Row count mismatch: pandas={pandas_rows}, chdb={chdb_rows}")
        self.assertEqual(pandas_cols, chdb_cols, f"Column count mismatch: pandas={pandas_cols}, chdb={chdb_cols}")

        # 4. Compare column names
        print("\n=== Column Names Comparison ===")
        pandas_columns = set(df_pandas.columns)
        chdb_columns = set(df_chdb.columns)

        missing_in_chdb = pandas_columns - chdb_columns
        missing_in_pandas = chdb_columns - pandas_columns

        if missing_in_chdb:
            print(f"Columns missing in chdb: {missing_in_chdb}")
        if missing_in_pandas:
            print(f"Columns missing in pandas: {missing_in_pandas}")

        # Assert column names match
        self.assertEqual(pandas_columns, chdb_columns, "Column names don't match between pandas and chdb")
        print("✓ Column names match")

        # 5. Compare data types
        print("\n=== Data Types Comparison ===")

        common_columns = list(pandas_columns.intersection(chdb_columns))
        self.assertEqual(len(common_columns), len(pandas_columns), "Column names don't match between pandas and chdb")

        print(f"Comparing data types for {len(common_columns)} columns:")
        print("-" * 80)

        dtype_mismatches = []
        for col in common_columns:
            pandas_dtype = str(df_pandas[col].dtype)
            chdb_dtype = str(df_chdb[col].dtype)

            # Print each column's data types
            match_status = "✓" if pandas_dtype == chdb_dtype else "✗"
            print(f"{match_status} {col:<20} | pandas: {pandas_dtype:<20} | chdb: {chdb_dtype:<20}")

            if pandas_dtype != chdb_dtype:
                dtype_mismatches.append({
                    'column': col,
                    'pandas': pandas_dtype,
                    'chdb': chdb_dtype
                })

        print("-" * 80)

        if dtype_mismatches:
            print(f"\nData type differences found in {len(dtype_mismatches)} columns:")
            for mismatch in dtype_mismatches:
                print(f"  ✗ {mismatch['column']}: pandas={mismatch['pandas']} vs chdb={mismatch['chdb']}")
            self.fail("Data type differences found between pandas and chdb")
        else:
            print("\n✓ All data types match perfectly!")

        # 6. Compare data values every 1000 rows
        print("\n=== Data Values Comparison (every 1000 rows) ===")

        # Sort both DataFrames by WatchID (unique identifier) to ensure consistent ordering
        sort_col = 'WatchID' if 'WatchID' in common_columns else common_columns[0]
        print(f"Sorting by column: {sort_col}")
        df_pandas = df_pandas.sort_values(by=sort_col).reset_index(drop=True)
        df_chdb = df_chdb.sort_values(by=sort_col).reset_index(drop=True)
        self.assertEqual(len(df_pandas), len(df_chdb))

        total_rows = len(df_pandas)
        sample_interval = 1000
        sample_indices = list(range(0, total_rows, sample_interval))

        # Add the last row if it's not already included
        if total_rows - 1 not in sample_indices:
            sample_indices.append(total_rows - 1)

        print(f"Comparing {len(sample_indices)} sample rows (every {sample_interval} rows)")

        data_mismatches = []
        for idx in sample_indices:
            row_mismatches = []

            for col in common_columns:
                pandas_val = df_pandas.iloc[idx][col]
                chdb_val = df_chdb.iloc[idx][col]

                # Handle different ways of representing None/NaN
                pandas_is_na = pd.isna(pandas_val)
                chdb_is_na = pd.isna(chdb_val)

                if pandas_is_na and chdb_is_na:
                    continue  # Both are NaN, considered equal
                elif pandas_is_na != chdb_is_na:
                    row_mismatches.append({
                        'column': col,
                        'pandas': pandas_val,
                        'chdb': chdb_val,
                        'reason': 'null_mismatch'
                    })
                elif pandas_val != chdb_val:
                    # Handle bytes vs string comparison
                    if isinstance(pandas_val, bytes) and isinstance(chdb_val, str):
                        # Convert string to bytes for comparison (preserve original binary data)
                        try:
                            chdb_bytes = chdb_val.encode('utf-8')
                            if pandas_val == chdb_bytes:
                                continue  # Values are equivalent after conversion
                        except:
                            print(f"Failed to encode string to bytes: {chdb_val}")
                            self.fail("Failed to encode string to bytes")
                    elif isinstance(pandas_val, str) and isinstance(chdb_val, bytes):
                        # Convert string to bytes for comparison (preserve original binary data)
                        try:
                            pandas_bytes = pandas_val.encode('utf-8')
                            if pandas_bytes == chdb_val:
                                continue  # Values are equivalent after conversion
                        except:
                            print(f"Failed to encode string to bytes: {pandas_val}")
                            self.fail("Failed to encode string to bytes")

                    # For floating point numbers, use approximate comparison
                    if isinstance(pandas_val, (float, int)) and isinstance(chdb_val, (float, int)):
                        if abs(float(pandas_val) - float(chdb_val)) > 1e-10:
                            row_mismatches.append({
                                'column': col,
                                'pandas': pandas_val,
                                'chdb': chdb_val,
                                'reason': 'value_mismatch'
                            })
                    else:
                        # Check if this is a bytes vs string type issue
                        is_bytes_string_mismatch = (
                            (isinstance(pandas_val, bytes) and isinstance(chdb_val, str)) or
                            (isinstance(pandas_val, str) and isinstance(chdb_val, bytes))
                        )

                        row_mismatches.append({
                            'column': col,
                            'pandas': pandas_val,
                            'chdb': chdb_val,
                            'reason': 'bytes_string_mismatch' if is_bytes_string_mismatch else 'value_mismatch'
                        })

            if row_mismatches:
                data_mismatches.append({
                    'row_index': idx,
                    'mismatches': row_mismatches
                })

        # 7. Report results
        print(f"\n=== Summary ===")
        print(f"✓ Row count: {pandas_rows:,}")
        print(f"✓ Column count: {pandas_cols}")
        print(f"✓ Sample rows checked: {len(sample_indices)}")

        if data_mismatches:
            print(f"Data mismatches found in {len(data_mismatches)} rows")

            # Show first few mismatches for debugging
            for i, mismatch in enumerate(data_mismatches[:3]):
                print(f"\nRow {mismatch['row_index']} mismatches:")
                for col_mismatch in mismatch['mismatches'][:5]:  # Show first 5 column mismatches
                    print(f"  {col_mismatch['column']}: pandas='{col_mismatch['pandas']}' vs chdb='{col_mismatch['chdb']}'")

            if len(data_mismatches) > 3:
                print(f"... and {len(data_mismatches) - 3} more rows with mismatches")

            self.fail(f"Data mismatches found in {len(data_mismatches)} rows")
        else:
            print("All sampled data values match")

        print("\nDataFrame consistency test completed!")


if __name__ == '__main__':
    unittest.main()
