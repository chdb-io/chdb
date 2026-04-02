"""
Merge/Join pandas compatibility tests.

Covers edge cases where SQL JOIN path might diverge from pandas merge:
1. Overlapping non-key columns (suffixes handling)
2. left_on / right_on with different column names
3. Different column orders between left and right
4. merge sort parameter
5. Various join types (inner, left, right, outer)
6. from_df() DataStore merge (SQL path vs pandas path)
7. File-backed DataStore merge
8. SQL UNION ALL column ordering in concat/union

All tests use Mirror Code Pattern: DataStore result == pandas result.
"""

import os
import tempfile
import pandas as pd
import pytest

from datastore import DataStore, concat as ds_concat
from tests.test_utils import assert_datastore_equals_pandas


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def left_data():
    return {"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [85, 92, 78]}


@pytest.fixture
def right_data():
    return {"user_id": [1, 2, 4], "city": ["NYC", "LA", "Chicago"], "age": [25, 30, 35]}


@pytest.fixture
def right_data_with_overlap():
    """Right table with same non-key column 'name' as left."""
    return {"user_id": [1, 2, 4], "name": ["Order-A", "Order-B", "Order-C"], "amount": [100, 200, 300]}


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ===========================================================================
# 1. Overlapping non-key columns → suffixes must work
# ===========================================================================

class TestMergeOverlappingColumns:
    """When both tables have non-key columns with the same name,
    pandas adds suffixes. SQL JOIN path must fall back to pandas."""

    def test_merge_overlapping_default_suffixes(self, left_data, right_data_with_overlap):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data_with_overlap)
        pd_result = pd_left.merge(pd_right, on="user_id")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data_with_overlap)
        ds_result = ds_left.merge(ds_right, on="user_id")

        assert "name_x" in list(pd_result.columns)
        assert "name_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_overlapping_custom_suffixes(self, left_data, right_data_with_overlap):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data_with_overlap)
        pd_result = pd_left.merge(pd_right, on="user_id", suffixes=("_left", "_right"))

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data_with_overlap)
        ds_result = ds_left.merge(ds_right, on="user_id", suffixes=("_left", "_right"))

        assert "name_left" in list(pd_result.columns)
        assert "name_right" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_overlapping_left_join(self, left_data, right_data_with_overlap):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data_with_overlap)
        pd_result = pd_left.merge(pd_right, on="user_id", how="left")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data_with_overlap)
        ds_result = ds_left.merge(ds_right, on="user_id", how="left")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_overlapping_outer_join(self, left_data, right_data_with_overlap):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data_with_overlap)
        pd_result = pd_left.merge(pd_right, on="user_id", how="outer")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data_with_overlap)
        ds_result = ds_left.merge(ds_right, on="user_id", how="outer")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_overlapping_multiple_columns(self):
        """Both 'val' and 'tag' overlap."""
        pd_left = pd.DataFrame({"key": [1, 2], "val": [10, 20], "tag": ["a", "b"]})
        pd_right = pd.DataFrame({"key": [1, 2], "val": [100, 200], "tag": ["x", "y"]})
        pd_result = pd_left.merge(pd_right, on="key")

        ds_left = DataStore({"key": [1, 2], "val": [10, 20], "tag": ["a", "b"]})
        ds_right = DataStore({"key": [1, 2], "val": [100, 200], "tag": ["x", "y"]})
        ds_result = ds_left.merge(ds_right, on="key")

        assert "val_x" in list(pd_result.columns)
        assert "val_y" in list(pd_result.columns)
        assert "tag_x" in list(pd_result.columns)
        assert "tag_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 2. No overlapping non-key columns → SQL path safe
# ===========================================================================

class TestMergeNoOverlap:
    """When tables have no overlapping non-key columns,
    the SQL JOIN path can be used safely."""

    def test_merge_no_overlap_inner(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_no_overlap_left(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id", how="left")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id", how="left")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_no_overlap_right(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id", how="right")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id", how="right")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_no_overlap_outer(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id", how="outer")

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id", how="outer")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 3. left_on / right_on with different column names
# ===========================================================================

class TestMergeLeftOnRightOn:
    """Merge with left_on/right_on where key column names differ."""

    def test_left_on_right_on_inner(self):
        pd_left = pd.DataFrame({"id": [1, 2, 3], "val_l": [10, 20, 30]})
        pd_right = pd.DataFrame({"uid": [1, 2, 4], "val_r": [100, 200, 400]})
        pd_result = pd_left.merge(pd_right, left_on="id", right_on="uid")

        ds_left = DataStore({"id": [1, 2, 3], "val_l": [10, 20, 30]})
        ds_right = DataStore({"uid": [1, 2, 4], "val_r": [100, 200, 400]})
        ds_result = ds_left.merge(ds_right, left_on="id", right_on="uid")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_left_on_right_on_left_join(self):
        pd_left = pd.DataFrame({"id": [1, 2, 3], "val_l": [10, 20, 30]})
        pd_right = pd.DataFrame({"uid": [1, 2, 4], "val_r": [100, 200, 400]})
        pd_result = pd_left.merge(pd_right, left_on="id", right_on="uid", how="left")

        ds_left = DataStore({"id": [1, 2, 3], "val_l": [10, 20, 30]})
        ds_right = DataStore({"uid": [1, 2, 4], "val_r": [100, 200, 400]})
        ds_result = ds_left.merge(ds_right, left_on="id", right_on="uid", how="left")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_left_on_right_on_with_overlapping_columns(self):
        """left_on/right_on but both have a 'name' column."""
        pd_left = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        pd_right = pd.DataFrame({"uid": [1, 2], "name": ["X", "Y"]})
        pd_result = pd_left.merge(pd_right, left_on="id", right_on="uid")

        ds_left = DataStore({"id": [1, 2], "name": ["A", "B"]})
        ds_right = DataStore({"uid": [1, 2], "name": ["X", "Y"]})
        ds_result = ds_left.merge(ds_right, left_on="id", right_on="uid")

        assert "name_x" in list(pd_result.columns)
        assert "name_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 4. File-backed DataStore merge (Parquet)
# ===========================================================================

class TestMergeFileBacked:
    """Merge between file-backed DataStores (Parquet)."""

    def test_merge_two_parquet_no_overlap(self, tmp_dir):
        pd_left = pd.DataFrame({"user_id": [1, 2, 3], "score": [85, 92, 78]})
        pd_right = pd.DataFrame({"user_id": [1, 2, 4], "city": ["NYC", "LA", "CHI"]})

        left_path = os.path.join(tmp_dir, "left.parquet")
        right_path = os.path.join(tmp_dir, "right.parquet")
        pd_left.to_parquet(left_path, index=False)
        pd_right.to_parquet(right_path, index=False)

        pd_result = pd_left.merge(pd_right, on="user_id")

        ds_left = DataStore.from_file(left_path)
        ds_right = DataStore.from_file(right_path)
        ds_result = ds_left.merge(ds_right, on="user_id")

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_two_parquet_with_overlap(self, tmp_dir):
        pd_left = pd.DataFrame({"user_id": [1, 2, 3], "name": ["A", "B", "C"]})
        pd_right = pd.DataFrame({"user_id": [1, 2, 4], "name": ["X", "Y", "Z"]})

        left_path = os.path.join(tmp_dir, "left2.parquet")
        right_path = os.path.join(tmp_dir, "right2.parquet")
        pd_left.to_parquet(left_path, index=False)
        pd_right.to_parquet(right_path, index=False)

        pd_result = pd_left.merge(pd_right, on="user_id")

        ds_left = DataStore.from_file(left_path)
        ds_right = DataStore.from_file(right_path)
        ds_result = ds_left.merge(ds_right, on="user_id")

        assert "name_x" in list(pd_result.columns)
        assert "name_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 5. Column order differences in concat/union
# ===========================================================================

class TestConcatColumnOrder:
    """SQL UNION ALL matches by position; pandas concat matches by name.
    Verify correct behavior when column orders differ."""

    def test_concat_same_columns_same_order(self):
        pd1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        pd2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        ds1 = DataStore({"A": [1, 2], "B": [3, 4]})
        ds2 = DataStore({"A": [5, 6], "B": [7, 8]})
        ds_result = ds_concat([ds1, ds2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_same_columns_different_order(self):
        """Critical test: columns in different order must align by name."""
        pd1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        pd2 = pd.DataFrame({"B": [7, 8], "A": [5, 6]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        ds1 = DataStore({"A": [1, 2], "B": [3, 4]})
        ds2 = DataStore({"B": [7, 8], "A": [5, 6]})
        ds_result = ds_concat([ds1, ds2], ignore_index=True)

        assert list(pd_result["A"]) == [1, 2, 5, 6]
        assert list(pd_result["B"]) == [3, 4, 7, 8]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_concat_different_column_sets(self):
        """When column sets differ, pandas fills missing with NaN."""
        pd1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        pd2 = pd.DataFrame({"B": [7, 8], "C": [9, 10]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        ds1 = DataStore({"A": [1, 2], "B": [3, 4]})
        ds2 = DataStore({"B": [7, 8], "C": [9, 10]})
        ds_result = ds_concat([ds1, ds2], ignore_index=True)

        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_union_same_columns_different_order(self):
        """union() SQL path should reorder columns correctly."""
        pd1 = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        pd2 = pd.DataFrame({"Y": [7, 8], "X": [5, 6]})
        pd_result = pd.concat([pd1, pd2], ignore_index=True)

        ds1 = DataStore({"X": [1, 2], "Y": [3, 4]})
        ds2 = DataStore({"Y": [7, 8], "X": [5, 6]})
        ds_result = ds1.union(ds2, all=True)

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 6. Merge then chained operations
# ===========================================================================

class TestMergeThenChain:
    """Verify merge results work correctly with subsequent operations."""

    def test_merge_then_filter(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id")
        pd_result = pd_result[pd_result["score"] > 80]

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id")
        ds_result = ds_result[ds_result["score"] > 80]

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_assign(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id")
        pd_result = pd_result.assign(score_age=pd_result["score"] + pd_result["age"])

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id")
        ds_result = ds_result.assign(score_age=ds_result["score"] + ds_result["age"])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_then_groupby(self):
        pd_left = pd.DataFrame({"key": [1, 1, 2, 2], "val": [10, 20, 30, 40]})
        pd_right = pd.DataFrame({"key": [1, 2], "label": ["A", "B"]})
        pd_merged = pd_left.merge(pd_right, on="key")
        pd_result = pd_merged.groupby("label")["val"].sum()

        ds_left = DataStore({"key": [1, 1, 2, 2], "val": [10, 20, 30, 40]})
        ds_right = DataStore({"key": [1, 2], "label": ["A", "B"]})
        ds_merged = ds_left.merge(ds_right, on="key")
        ds_result = ds_merged.groupby("label")["val"].sum()

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_overlapping_then_access_suffixed_column(self):
        """After merge with suffixes, accessing _x/_y columns must work."""
        pd_left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        pd_right = pd.DataFrame({"key": [1, 2], "val": [100, 200]})
        pd_result = pd_left.merge(pd_right, on="key")

        ds_left = DataStore({"key": [1, 2], "val": [10, 20]})
        ds_right = DataStore({"key": [1, 2], "val": [100, 200]})
        ds_result = ds_left.merge(ds_right, on="key")

        pd_sum = pd_result["val_x"] + pd_result["val_y"]
        ds_sum = ds_result["val_x"] + ds_result["val_y"]

        pd.testing.assert_series_equal(
            pd.Series(list(ds_sum), dtype=pd_sum.dtype),
            pd_sum.reset_index(drop=True),
            check_names=False,
        )


# ===========================================================================
# 7. Merge with indicator parameter
# ===========================================================================

class TestMergeIndicator:
    """indicator parameter should always fall back to pandas."""

    def test_merge_with_indicator(self, left_data, right_data):
        pd_left = pd.DataFrame(left_data)
        pd_right = pd.DataFrame(right_data)
        pd_result = pd_left.merge(pd_right, on="user_id", how="outer", indicator=True)

        ds_left = DataStore(left_data)
        ds_right = DataStore(right_data)
        ds_result = ds_left.merge(ds_right, on="user_id", how="outer", indicator=True)

        assert "_merge" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 8. Merge with multiple key columns
# ===========================================================================

class TestMergeMultipleKeys:

    def test_merge_two_key_columns(self):
        pd_left = pd.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v1": [10, 20, 30]})
        pd_right = pd.DataFrame({"k1": [1, 2], "k2": ["a", "a"], "v2": [100, 200]})
        pd_result = pd_left.merge(pd_right, on=["k1", "k2"])

        ds_left = DataStore({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v1": [10, 20, 30]})
        ds_right = DataStore({"k1": [1, 2], "k2": ["a", "a"], "v2": [100, 200]})
        ds_result = ds_left.merge(ds_right, on=["k1", "k2"])

        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_two_keys_with_overlap(self):
        """Two key columns + overlapping non-key 'val'."""
        pd_left = pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "val": [10, 20]})
        pd_right = pd.DataFrame({"k1": [1, 2], "k2": ["a", "b"], "val": [100, 200]})
        pd_result = pd_left.merge(pd_right, on=["k1", "k2"])

        ds_left = DataStore({"k1": [1, 2], "k2": ["a", "b"], "val": [10, 20]})
        ds_right = DataStore({"k1": [1, 2], "k2": ["a", "b"], "val": [100, 200]})
        ds_result = ds_left.merge(ds_right, on=["k1", "k2"])

        assert "val_x" in list(pd_result.columns)
        assert "val_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 9. merge sort parameter (SQL JOIN path must honor sort=True)
# ===========================================================================

class TestMergeSortParam:
    """merge(sort=True) must sort result by join keys, even when
    the SQL JOIN path would otherwise be used."""

    @pytest.fixture
    def parquet_pair(self, tmp_dir):
        """Create a pair of parquet-backed DataStores with unsorted keys."""
        pd_left = pd.DataFrame({"key": [3, 1, 2, 1, 3], "val": [30, 10, 20, 11, 31]})
        pd_right = pd.DataFrame({"key": [2, 3, 1], "info": ["b", "c", "a"]})

        left_path = os.path.join(tmp_dir, "sort_left.parquet")
        right_path = os.path.join(tmp_dir, "sort_right.parquet")
        pd_left.to_parquet(left_path, index=False)
        pd_right.to_parquet(right_path, index=False)

        return (
            DataStore.from_file(left_path),
            DataStore.from_file(right_path),
            pd_left,
            pd_right,
        )

    def test_merge_sort_true_file_backed(self, parquet_pair):
        ds_left, ds_right, pd_left, pd_right = parquet_pair
        pd_result = pd_left.merge(pd_right, on="key", sort=True)
        ds_result = ds_left.merge(ds_right, on="key", sort=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_sort_false_file_backed(self, parquet_pair):
        ds_left, ds_right, pd_left, pd_right = parquet_pair
        pd_result = pd_left.merge(pd_right, on="key", sort=False)
        ds_result = ds_left.merge(ds_right, on="key", sort=False)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_merge_sort_true_dict_backed(self):
        pd_left = pd.DataFrame({"key": [3, 1, 2, 1, 3], "val": [30, 10, 20, 11, 31]})
        pd_right = pd.DataFrame({"key": [2, 3, 1], "info": ["b", "c", "a"]})
        pd_result = pd_left.merge(pd_right, on="key", sort=True)

        ds_left = DataStore({"key": [3, 1, 2, 1, 3], "val": [30, 10, 20, 11, 31]})
        ds_right = DataStore({"key": [2, 3, 1], "info": ["b", "c", "a"]})
        ds_result = ds_left.merge(ds_right, on="key", sort=True)

        assert list(pd_result["key"]) == [1, 1, 2, 3, 3]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_merge_sort_true_left_join_file_backed(self, parquet_pair):
        ds_left, ds_right, pd_left, pd_right = parquet_pair
        pd_result = pd_left.merge(pd_right, on="key", how="left", sort=True)
        ds_result = ds_left.merge(ds_right, on="key", how="left", sort=True)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ===========================================================================
# 10. merge validate parameter
# ===========================================================================

class TestMergeValidateParam:
    """merge(validate=...) must raise MergeError for invalid cardinalities."""

    @pytest.fixture
    def parquet_one_to_many(self, tmp_dir):
        pd_left = pd.DataFrame({"key": [1, 1, 2], "val": [10, 11, 20]})
        pd_right = pd.DataFrame({"key": [1, 2], "info": ["a", "b"]})

        left_path = os.path.join(tmp_dir, "val_left.parquet")
        right_path = os.path.join(tmp_dir, "val_right.parquet")
        pd_left.to_parquet(left_path, index=False)
        pd_right.to_parquet(right_path, index=False)

        return (
            DataStore.from_file(left_path),
            DataStore.from_file(right_path),
            pd_left,
            pd_right,
        )

    def test_validate_one_to_one_raises_on_duplicates(self, parquet_one_to_many):
        """Left has duplicate key 1 → one_to_one should raise."""
        ds_left, ds_right, pd_left, pd_right = parquet_one_to_many

        with pytest.raises(pd.errors.MergeError):
            pd_left.merge(pd_right, on="key", validate="one_to_one")

        with pytest.raises(pd.errors.MergeError):
            ds_left.merge(ds_right, on="key", validate="one_to_one")

    def test_validate_one_to_many_raises(self, parquet_one_to_many):
        """Left has duplicate key 1 → one_to_many requires unique left keys → raises."""
        ds_left, ds_right, pd_left, pd_right = parquet_one_to_many

        with pytest.raises(pd.errors.MergeError):
            pd_left.merge(pd_right, on="key", validate="one_to_many")

        with pytest.raises(pd.errors.MergeError):
            ds_left.merge(ds_right, on="key", validate="one_to_many")

    def test_validate_many_to_one_succeeds(self, parquet_one_to_many):
        """Left has duplicate key, right is unique → many_to_one should succeed."""
        ds_left, ds_right, pd_left, pd_right = parquet_one_to_many
        pd_result = pd_left.merge(pd_right, on="key", validate="many_to_one")
        ds_result = ds_left.merge(ds_right, on="key", validate="many_to_one")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_validate_many_to_many_always_ok(self, parquet_one_to_many):
        """many_to_many never raises."""
        ds_left, ds_right, pd_left, pd_right = parquet_one_to_many
        pd_result = pd_left.merge(pd_right, on="key", validate="many_to_many")
        ds_result = ds_left.merge(ds_right, on="key", validate="many_to_many")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_validate_with_dict_backed(self):
        pd_left = pd.DataFrame({"key": [1, 1], "v": [10, 20]})
        pd_right = pd.DataFrame({"key": [1], "w": [100]})

        with pytest.raises(pd.errors.MergeError):
            pd_left.merge(pd_right, on="key", validate="one_to_one")

        ds_left = DataStore({"key": [1, 1], "v": [10, 20]})
        ds_right = DataStore({"key": [1], "w": [100]})

        with pytest.raises(pd.errors.MergeError):
            ds_left.merge(ds_right, on="key", validate="one_to_one")


# ===========================================================================
# 11. SQL JOIN path vs pandas path consistency (file-backed)
# ===========================================================================

class TestSQLJoinPathConsistency:
    """File-backed DataStores go through SQL JOIN path; verify results
    match pandas exactly."""

    @pytest.fixture
    def file_stores(self, tmp_dir):
        pd_users = pd.DataFrame({
            "uid": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "Dave"],
        })
        pd_orders = pd.DataFrame({
            "uid": [1, 2, 2, 5],
            "product": ["Widget", "Gadget", "Gizmo", "Doohickey"],
            "amount": [10.0, 20.0, 30.0, 40.0],
        })
        u_path = os.path.join(tmp_dir, "users.parquet")
        o_path = os.path.join(tmp_dir, "orders.parquet")
        pd_users.to_parquet(u_path, index=False)
        pd_orders.to_parquet(o_path, index=False)

        return (
            DataStore.from_file(u_path),
            DataStore.from_file(o_path),
            pd_users,
            pd_orders,
        )

    def test_inner_join_file_backed(self, file_stores):
        ds_users, ds_orders, pd_users, pd_orders = file_stores
        pd_result = pd_users.merge(pd_orders, on="uid")
        ds_result = ds_users.merge(ds_orders, on="uid")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_left_join_file_backed(self, file_stores):
        ds_users, ds_orders, pd_users, pd_orders = file_stores
        pd_result = pd_users.merge(pd_orders, on="uid", how="left")
        ds_result = ds_users.merge(ds_orders, on="uid", how="left")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_right_join_file_backed(self, file_stores):
        ds_users, ds_orders, pd_users, pd_orders = file_stores
        pd_result = pd_users.merge(pd_orders, on="uid", how="right")
        ds_result = ds_users.merge(ds_orders, on="uid", how="right")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_outer_join_file_backed(self, file_stores):
        ds_users, ds_orders, pd_users, pd_orders = file_stores
        pd_result = pd_users.merge(pd_orders, on="uid", how="outer")
        ds_result = ds_users.merge(ds_orders, on="uid", how="outer")
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_file_backed_overlapping_columns_fallback(self, tmp_dir):
        """File-backed DataStores with overlapping non-key columns
        must fall back to pandas and produce correct suffixes."""
        pd_left = pd.DataFrame({"key": [1, 2], "val": [10, 20]})
        pd_right = pd.DataFrame({"key": [1, 2], "val": [100, 200]})

        l_path = os.path.join(tmp_dir, "overlap_l.parquet")
        r_path = os.path.join(tmp_dir, "overlap_r.parquet")
        pd_left.to_parquet(l_path, index=False)
        pd_right.to_parquet(r_path, index=False)

        pd_result = pd_left.merge(pd_right, on="key")
        ds_result = DataStore.from_file(l_path).merge(
            DataStore.from_file(r_path), on="key"
        )

        assert "val_x" in list(pd_result.columns)
        assert "val_y" in list(pd_result.columns)
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 12. Many-to-many merge
# ===========================================================================

class TestMergeManyToMany:

    def test_many_to_many_inner(self):
        pd_left = pd.DataFrame({"key": [1, 1, 2], "a": ["x", "y", "z"]})
        pd_right = pd.DataFrame({"key": [1, 1, 2], "b": ["p", "q", "r"]})
        pd_result = pd_left.merge(pd_right, on="key")

        ds_left = DataStore({"key": [1, 1, 2], "a": ["x", "y", "z"]})
        ds_right = DataStore({"key": [1, 1, 2], "b": ["p", "q", "r"]})
        ds_result = ds_left.merge(ds_right, on="key")

        assert len(pd_result) == 5  # 2*2 + 1*1
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_many_to_many_file_backed(self, tmp_dir):
        pd_left = pd.DataFrame({"key": [1, 1, 2], "a": ["x", "y", "z"]})
        pd_right = pd.DataFrame({"key": [1, 1, 2], "b": ["p", "q", "r"]})

        l_path = os.path.join(tmp_dir, "m2m_l.parquet")
        r_path = os.path.join(tmp_dir, "m2m_r.parquet")
        pd_left.to_parquet(l_path, index=False)
        pd_right.to_parquet(r_path, index=False)

        pd_result = pd_left.merge(pd_right, on="key")
        ds_result = DataStore.from_file(l_path).merge(
            DataStore.from_file(r_path), on="key"
        )

        assert len(pd_result) == 5
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ===========================================================================
# 13. Empty merge results
# ===========================================================================

class TestMergeEmpty:

    def test_merge_no_matching_keys(self):
        pd_left = pd.DataFrame({"key": [1, 2], "v": [10, 20]})
        pd_right = pd.DataFrame({"key": [3, 4], "w": [30, 40]})
        pd_result = pd_left.merge(pd_right, on="key")

        ds_left = DataStore({"key": [1, 2], "v": [10, 20]})
        ds_right = DataStore({"key": [3, 4], "w": [30, 40]})
        ds_result = ds_left.merge(ds_right, on="key")

        assert len(pd_result) == 0
        assert len(ds_result) == 0

    def test_merge_no_matching_keys_file_backed(self, tmp_dir):
        pd_left = pd.DataFrame({"key": [1, 2], "v": [10, 20]})
        pd_right = pd.DataFrame({"key": [3, 4], "w": [30, 40]})

        l_path = os.path.join(tmp_dir, "empty_l.parquet")
        r_path = os.path.join(tmp_dir, "empty_r.parquet")
        pd_left.to_parquet(l_path, index=False)
        pd_right.to_parquet(r_path, index=False)

        ds_result = DataStore.from_file(l_path).merge(
            DataStore.from_file(r_path), on="key"
        )

        assert len(ds_result) == 0
