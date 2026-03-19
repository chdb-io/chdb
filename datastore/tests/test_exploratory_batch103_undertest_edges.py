"""
Exploratory batch 103: Under-tested edge cases

Targets areas with low test coverage or complex logic prone to bugs:
- eval() and query() with complex expressions
- clip() boundary conditions
- replace() with dicts, regex, NaN
- combine_first() with different columns/indexes
- groupby transform/filter/apply edge cases
- cumulative operations (cumprod with zero, cumsum with NaN)
- nlargest/nsmallest with ties and NaN
- idxmin/idxmax edge cases
- Complex chained operations
- pipe() chaining
- DataFrame.map() (applymap) edge cases
- between() inclusive parameter variations
- iterrows/itertuples consistency
- Type coercion in mixed arithmetic

Mirror Code Pattern: every test compares DataStore vs pandas.
"""

import unittest

import numpy as np
import pandas as pd

from datastore import DataStore
from datastore.tests.test_utils import (
    assert_datastore_equals_pandas,
    assert_series_equal,
    get_dataframe,
    get_series,
    get_value,
)


# ============================================================
# eval() and query() edge cases
# ============================================================
class TestEvalEdgeCases(unittest.TestCase):
    """Test DataFrame.eval() with complex expressions."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50], "c": [100, 200, 300, 400, 500]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_eval_simple_arithmetic(self):
        pd_result = self.pd_df.eval("a + b")
        ds_result = get_series(self.ds_df.eval("a + b"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_multi_column_expression(self):
        pd_result = self.pd_df.eval("a + b * c")
        ds_result = get_series(self.ds_df.eval("a + b * c"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_new_column_assignment(self):
        pd_result = self.pd_df.eval("d = a + b")
        ds_result = self.ds_df.eval("d = a + b")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_eval_division(self):
        pd_result = self.pd_df.eval("b / a")
        ds_result = get_series(self.ds_df.eval("b / a"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_comparison(self):
        pd_result = self.pd_df.eval("a > 3")
        ds_result = get_series(self.ds_df.eval("a > 3"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_boolean_and(self):
        pd_result = self.pd_df.eval("a > 2 and b < 40")
        ds_result = get_series(self.ds_df.eval("a > 2 and b < 40"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_parenthesized_expression(self):
        pd_result = self.pd_df.eval("(a + b) * c")
        ds_result = get_series(self.ds_df.eval("(a + b) * c"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_negative_expression(self):
        pd_result = self.pd_df.eval("-a + b")
        ds_result = get_series(self.ds_df.eval("-a + b"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_modulo(self):
        pd_result = self.pd_df.eval("b % a")
        ds_result = get_series(self.ds_df.eval("b % a"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_eval_power(self):
        pd_result = self.pd_df.eval("a ** 2")
        ds_result = get_series(self.ds_df.eval("a ** 2"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)


class TestQueryEdgeCases(unittest.TestCase):
    """Test DataFrame.query() with complex expressions."""

    def setUp(self):
        self.data = {"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1], "z": ["a", "b", "c", "d", "e"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_query_simple_filter(self):
        pd_result = self.pd_df.query("x > 3")
        ds_result = self.ds_df.query("x > 3")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_compound_condition(self):
        pd_result = self.pd_df.query("x > 2 and y > 2")
        ds_result = self.ds_df.query("x > 2 and y > 2")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_or_condition(self):
        pd_result = self.pd_df.query("x == 1 or x == 5")
        ds_result = self.ds_df.query("x == 1 or x == 5")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_not_condition(self):
        pd_result = self.pd_df.query("not (x > 3)")
        ds_result = self.ds_df.query("not (x > 3)")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_string_comparison(self):
        pd_result = self.pd_df.query('z == "c"')
        ds_result = self.ds_df.query('z == "c"')
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_in_list(self):
        pd_result = self.pd_df.query("x in [1, 3, 5]")
        ds_result = self.ds_df.query("x in [1, 3, 5]")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_not_in_list(self):
        pd_result = self.pd_df.query("x not in [1, 3, 5]")
        ds_result = self.ds_df.query("x not in [1, 3, 5]")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_column_comparison(self):
        pd_result = self.pd_df.query("x > y")
        ds_result = self.ds_df.query("x > y")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_empty_result(self):
        pd_result = self.pd_df.query("x > 100")
        ds_result = self.ds_df.query("x > 100")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_with_variable(self):
        threshold = 3
        pd_result = self.pd_df.query("x > @threshold")
        ds_result = self.ds_df.query("x > @threshold")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_query_arithmetic_in_condition(self):
        pd_result = self.pd_df.query("x + y > 5")
        ds_result = self.ds_df.query("x + y > 5")
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# clip() edge cases
# ============================================================
class TestClipEdgeCases(unittest.TestCase):
    """Test clip() with various boundary conditions."""

    def setUp(self):
        self.data = {"a": [1, 5, 10, 15, 20], "b": [-3.0, 0.0, 3.5, 7.0, 100.0]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_clip_both_bounds(self):
        pd_result = self.pd_df.clip(lower=5, upper=15)
        ds_result = self.ds_df.clip(lower=5, upper=15)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_lower_only(self):
        pd_result = self.pd_df.clip(lower=5)
        ds_result = self.ds_df.clip(lower=5)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_upper_only(self):
        pd_result = self.pd_df.clip(upper=10)
        ds_result = self.ds_df.clip(upper=10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_single_column(self):
        pd_result = self.pd_df["a"].clip(lower=5, upper=15)
        ds_result = get_series(self.ds_df["a"].clip(lower=5, upper=15))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_with_nan(self):
        data = {"a": [1.0, np.nan, 5.0, np.nan, 10.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.clip(lower=3, upper=8)
        ds_result = ds_df.clip(lower=3, upper=8)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_clip_all_within_range(self):
        pd_result = self.pd_df["a"].clip(lower=0, upper=100)
        ds_result = get_series(self.ds_df["a"].clip(lower=0, upper=100))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_all_below_lower(self):
        pd_result = self.pd_df["a"].clip(lower=100)
        ds_result = get_series(self.ds_df["a"].clip(lower=100))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_all_above_upper(self):
        pd_result = self.pd_df["a"].clip(upper=0)
        ds_result = get_series(self.ds_df["a"].clip(upper=0))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_clip_float_bounds(self):
        pd_result = self.pd_df["b"].clip(lower=0.5, upper=50.5)
        ds_result = get_series(self.ds_df["b"].clip(lower=0.5, upper=50.5))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)


# ============================================================
# replace() edge cases
# ============================================================
class TestReplaceEdgeCases(unittest.TestCase):
    """Test replace() with various replacement patterns."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 2, 1], "b": ["x", "y", "z", "x", "y"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_replace_scalar_to_scalar(self):
        pd_result = self.pd_df.replace(1, 100)
        ds_result = self.ds_df.replace(1, 100)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_dict(self):
        pd_result = self.pd_df.replace({"a": {1: 100, 2: 200}})
        ds_result = self.ds_df.replace({"a": {1: 100, 2: 200}})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_list_to_list(self):
        pd_result = self.pd_df.replace([1, 2], [100, 200])
        ds_result = self.ds_df.replace([1, 2], [100, 200])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_string(self):
        pd_result = self.pd_df.replace("x", "replaced")
        ds_result = self.ds_df.replace("x", "replaced")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_with_nan(self):
        pd_result = self.pd_df.replace(1, np.nan)
        ds_result = self.ds_df.replace(1, np.nan)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_nan_with_value(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.replace(np.nan, 0)
        ds_result = ds_df.replace(np.nan, 0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_multiple_values_to_one(self):
        pd_result = self.pd_df.replace([1, 3], 0)
        ds_result = self.ds_df.replace([1, 3], 0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_regex_string(self):
        pd_result = self.pd_df.replace(r"^x$", "replaced", regex=True)
        ds_result = self.ds_df.replace(r"^x$", "replaced", regex=True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_flat_dict(self):
        pd_result = self.pd_df.replace({1: 100, "x": "replaced"})
        ds_result = self.ds_df.replace({1: 100, "x": "replaced"})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_replace_no_match(self):
        pd_result = self.pd_df.replace(999, 0)
        ds_result = self.ds_df.replace(999, 0)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# combine_first() edge cases
# ============================================================
class TestCombineFirstEdgeCases(unittest.TestCase):
    """Test combine_first() with various DataFrame configurations."""

    def test_combine_first_basic(self):
        data1 = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]}
        data2 = {"a": [10.0, 20.0, 30.0], "b": [40.0, 50.0, 60.0]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)
        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_different_columns(self):
        data1 = {"a": [1.0, np.nan], "b": [3.0, 4.0]}
        data2 = {"a": [10.0, 20.0], "c": [30.0, 40.0]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)
        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)
        assert_datastore_equals_pandas(ds_result, pd_result, check_column_order=False)

    def test_combine_first_no_nulls(self):
        data1 = {"a": [1, 2, 3]}
        data2 = {"a": [10, 20, 30]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)
        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_all_null_self(self):
        data1 = {"a": [np.nan, np.nan, np.nan]}
        data2 = {"a": [1.0, 2.0, 3.0]}
        pd_df1 = pd.DataFrame(data1)
        pd_df2 = pd.DataFrame(data2)
        ds_df1 = DataStore(data1)
        ds_df2 = DataStore(data2)
        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(ds_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_combine_first_with_pandas_other(self):
        """Test DataStore.combine_first with a plain pandas DataFrame as other."""
        data1 = {"a": [1.0, np.nan, 3.0]}
        data2 = {"a": [10.0, 20.0, 30.0]}
        ds_df1 = DataStore(data1)
        pd_df2 = pd.DataFrame(data2)
        pd_df1 = pd.DataFrame(data1)
        pd_result = pd_df1.combine_first(pd_df2)
        ds_result = ds_df1.combine_first(pd_df2)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# groupby transform edge cases
# ============================================================
class TestGroupByTransformEdgeCases(unittest.TestCase):
    """Test groupby().transform() edge cases."""

    def setUp(self):
        self.data = {"grp": ["a", "a", "b", "b", "b"], "val": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_transform_sum(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("sum")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("sum"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_mean(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("mean")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("mean"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_count(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("count")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("count"))
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_transform_min(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("min")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("min"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_max(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("max")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("max"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_std(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform("std")
        ds_result = get_series(self.ds_df.groupby("grp")["val"].transform("std"))
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, rtol=1e-5
        )

    def test_transform_with_nan_in_values(self):
        data = {"grp": ["a", "a", "b", "b"], "val": [1.0, np.nan, 3.0, 4.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.groupby("grp")["val"].transform("sum")
        ds_result = get_series(ds_df.groupby("grp")["val"].transform("sum"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_lambda(self):
        pd_result = self.pd_df.groupby("grp")["val"].transform(lambda x: x - x.mean())
        ds_result = get_series(
            self.ds_df.groupby("grp")["val"].transform(lambda x: x - x.mean())
        )
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_transform_dataframe_level(self):
        pd_result = self.pd_df.groupby("grp").transform("sum")
        ds_result = self.ds_df.groupby("grp").transform("sum")
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# groupby filter edge cases
# ============================================================
class TestGroupByFilterEdgeCases(unittest.TestCase):
    """Test groupby().filter() edge cases."""

    def setUp(self):
        self.data = {
            "grp": ["a", "a", "b", "b", "c"],
            "val": [10, 20, 30, 40, 50],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_filter_by_group_size(self):
        pd_result = self.pd_df.groupby("grp").filter(lambda x: len(x) >= 2)
        ds_result = self.ds_df.groupby("grp").filter(lambda x: len(x) >= 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_group_mean(self):
        pd_result = self.pd_df.groupby("grp").filter(lambda x: x["val"].mean() > 20)
        ds_result = self.ds_df.groupby("grp").filter(lambda x: x["val"].mean() > 20)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_all_groups_pass(self):
        pd_result = self.pd_df.groupby("grp").filter(lambda x: True)
        ds_result = self.ds_df.groupby("grp").filter(lambda x: True)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_no_groups_pass(self):
        pd_result = self.pd_df.groupby("grp").filter(lambda x: False)
        ds_result = self.ds_df.groupby("grp").filter(lambda x: False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_filter_by_group_sum(self):
        pd_result = self.pd_df.groupby("grp").filter(lambda x: x["val"].sum() > 50)
        ds_result = self.ds_df.groupby("grp").filter(lambda x: x["val"].sum() > 50)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# groupby apply edge cases
# ============================================================
class TestGroupByApplyEdgeCases(unittest.TestCase):
    """Test groupby().apply() edge cases."""

    def setUp(self):
        self.data = {"grp": ["a", "a", "b", "b", "b"], "val": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_apply_head(self):
        pd_result = self.pd_df.groupby("grp").apply(lambda x: x.head(1), include_groups=False)
        ds_result = self.ds_df.groupby("grp").apply(lambda x: x.head(1), include_groups=False)
        ds_df = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df.reset_index(drop=True), pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_apply_sort_within_group(self):
        pd_result = self.pd_df.groupby("grp").apply(
            lambda x: x.sort_values("val", ascending=False), include_groups=False
        )
        ds_result = self.ds_df.groupby("grp").apply(
            lambda x: x.sort_values("val", ascending=False), include_groups=False
        )
        ds_df = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df.reset_index(drop=True), pd_result.reset_index(drop=True),
            check_dtype=False
        )

    def test_apply_nlargest(self):
        pd_result = self.pd_df.groupby("grp").apply(
            lambda x: x.nlargest(2, "val"), include_groups=False
        )
        ds_result = self.ds_df.groupby("grp").apply(
            lambda x: x.nlargest(2, "val"), include_groups=False
        )
        ds_df = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df.reset_index(drop=True), pd_result.reset_index(drop=True),
            check_dtype=False
        )


# ============================================================
# Cumulative operations edge cases
# ============================================================
class TestCumulativeEdgeCases(unittest.TestCase):
    """Test cumulative operations with edge cases."""

    def test_cumsum_basic(self):
        data = {"a": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumsum()
        ds_result = get_series(ds_df["a"].cumsum())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cumsum_with_nan(self):
        data = {"a": [1.0, np.nan, 3.0, 4.0, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumsum()
        ds_result = get_series(ds_df["a"].cumsum())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cumprod_basic(self):
        data = {"a": [1, 2, 3, 4]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumprod()
        ds_result = get_series(ds_df["a"].cumprod())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_cumprod_with_zero(self):
        data = {"a": [2, 3, 0, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumprod()
        ds_result = get_series(ds_df["a"].cumprod())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_cumprod_with_nan(self):
        data = {"a": [1.0, np.nan, 3.0, 4.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumprod()
        ds_result = get_series(ds_df["a"].cumprod())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cummax_basic(self):
        data = {"a": [3, 1, 4, 1, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cummax()
        ds_result = get_series(ds_df["a"].cummax())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_cummin_basic(self):
        data = {"a": [5, 3, 4, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cummin()
        ds_result = get_series(ds_df["a"].cummin())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_cummax_with_nan(self):
        data = {"a": [3.0, np.nan, 4.0, 1.0, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cummax()
        ds_result = get_series(ds_df["a"].cummax())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cummin_with_nan(self):
        data = {"a": [5.0, np.nan, 3.0, 4.0, 1.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cummin()
        ds_result = get_series(ds_df["a"].cummin())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cumsum_negative_values(self):
        data = {"a": [1, -2, 3, -4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumsum()
        ds_result = get_series(ds_df["a"].cumsum())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_cumprod_negative_values(self):
        data = {"a": [1, -2, 3, -4]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].cumprod()
        ds_result = get_series(ds_df["a"].cumprod())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )


# ============================================================
# Groupby cumulative operations
# ============================================================
class TestGroupByCumulative(unittest.TestCase):
    """Test cumulative operations within groupby."""

    def setUp(self):
        self.data = {"grp": ["a", "a", "b", "b", "a"], "val": [1, 2, 3, 4, 5]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_groupby_cumsum(self):
        pd_result = self.pd_df.groupby("grp")["val"].cumsum()
        ds_result = get_series(self.ds_df.groupby("grp")["val"].cumsum())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_groupby_cumcount(self):
        pd_result = self.pd_df.groupby("grp").cumcount()
        ds_result = get_series(self.ds_df.groupby("grp").cumcount())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_groupby_cumcount_ascending_false(self):
        pd_result = self.pd_df.groupby("grp").cumcount(ascending=False)
        ds_result = get_series(self.ds_df.groupby("grp").cumcount(ascending=False))
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_groupby_cummax(self):
        pd_result = self.pd_df.groupby("grp")["val"].cummax()
        ds_result = get_series(self.ds_df.groupby("grp")["val"].cummax())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_groupby_cummin(self):
        pd_result = self.pd_df.groupby("grp")["val"].cummin()
        ds_result = get_series(self.ds_df.groupby("grp")["val"].cummin())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )


# ============================================================
# nlargest / nsmallest edge cases
# ============================================================
class TestNlargestNsmallest(unittest.TestCase):
    """Test nlargest/nsmallest with ties, NaN, and edge cases."""

    def setUp(self):
        self.data = {"name": ["a", "b", "c", "d", "e"], "val": [3, 1, 4, 1, 5]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_nlargest_basic(self):
        pd_result = self.pd_df.nlargest(3, "val")
        ds_result = self.ds_df.nlargest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_basic(self):
        pd_result = self.pd_df.nsmallest(3, "val")
        ds_result = self.ds_df.nsmallest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_ties_keep_first(self):
        data = {"val": [3, 3, 3, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(2, "val", keep="first")
        ds_result = ds_df.nlargest(2, "val", keep="first")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_ties_keep_last(self):
        data = {"val": [3, 3, 3, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(2, "val", keep="last")
        ds_result = ds_df.nlargest(2, "val", keep="last")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_ties_keep_all(self):
        data = {"val": [3, 3, 3, 1, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(2, "val", keep="all")
        ds_result = ds_df.nlargest(2, "val", keep="all")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_with_ties_keep_all(self):
        data = {"val": [1, 1, 1, 3, 2]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nsmallest(2, "val", keep="all")
        ds_result = ds_df.nsmallest(2, "val", keep="all")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_n_exceeds_rows(self):
        pd_result = self.pd_df.nlargest(100, "val")
        ds_result = self.ds_df.nlargest(100, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_n_exceeds_rows(self):
        pd_result = self.pd_df.nsmallest(100, "val")
        ds_result = self.ds_df.nsmallest(100, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_with_nan(self):
        data = {"val": [3.0, np.nan, 4.0, 1.0, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nlargest(3, "val")
        ds_result = ds_df.nlargest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nsmallest_with_nan(self):
        data = {"val": [3.0, np.nan, 4.0, 1.0, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nsmallest(3, "val")
        ds_result = ds_df.nsmallest(3, "val")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_nlargest_series(self):
        pd_result = self.pd_df["val"].nlargest(3)
        ds_result = get_series(self.ds_df["val"].nlargest(3))
        pd.testing.assert_series_equal(
            ds_result.reset_index(drop=True), pd_result.reset_index(drop=True),
            check_names=False, check_dtype=False,
        )

    def test_nsmallest_series(self):
        pd_result = self.pd_df["val"].nsmallest(3)
        ds_result = get_series(self.ds_df["val"].nsmallest(3))
        pd.testing.assert_series_equal(
            ds_result.reset_index(drop=True), pd_result.reset_index(drop=True),
            check_names=False, check_dtype=False,
        )


# ============================================================
# idxmin / idxmax edge cases
# ============================================================
class TestIdxminIdxmax(unittest.TestCase):
    """Test idxmin/idxmax edge cases."""

    def test_idxmin_basic(self):
        data = {"a": [3, 1, 4, 1, 5], "b": [9, 7, 5, 3, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.idxmin()
        ds_result = ds_df.idxmin()
        if not isinstance(ds_result, pd.Series):
            ds_result = get_series(ds_result)
        pd.testing.assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_idxmax_basic(self):
        data = {"a": [3, 1, 4, 1, 5], "b": [9, 7, 5, 3, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.idxmax()
        ds_result = ds_df.idxmax()
        if not isinstance(ds_result, pd.Series):
            ds_result = get_series(ds_result)
        pd.testing.assert_series_equal(ds_result, pd_result, check_dtype=False)

    def test_idxmin_series(self):
        data = {"a": [3, 1, 4, 1, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].idxmin()
        ds_result = get_value(ds_df["a"].idxmin())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_idxmax_series(self):
        data = {"a": [3, 1, 4, 1, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].idxmax()
        ds_result = get_value(ds_df["a"].idxmax())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_idxmin_with_nan(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].idxmin()
        ds_result = get_value(ds_df["a"].idxmin())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_idxmax_with_nan(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].idxmax()
        ds_result = get_value(ds_df["a"].idxmax())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_idxmin_ties(self):
        """idxmin should return first occurrence when tied."""
        data = {"a": [1, 1, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].idxmin()
        ds_result = get_value(ds_df["a"].idxmin())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"


# ============================================================
# pipe() chaining
# ============================================================
class TestPipeChaining(unittest.TestCase):
    """Test pipe() for function chaining."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_pipe_simple_function(self):
        def double_values(df):
            return df * 2

        pd_result = self.pd_df.pipe(double_values)
        ds_result = self.ds_df.pipe(double_values)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_with_args(self):
        def add_constant(df, constant):
            return df + constant

        pd_result = self.pd_df.pipe(add_constant, 10)
        ds_result = self.ds_df.pipe(add_constant, 10)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_chaining_multiple(self):
        def double_values(df):
            return df * 2

        def add_constant(df, constant):
            return df + constant

        pd_result = self.pd_df.pipe(double_values).pipe(add_constant, 5)
        ds_result = self.ds_df.pipe(double_values).pipe(add_constant, 5)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_filter_function(self):
        def keep_big(df):
            return df[df["a"] > 2]

        pd_result = self.pd_df.pipe(keep_big)
        ds_result = self.ds_df.pipe(keep_big)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_pipe_rename(self):
        def rename_cols(df):
            return df.rename(columns={"a": "x", "b": "y"})

        pd_result = self.pd_df.pipe(rename_cols)
        ds_result = self.ds_df.pipe(rename_cols)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# between() edge cases
# ============================================================
class TestBetweenEdgeCases(unittest.TestCase):
    """Test between() with inclusive parameter variations."""

    def setUp(self):
        self.data = {"val": [1, 2, 3, 4, 5]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_between_both_inclusive(self):
        pd_result = self.pd_df[self.pd_df["val"].between(2, 4, inclusive="both")]
        ds_result = self.ds_df[self.ds_df["val"].between(2, 4, inclusive="both")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_neither_inclusive(self):
        pd_result = self.pd_df[self.pd_df["val"].between(2, 4, inclusive="neither")]
        ds_result = self.ds_df[self.ds_df["val"].between(2, 4, inclusive="neither")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_left_inclusive(self):
        pd_result = self.pd_df[self.pd_df["val"].between(2, 4, inclusive="left")]
        ds_result = self.ds_df[self.ds_df["val"].between(2, 4, inclusive="left")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_right_inclusive(self):
        pd_result = self.pd_df[self.pd_df["val"].between(2, 4, inclusive="right")]
        ds_result = self.ds_df[self.ds_df["val"].between(2, 4, inclusive="right")]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_series_result(self):
        pd_result = self.pd_df["val"].between(2, 4)
        ds_result = get_series(self.ds_df["val"].between(2, 4))
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_between_float_values(self):
        data = {"val": [1.5, 2.5, 3.5, 4.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df["val"].between(2.0, 4.0)]
        ds_result = ds_df[ds_df["val"].between(2.0, 4.0)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_no_match(self):
        pd_result = self.pd_df[self.pd_df["val"].between(10, 20)]
        ds_result = self.ds_df[self.ds_df["val"].between(10, 20)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_between_all_match(self):
        pd_result = self.pd_df[self.pd_df["val"].between(0, 100)]
        ds_result = self.ds_df[self.ds_df["val"].between(0, 100)]
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# DataFrame.map() / applymap() edge cases
# ============================================================
class TestMapApplymap(unittest.TestCase):
    """Test element-wise map/applymap operations."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_applymap_square(self):
        pd_result = self.pd_df.map(lambda x: x ** 2)
        ds_result = self.ds_df.applymap(lambda x: x ** 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_string_format(self):
        pd_result = self.pd_df.map(lambda x: f"val_{x}")
        ds_result = self.ds_df.applymap(lambda x: f"val_{x}")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_with_nan(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.map(lambda x: x * 2, na_action="ignore")
        ds_result = ds_df.applymap(lambda x: x * 2, na_action="ignore")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_applymap_type_change(self):
        pd_result = self.pd_df.map(str)
        ds_result = self.ds_df.applymap(str)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# Series.map() edge cases
# ============================================================
class TestSeriesMapEdgeCases(unittest.TestCase):
    """Test Series.map() with dict, function, and Series mapping."""

    def setUp(self):
        self.data = {"a": [1, 2, 3, 2, 1]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_map_with_dict(self):
        mapping = {1: "one", 2: "two", 3: "three"}
        pd_result = self.pd_df["a"].map(mapping)
        ds_result = get_series(self.ds_df["a"].map(mapping))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_map_with_function(self):
        pd_result = self.pd_df["a"].map(lambda x: x * 10)
        ds_result = get_series(self.ds_df["a"].map(lambda x: x * 10))
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_map_dict_missing_key(self):
        """Keys not in dict should become NaN."""
        mapping = {1: "one", 2: "two"}
        pd_result = self.pd_df["a"].map(mapping)
        ds_result = get_series(self.ds_df["a"].map(mapping))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_map_with_nan_in_data(self):
        data = {"a": [1.0, np.nan, 3.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].map(lambda x: x * 2)
        ds_result = get_series(ds_df["a"].map(lambda x: x * 2))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_map_na_action_ignore(self):
        data = {"a": [1.0, np.nan, 3.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].map(lambda x: x * 2, na_action="ignore")
        ds_result = get_series(ds_df["a"].map(lambda x: x * 2, na_action="ignore"))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)


# ============================================================
# iterrows / itertuples consistency
# ============================================================
class TestIterationMethods(unittest.TestCase):
    """Test iterrows/itertuples/items consistency with pandas."""

    def setUp(self):
        self.data = {"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_iterrows_values(self):
        pd_rows = list(self.pd_df.iterrows())
        ds_rows = list(self.ds_df.iterrows())
        assert len(pd_rows) == len(ds_rows)
        for (pd_idx, pd_row), (ds_idx, ds_row) in zip(pd_rows, ds_rows):
            assert pd_idx == ds_idx, f"Index mismatch: {pd_idx} vs {ds_idx}"
            pd.testing.assert_series_equal(ds_row, pd_row)

    def test_itertuples_values(self):
        pd_tuples = list(self.pd_df.itertuples())
        ds_tuples = list(self.ds_df.itertuples())
        assert len(pd_tuples) == len(ds_tuples)
        for pd_tup, ds_tup in zip(pd_tuples, ds_tuples):
            assert pd_tup == ds_tup, f"Tuple mismatch: {pd_tup} vs {ds_tup}"

    def test_itertuples_no_index(self):
        pd_tuples = list(self.pd_df.itertuples(index=False))
        ds_tuples = list(self.ds_df.itertuples(index=False))
        assert len(pd_tuples) == len(ds_tuples)
        for pd_tup, ds_tup in zip(pd_tuples, ds_tuples):
            assert pd_tup == ds_tup, f"Tuple mismatch: {pd_tup} vs {ds_tup}"

    def test_items_columns(self):
        pd_items = list(self.pd_df.items())
        ds_items = list(self.ds_df.items())
        assert len(pd_items) == len(ds_items)
        for (pd_name, pd_col), (ds_name, ds_col) in zip(pd_items, ds_items):
            assert pd_name == ds_name
            pd.testing.assert_series_equal(ds_col, pd_col)


# ============================================================
# Complex chained operations
# ============================================================
class TestComplexChains(unittest.TestCase):
    """Test complex chains of operations for correctness."""

    def test_filter_assign_groupby_agg(self):
        data = {
            "dept": ["eng", "eng", "sales", "sales", "eng", "sales"],
            "salary": [100, 120, 80, 90, 110, 85],
            "bonus": [10, 15, 8, 9, 12, 7],
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = (
            pd_df[pd_df["salary"] > 85]
            .assign(total=lambda df: df["salary"] + df["bonus"])
            .groupby("dept")
            .agg({"total": "mean"})
            .reset_index()
        )
        ds_result = (
            ds_df[ds_df["salary"] > 85]
            .assign(total=lambda df: df["salary"] + df["bonus"])
            .groupby("dept")
            .agg({"total": "mean"})
            .reset_index()
        )
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_groupby_agg_sort_head(self):
        data = {
            "category": ["A", "B", "A", "C", "B", "C", "A"],
            "value": [10, 20, 30, 40, 50, 60, 70],
        }
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = (
            pd_df.groupby("category")
            .agg({"value": "sum"})
            .reset_index()
            .sort_values("value", ascending=False)
            .head(2)
        )
        ds_result = (
            ds_df.groupby("category")
            .agg({"value": "sum"})
            .reset_index()
            .sort_values("value", ascending=False)
            .head(2)
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_multi_filter_chain(self):
        data = {"a": list(range(20)), "b": [x % 3 for x in range(20)]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[(pd_df["a"] > 5) & (pd_df["a"] < 15) & (pd_df["b"] == 0)]
        ds_result = ds_df[(ds_df["a"] > 5) & (ds_df["a"] < 15) & (ds_df["b"] == 0)]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_assign_then_filter(self):
        data = {"x": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df.assign(x_squared=lambda df: df["x"] ** 2).query(
            "x_squared > 5"
        )
        ds_result = ds_df.assign(x_squared=lambda df: df["x"] ** 2).query(
            "x_squared > 5"
        )
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_groupby(self):
        data = {"dept": ["A", "A", "B", "B"], "sal": [100, 200, 150, 250]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = (
            pd_df.rename(columns={"dept": "department", "sal": "salary"})
            .groupby("department")
            .agg({"salary": "mean"})
            .reset_index()
        )
        ds_result = (
            ds_df.rename(columns={"dept": "department", "sal": "salary"})
            .groupby("department")
            .agg({"salary": "mean"})
            .reset_index()
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_value_counts_after_filter(self):
        data = {"color": ["red", "blue", "red", "green", "blue", "red", "green"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)

        pd_result = pd_df[pd_df["color"] != "green"]["color"].value_counts()
        ds_result = get_series(
            ds_df[ds_df["color"] != "green"]["color"].value_counts()
        )
        pd.testing.assert_series_equal(
            ds_result.sort_index(), pd_result.sort_index(),
            check_names=False, check_dtype=False,
        )


# ============================================================
# Type coercion in arithmetic
# ============================================================
class TestTypeCoercion(unittest.TestCase):
    """Test type handling in mixed arithmetic and comparisons."""

    def test_int_float_addition(self):
        data = {"int_col": [1, 2, 3], "float_col": [1.5, 2.5, 3.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["int_col"] + pd_df["float_col"]
        ds_result = get_series(ds_df["int_col"] + ds_df["float_col"])
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_int_float_comparison(self):
        data = {"int_col": [1, 2, 3], "float_col": [1.5, 2.0, 2.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df["int_col"] >= pd_df["float_col"]]
        ds_result = ds_df[ds_df["int_col"] >= ds_df["float_col"]]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_scalar_broadcast_arithmetic(self):
        data = {"a": [10, 20, 30]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"] / 3
        ds_result = get_series(ds_df["a"] / 3)
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, rtol=1e-5
        )

    def test_boolean_column_sum(self):
        data = {"flag": [True, False, True, True, False]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["flag"].sum()
        ds_result = get_value(ds_df["flag"].sum())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_boolean_column_mean(self):
        data = {"flag": [True, False, True, True, False]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["flag"].mean()
        ds_result = get_value(ds_df["flag"].mean())
        assert abs(pd_result - ds_result) < 1e-6, f"Expected {pd_result}, got {ds_result}"

    def test_string_concatenation_column(self):
        data = {"first": ["John", "Jane", "Bob"], "last": ["Doe", "Smith", "Lee"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["first"] + " " + pd_df["last"]
        ds_result = get_series(ds_df["first"] + " " + ds_df["last"])
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)


# ============================================================
# DataFrame describe() edge cases
# ============================================================
class TestDescribeEdgeCases(unittest.TestCase):
    """Test describe() with various DataFrame configurations."""

    def test_describe_numeric(self):
        data = {"a": [1, 2, 3, 4, 5], "b": [1.0, 2.5, 3.5, 4.0, 5.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        ds_df_result = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df_result, pd_result, check_dtype=False, rtol=1e-5,
        )

    def test_describe_mixed_types(self):
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        ds_df_result = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df_result, pd_result, check_dtype=False, rtol=1e-5,
        )

    def test_describe_include_all(self):
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe(include="all")
        ds_result = ds_df.describe(include="all")
        ds_df_result = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df_result, pd_result, check_dtype=False, rtol=1e-5,
        )

    def test_describe_single_column(self):
        data = {"a": [1, 2, 3, 4, 5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].describe()
        ds_result = ds_df["a"].describe()
        ds_val = get_series(ds_result) if not isinstance(ds_result, pd.Series) else ds_result
        pd.testing.assert_series_equal(
            ds_val, pd_result, check_names=False, check_dtype=False, rtol=1e-5,
        )

    def test_describe_with_nan(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.describe()
        ds_result = ds_df.describe()
        ds_df_result = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df_result, pd_result, check_dtype=False, rtol=1e-5,
        )


# ============================================================
# Empty DataFrame operations
# ============================================================
class TestEmptyDataFrame(unittest.TestCase):
    """Test operations on empty DataFrames."""

    def test_empty_after_filter(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df["a"] > 100]
        ds_result = ds_df[ds_df["a"] > 100]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_groupby_agg(self):
        data = {"grp": pd.Series([], dtype="str"), "val": pd.Series([], dtype="int64")}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.groupby("grp").agg({"val": "sum"}).reset_index()
        ds_result = ds_df.groupby("grp").agg({"val": "sum"}).reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_empty_shape(self):
        data = {"a": [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df["a"] > 100]
        ds_result = ds_df[ds_df["a"] > 100]
        pd_shape = pd_result.shape
        ds_shape = ds_result.shape
        assert pd_shape == ds_shape, f"Shape mismatch: pd={pd_shape}, ds={ds_shape}"

    def test_empty_columns_preserved(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df[pd_df["a"] > 100]
        ds_result = ds_df[ds_df["a"] > 100]
        assert list(pd_result.columns) == list(ds_result.columns)

    def test_empty_sum(self):
        data = {"a": [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_empty = pd_df[pd_df["a"] > 100]
        ds_empty = ds_df[ds_df["a"] > 100]
        pd_result = pd_empty["a"].sum()
        ds_result = get_value(ds_empty["a"].sum())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"


# ============================================================
# Single-row DataFrame operations
# ============================================================
class TestSingleRowDataFrame(unittest.TestCase):
    """Test operations on single-row DataFrames."""

    def setUp(self):
        self.data = {"a": [42], "b": [3.14], "c": ["hello"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_single_row_select(self):
        pd_result = self.pd_df[["a", "b"]]
        ds_result = self.ds_df[["a", "b"]]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_single_row_describe(self):
        pd_result = self.pd_df.describe()
        ds_result = self.ds_df.describe()
        ds_df_result = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df_result, pd_result, check_dtype=False, rtol=1e-5,
        )

    def test_single_row_value(self):
        pd_val = self.pd_df["a"].iloc[0]
        ds_val = get_value(self.ds_df["a"].iloc[0])
        assert pd_val == ds_val

    def test_single_row_shape(self):
        assert self.pd_df.shape == self.ds_df.shape

    def test_single_row_sum(self):
        pd_result = self.pd_df["a"].sum()
        ds_result = get_value(self.ds_df["a"].sum())
        assert pd_result == ds_result


# ============================================================
# Multi-column groupby with named aggregation
# ============================================================
class TestMultiColumnGroupBy(unittest.TestCase):
    """Test groupby with multiple group keys and named aggregation."""

    def setUp(self):
        self.data = {
            "dept": ["eng", "eng", "sales", "sales", "eng", "sales"],
            "level": ["jr", "sr", "jr", "sr", "sr", "jr"],
            "salary": [80, 120, 70, 110, 130, 75],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_multi_key_groupby_sum(self):
        pd_result = self.pd_df.groupby(["dept", "level"]).agg({"salary": "sum"}).reset_index()
        ds_result = self.ds_df.groupby(["dept", "level"]).agg({"salary": "sum"}).reset_index()
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)

    def test_multi_key_groupby_multiple_agg(self):
        pd_result = (
            self.pd_df.groupby(["dept", "level"])
            .agg({"salary": ["mean", "sum", "count"]})
            .reset_index()
        )
        ds_result = (
            self.ds_df.groupby(["dept", "level"])
            .agg({"salary": ["mean", "sum", "count"]})
            .reset_index()
        )
        ds_df = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_dtype=False,
            rtol=1e-5,
        )

    def test_multi_key_groupby_size(self):
        pd_result = self.pd_df.groupby(["dept", "level"]).size().reset_index(name="count")
        ds_result = self.ds_df.groupby(["dept", "level"]).size().reset_index(name="count")
        ds_df = get_dataframe(ds_result)
        pd.testing.assert_frame_equal(
            ds_df.sort_values(["dept", "level"]).reset_index(drop=True),
            pd_result.sort_values(["dept", "level"]).reset_index(drop=True),
            check_dtype=False,
        )

    def test_multi_key_named_agg(self):
        pd_result = (
            self.pd_df.groupby("dept")
            .agg(avg_salary=("salary", "mean"), max_salary=("salary", "max"))
            .reset_index()
        )
        ds_result = (
            self.ds_df.groupby("dept")
            .agg(avg_salary=("salary", "mean"), max_salary=("salary", "max"))
            .reset_index()
        )
        assert_datastore_equals_pandas(ds_result, pd_result, check_row_order=False)


# ============================================================
# DataFrame arithmetic operations
# ============================================================
class TestDataFrameArithmetic(unittest.TestCase):
    """Test DataFrame-level arithmetic operations."""

    def setUp(self):
        self.data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_df_add_scalar(self):
        pd_result = self.pd_df + 10
        ds_result = self.ds_df + 10
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_multiply_scalar(self):
        pd_result = self.pd_df * 2
        ds_result = self.ds_df * 2
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_subtract_scalar(self):
        pd_result = self.pd_df - 1
        ds_result = self.ds_df - 1
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_floor_division(self):
        pd_result = self.pd_df // 2
        ds_result = self.ds_df // 2
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_modulo(self):
        pd_result = self.pd_df % 2
        ds_result = self.ds_df % 2
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_df_negative(self):
        pd_result = -self.pd_df
        ds_result = -self.ds_df
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# Unique and nunique edge cases
# ============================================================
class TestUniqueNunique(unittest.TestCase):
    """Test unique() and nunique() edge cases."""

    def test_nunique_basic(self):
        data = {"a": [1, 2, 2, 3, 3, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].nunique()
        ds_result = get_value(ds_df["a"].nunique())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_nunique_with_nan(self):
        data = {"a": [1.0, 2.0, np.nan, 2.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].nunique()
        ds_result = get_value(ds_df["a"].nunique())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_unique_basic(self):
        data = {"a": [3, 1, 2, 1, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = sorted(pd_df["a"].unique().tolist())
        ds_result = sorted(get_value(ds_df["a"].unique()).tolist())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_unique_strings(self):
        data = {"a": ["b", "a", "c", "a", "b"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = sorted(pd_df["a"].unique().tolist())
        ds_result = sorted(get_value(ds_df["a"].unique()).tolist())
        assert pd_result == ds_result, f"Expected {pd_result}, got {ds_result}"

    def test_nunique_dataframe(self):
        data = {"a": [1, 2, 2], "b": ["x", "x", "y"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.nunique()
        ds_result = ds_df.nunique()
        if not isinstance(ds_result, pd.Series):
            ds_result = get_series(ds_result)
        pd.testing.assert_series_equal(ds_result, pd_result, check_dtype=False)


# ============================================================
# drop_duplicates edge cases
# ============================================================
class TestDropDuplicatesEdgeCases(unittest.TestCase):
    """Test drop_duplicates with various parameters."""

    def setUp(self):
        self.data = {"a": [1, 2, 2, 3, 3, 3], "b": ["x", "y", "y", "z", "z", "z"]}
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_drop_duplicates_default(self):
        pd_result = self.pd_df.drop_duplicates()
        ds_result = self.ds_df.drop_duplicates()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_subset(self):
        pd_result = self.pd_df.drop_duplicates(subset=["a"])
        ds_result = self.ds_df.drop_duplicates(subset=["a"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_last(self):
        pd_result = self.pd_df.drop_duplicates(keep="last")
        ds_result = self.ds_df.drop_duplicates(keep="last")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_duplicates_keep_false(self):
        pd_result = self.pd_df.drop_duplicates(keep=False)
        ds_result = self.ds_df.drop_duplicates(keep=False)
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_row_order=False
        )

    def test_drop_duplicates_no_duplicates(self):
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.drop_duplicates()
        ds_result = ds_df.drop_duplicates()
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# sort_values edge cases
# ============================================================
class TestSortValuesEdgeCases(unittest.TestCase):
    """Test sort_values with various configurations."""

    def setUp(self):
        self.data = {
            "a": [3, 1, 4, 1, 5],
            "b": ["x", "y", "z", "w", "v"],
        }
        self.pd_df = pd.DataFrame(self.data)
        self.ds_df = DataStore(self.data)

    def test_sort_ascending(self):
        pd_result = self.pd_df.sort_values("a")
        ds_result = self.ds_df.sort_values("a")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_descending(self):
        pd_result = self.pd_df.sort_values("a", ascending=False)
        ds_result = self.ds_df.sort_values("a", ascending=False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_columns(self):
        data = {"a": [1, 1, 2, 2], "b": [4, 3, 2, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values(["a", "b"])
        ds_result = ds_df.sort_values(["a", "b"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_multiple_ascending(self):
        data = {"a": [1, 1, 2, 2], "b": [4, 3, 2, 1]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values(["a", "b"], ascending=[True, False])
        ds_result = ds_df.sort_values(["a", "b"], ascending=[True, False])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_with_nan(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 2.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values("a")
        ds_result = ds_df.sort_values("a")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_na_position_first(self):
        data = {"a": [3.0, np.nan, 1.0, np.nan, 2.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.sort_values("a", na_position="first")
        ds_result = ds_df.sort_values("a", na_position="first")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_sort_strings(self):
        pd_result = self.pd_df.sort_values("b")
        ds_result = self.ds_df.sort_values("b")
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# fillna edge cases
# ============================================================
class TestFillnaEdgeCases(unittest.TestCase):
    """Test fillna with various configurations."""

    def test_fillna_scalar(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_dict(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.fillna({"a": 0, "b": -1})
        ds_result = ds_df.fillna({"a": 0, "b": -1})
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_ffill(self):
        data = {"a": [1.0, np.nan, np.nan, 4.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.ffill()
        ds_result = ds_df.ffill()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_method_bfill(self):
        data = {"a": [np.nan, np.nan, 3.0, 4.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.bfill()
        ds_result = ds_df.bfill()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_series(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan, 5.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].fillna(0)
        ds_result = get_series(ds_df["a"].fillna(0))
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_fillna_no_nulls(self):
        data = {"a": [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.fillna(0)
        ds_result = ds_df.fillna(0)
        assert_datastore_equals_pandas(ds_result, pd_result)


# ============================================================
# dropna edge cases
# ============================================================
class TestDropnaEdgeCases(unittest.TestCase):
    """Test dropna with various configurations."""

    def test_dropna_any(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all(self):
        data = {"a": [1.0, np.nan, np.nan], "b": [np.nan, np.nan, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.dropna(how="all")
        ds_result = ds_df.dropna(how="all")
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_subset(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 5.0, 6.0]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.dropna(subset=["a"])
        ds_result = ds_df.dropna(subset=["a"])
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_no_nulls(self):
        data = {"a": [1, 2, 3], "b": [4, 5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_all_nulls(self):
        data = {"a": [np.nan, np.nan, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.dropna()
        ds_result = ds_df.dropna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_dropna_series(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].dropna()
        ds_result = get_series(ds_df["a"].dropna())
        pd.testing.assert_series_equal(
            ds_result.reset_index(drop=True),
            pd_result.reset_index(drop=True),
            check_names=False,
        )


# ============================================================
# isna / notna edge cases
# ============================================================
class TestIsnaNotna(unittest.TestCase):
    """Test isna/notna operations."""

    def test_isna_dataframe(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.isna()
        ds_result = ds_df.isna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_notna_dataframe(self):
        data = {"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.notna()
        ds_result = ds_df.notna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_series(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].isna()
        ds_result = get_series(ds_df["a"].isna())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_notna_series(self):
        data = {"a": [1.0, np.nan, 3.0, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].notna()
        ds_result = get_series(ds_df["a"].notna())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_isna_no_nulls(self):
        data = {"a": [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.isna()
        ds_result = ds_df.isna()
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_isna_all_nulls(self):
        data = {"a": [np.nan, np.nan, np.nan]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].isna()
        ds_result = get_series(ds_df["a"].isna())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )


# ============================================================
# abs() edge cases
# ============================================================
class TestAbsEdgeCases(unittest.TestCase):
    """Test abs() operations."""

    def test_abs_positive(self):
        data = {"a": [1, 2, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].abs()
        ds_result = get_series(ds_df["a"].abs())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_abs_negative(self):
        data = {"a": [-1, -2, -3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].abs()
        ds_result = get_series(ds_df["a"].abs())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_abs_mixed(self):
        data = {"a": [-3, -1, 0, 1, 3]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].abs()
        ds_result = get_series(ds_df["a"].abs())
        pd.testing.assert_series_equal(
            ds_result, pd_result, check_names=False, check_dtype=False
        )

    def test_abs_float(self):
        data = {"a": [-1.5, -0.5, 0.0, 0.5, 1.5]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df["a"].abs()
        ds_result = get_series(ds_df["a"].abs())
        pd.testing.assert_series_equal(ds_result, pd_result, check_names=False)

    def test_abs_dataframe(self):
        data = {"a": [-1, 2, -3], "b": [4, -5, 6]}
        pd_df = pd.DataFrame(data)
        ds_df = DataStore(data)
        pd_result = pd_df.abs()
        ds_result = ds_df.abs()
        assert_datastore_equals_pandas(ds_result, pd_result, check_nullable_dtype=False)


if __name__ == "__main__":
    unittest.main()
