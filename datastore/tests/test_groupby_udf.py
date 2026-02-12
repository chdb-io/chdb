"""
GroupBy UDF (User Defined Function) Test Suite

Tests complex Python functions with groupby().transform() and groupby().filter():
- Python standard library functions (math, re, collections)
- NumPy advanced functions
- Closures and external variables
- Conditional logic
- String parsing and processing
- Complex business logic
- Statistical calculations

These tests verify that arbitrary Python functions work correctly with groupby operations.
"""

import unittest
import math
import re
from collections import Counter

import numpy as np
import pandas as pd

import datastore as ds
from tests.test_utils import assert_frame_equal


class TestTransformUDF(unittest.TestCase):
    """Test UDF-style functions with groupby().transform()."""

    def setUp(self):
        self.data_pd = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C'],
                'value': [10, 20, 30, 40, 50, 60, 15, 25],
                'text': [
                    'hello world',
                    'foo bar',
                    'hello python',
                    'foo baz',
                    'hello data',
                    'foo qux',
                    'test one',
                    'test two',
                ],
                'scores': ['1,2,3', '4,5,6', '7,8,9', '10,11,12', '13,14,15', '16,17,18', '1,1,1', '2,2,2'],
            }
        )
        self.data_ds = ds.DataStore.from_df(
            pd.DataFrame(
                {
                    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C'],
                    'value': [10, 20, 30, 40, 50, 60, 15, 25],
                    'text': [
                        'hello world',
                        'foo bar',
                        'hello python',
                        'foo baz',
                        'hello data',
                        'foo qux',
                        'test one',
                        'test two',
                    ],
                    'scores': ['1,2,3', '4,5,6', '7,8,9', '10,11,12', '13,14,15', '16,17,18', '1,1,1', '2,2,2'],
                }
            )
        )

    def test_math_log_transform(self):
        """Test using math.log for transformation."""

        def log_transform(x):
            return x.apply(lambda v: math.log(v + 1))

        pd_result = self.data_pd.groupby('category')['value'].transform(log_transform)
        ds_result = self.data_ds.groupby('category')['value'].transform(log_transform)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_string_word_count(self):
        """Test string processing - word count."""

        def extract_word_count(x):
            return x.apply(lambda s: len(s.split()))

        pd_result = self.data_pd.groupby('category')['text'].transform(extract_word_count)
        ds_result = self.data_ds.groupby('category')['text'].transform(extract_word_count)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_csv_parse_and_sum(self):
        """Test CSV string parsing and calculation."""

        def parse_and_sum_scores(x):
            def calc(s):
                nums = [int(n) for n in s.split(',')]
                return sum(nums)

            return x.apply(calc)

        pd_result = self.data_pd.groupby('category')['scores'].transform(parse_and_sum_scores)
        ds_result = self.data_ds.groupby('category')['scores'].transform(parse_and_sum_scores)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_conditional_scaling(self):
        """Test conditional logic with custom scaling."""

        def custom_scaling(x):
            mean_val = x.mean()

            def scale(v):
                if v > mean_val:
                    return math.sqrt(v)
                else:
                    return v**2 / 100

            return x.apply(scale)

        pd_result = self.data_pd.groupby('category')['value'].transform(custom_scaling)
        ds_result = self.data_ds.groupby('category')['value'].transform(custom_scaling)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_closure_variables(self):
        """Test using closure variables from outer scope."""
        multiplier = 2.5
        offset = 10

        def closure_transform(x):
            return x * multiplier + offset

        pd_result = self.data_pd.groupby('category')['value'].transform(closure_transform)
        ds_result = self.data_ds.groupby('category')['value'].transform(closure_transform)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_regex_extraction(self):
        """Test regex pattern extraction."""

        def regex_extract(x):
            def extract_first_num(s):
                match = re.search(r'\d+', s)
                return int(match.group()) if match else 0

            return x.apply(extract_first_num)

        pd_result = self.data_pd.groupby('category')['scores'].transform(regex_extract)
        ds_result = self.data_ds.groupby('category')['scores'].transform(regex_extract)
        np.testing.assert_array_equal(ds_result.values, pd_result.values)

    def test_winsorize_function(self):
        """Test winsorize - limiting extreme values."""

        def winsorize(x, lower_percentile=0.1, upper_percentile=0.9):
            lower = x.quantile(lower_percentile)
            upper = x.quantile(upper_percentile)
            return x.clip(lower, upper)

        pd_result = self.data_pd.groupby('category')['value'].transform(winsorize)
        ds_result = self.data_ds.groupby('category')['value'].transform(winsorize)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_min_max_scale(self):
        """Test min-max normalization to [0, 1]."""

        def min_max_scale(x):
            min_val = x.min()
            max_val = x.max()
            if max_val == min_val:
                return pd.Series([0.5] * len(x), index=x.index)
            return (x - min_val) / (max_val - min_val)

        pd_result = self.data_pd.groupby('category')['value'].transform(min_max_scale)
        ds_result = self.data_ds.groupby('category')['value'].transform(min_max_scale)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)

    def test_rolling_mean_fill(self):
        """Test rolling mean within groups."""

        def rolling_mean_fill(x, window=2):
            return x.rolling(window=window, min_periods=1).mean()

        pd_result = self.data_pd.groupby('category')['value'].transform(rolling_mean_fill)
        ds_result = self.data_ds.groupby('category')['value'].transform(rolling_mean_fill)
        np.testing.assert_array_almost_equal(ds_result.values, pd_result.values)


class TestFilterUDF(unittest.TestCase):
    """Test UDF-style functions with groupby().filter()."""

    def setUp(self):
        self.data_pd = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C'],
                'value': [10, 20, 30, 40, 50, 60, 15, 25],
                'text': [
                    'hello world',
                    'foo bar',
                    'hello python',
                    'foo baz',
                    'hello data',
                    'foo qux',
                    'test one',
                    'test two',
                ],
                'price': [100, 200, 150, 250, 180, 300, 90, 110],
            }
        )
        self.data_ds = ds.DataStore.from_df(
            pd.DataFrame(
                {
                    'category': ['A', 'B', 'A', 'B', 'A', 'B', 'C', 'C'],
                    'value': [10, 20, 30, 40, 50, 60, 15, 25],
                    'text': [
                        'hello world',
                        'foo bar',
                        'hello python',
                        'foo baz',
                        'hello data',
                        'foo qux',
                        'test one',
                        'test two',
                    ],
                    'price': [100, 200, 150, 250, 180, 300, 90, 110],
                }
            )
        )

    def test_counter_word_frequency(self):
        """Test using Counter to check word frequency."""

        def has_common_words(group):
            all_words = []
            for text in group['text']:
                all_words.extend(text.split())
            counts = Counter(all_words)
            return any(c > 1 for c in counts.values())

        pd_result = self.data_pd.groupby('category').filter(has_common_words)
        ds_result = self.data_ds.groupby('category').filter(has_common_words)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_complex_business_rules(self):
        """Test complex business rule checking."""

        def business_rule_check(group):
            rule1 = group['value'].mean() > 20
            rule2 = any('hello' in t for t in group['text'])
            rule3 = group['price'].min() > 80
            return rule1 and (rule2 or rule3)

        pd_result = self.data_pd.groupby('category').filter(business_rule_check)
        ds_result = self.data_ds.groupby('category').filter(business_rule_check)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_high_variance_filter(self):
        """Test filtering by variance threshold."""

        def has_high_variance(group):
            return group['value'].var() > 200

        pd_result = self.data_pd.groupby('category').filter(has_high_variance)
        ds_result = self.data_ds.groupby('category').filter(has_high_variance)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_positive_correlation_filter(self):
        """Test filtering by correlation."""

        def has_positive_correlation(group):
            if len(group) < 2:
                return False
            corr = group['value'].corr(group['price'])
            return corr > 0

        pd_result = self.data_pd.groupby('category').filter(has_positive_correlation)
        ds_result = self.data_ds.groupby('category').filter(has_positive_correlation)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_skewness_filter(self):
        """Test statistical skewness detection."""

        def is_normally_distributed_like(group):
            values = group['value']
            if len(values) < 3:
                return False
            mean_val = values.mean()
            std_val = values.std()
            if std_val == 0:
                return False
            skewness = ((values - mean_val) ** 3).mean() / (std_val**3)
            return abs(skewness) < 1.0

        pd_result = self.data_pd.groupby('category').filter(is_normally_distributed_like)
        ds_result = self.data_ds.groupby('category').filter(is_normally_distributed_like)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_iqr_outlier_detection(self):
        """Test IQR-based outlier detection."""

        def has_no_outliers(group):
            values = group['value']
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = values[(values < lower) | (values > upper)]
            return len(outliers) == 0

        pd_result = self.data_pd.groupby('category').filter(has_no_outliers)
        ds_result = self.data_ds.groupby('category').filter(has_no_outliers)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_multiple_criteria_filter(self):
        """Test multiple criteria with different column operations."""

        def meets_multiple_criteria(group):
            return group['value'].mean() > 25 and group['price'].min() > 100 and len(group) >= 2

        pd_result = self.data_pd.groupby('category').filter(meets_multiple_criteria)
        ds_result = self.data_ds.groupby('category').filter(meets_multiple_criteria)
        assert_frame_equal(ds_result.to_pandas(), pd_result)

    def test_std_threshold_filter(self):
        """Test filtering by standard deviation threshold."""
        pd_result = self.data_pd.groupby('category').filter(lambda x: x['value'].std() > 15)
        ds_result = self.data_ds.groupby('category').filter(lambda x: x['value'].std() > 15)
        assert_frame_equal(ds_result.to_pandas(), pd_result)


class TestTransformWithGroupByDataFrame(unittest.TestCase):
    """Test transform on full DataFrame level via LazyGroupBy."""

    def setUp(self):
        self.data_pd = pd.DataFrame(
            {
                'category': ['A', 'B', 'A', 'B', 'A', 'B'],
                'value': [10, 20, 30, 40, 50, 60],
                'price': [100, 200, 150, 250, 180, 300],
            }
        )
        self.data_ds = ds.DataStore.from_df(
            pd.DataFrame(
                {
                    'category': ['A', 'B', 'A', 'B', 'A', 'B'],
                    'value': [10, 20, 30, 40, 50, 60],
                    'price': [100, 200, 150, 250, 180, 300],
                }
            )
        )

    def test_groupby_transform_all_numeric(self):
        """Test transform on all numeric columns."""
        pd_result = self.data_pd.groupby('category').transform(lambda x: x - x.mean())
        ds_result = self.data_ds.groupby('category').transform(lambda x: x - x.mean())
        np.testing.assert_array_almost_equal(ds_result.to_pandas().values, pd_result.values)


if __name__ == '__main__':
    unittest.main()
