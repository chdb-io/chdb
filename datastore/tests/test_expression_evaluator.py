"""Tests for pandas expression evaluation."""

import pandas as pd

from datastore import DataStore, Field
from datastore.expression_evaluator import ExpressionEvaluator


class TestLikeConditionPandasFallback:
    """Test SQL LIKE semantics in the pandas evaluator."""

    @staticmethod
    def _evaluator():
        df = pd.DataFrame({"text": ["foobar", "foo", "fooo", "a.b", "axb", "FOO"]})
        return df, ExpressionEvaluator(df, DataStore(table="test"))

    def test_like_converts_percent_to_multi_character_wildcard(self):
        df, evaluator = self._evaluator()

        result = evaluator.evaluate(Field("text").like("%foo%"))

        expected = pd.Series([True, True, True, False, False, False], index=df.index, name="text")
        pd.testing.assert_series_equal(result, expected)

    def test_like_converts_underscore_to_single_character_wildcard(self):
        df, evaluator = self._evaluator()

        result = evaluator.evaluate(Field("text").like("_oo"))

        expected = pd.Series([False, True, False, False, False, False], index=df.index, name="text")
        pd.testing.assert_series_equal(result, expected)

    def test_like_escapes_regex_metacharacters(self):
        df, evaluator = self._evaluator()

        result = evaluator.evaluate(Field("text").like("a.b"))

        expected = pd.Series([False, False, False, True, False, False], index=df.index, name="text")
        pd.testing.assert_series_equal(result, expected)

    def test_ilike_is_case_insensitive_with_wildcards(self):
        df, evaluator = self._evaluator()

        result = evaluator.evaluate(Field("text").ilike("%foo%"))

        expected = pd.Series([True, True, True, False, False, True], index=df.index, name="text")
        pd.testing.assert_series_equal(result, expected)

    def test_notlike_inverts_the_wildcard_match(self):
        df, evaluator = self._evaluator()

        result = evaluator.evaluate(Field("text").notlike("%foo%"))

        expected = pd.Series([False, False, False, True, True, True], index=df.index, name="text")
        pd.testing.assert_series_equal(result, expected)
