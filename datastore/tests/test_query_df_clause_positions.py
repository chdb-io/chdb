"""
query_df() adds ``_row_id`` ordering to SQL on a DataFrame by locating the
ORDER BY, LIMIT, OFFSET and SETTINGS clauses. Quoted strings and identifiers
that contain those words are not clauses.
"""

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

CLAUSE_WORDS = ["x SETTINGS y", "x LIMIT y", "x OFFSET y", "x ORDER BY y"]


@pytest.mark.parametrize("name", CLAUSE_WORDS)
def test_sort_then_head_on_column_named_like_a_clause(name):
    def chain(df):
        return df.sort_values(name, kind="stable").head(3)

    pdf = pd.DataFrame({name: [2, 1, 2, 1, 3], "v": [0, 1, 2, 3, 4]})
    assert_datastore_equals_pandas(chain(DataStore(pdf.copy())), chain(pdf))


@pytest.mark.parametrize("word", CLAUSE_WORDS)
def test_filter_on_literal_like_a_clause_then_head(word):
    def chain(df):
        return df[df["s"] != word].head(2)

    pdf = pd.DataFrame({"s": [word, "a", word, "b", "c"], "v": [0, 1, 2, 3, 4]})
    assert_datastore_equals_pandas(chain(DataStore(pdf.copy())), chain(pdf))


@pytest.mark.parametrize("word", CLAUSE_WORDS)
def test_sort_by_comparison_with_literal_like_a_clause(word):
    def chain(df):
        df = df.assign(hit=df["s"] == word)
        return df.sort_values(["hit", "v"], ascending=[False, True]).head(2)

    pdf = pd.DataFrame({"s": ["a", word, "b", word], "v": [3, 2, 1, 0]})
    assert_datastore_equals_pandas(chain(DataStore(pdf.copy())), chain(pdf))
