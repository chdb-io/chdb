"""
Verbatim journey regressions for falsifying examples surfaced by
``tests/test_property_based_chains.py``.

The cursor rule for property-based sweeps (see
``.cursor/rules/chdb-ds.mdc`` rule #6) requires that every hypothesis
falsifier becomes a permanent verbatim regression here, marked
``@unittest.expectedFailure`` while the underlying bug is open. The
property test then skips the exact shape (referencing this file by
name) so the sweep stays green and the bug stays tracked.

When the bug is fixed, flip the ``expectedFailure`` and drop the
corresponding skip clause in the property test in the same commit.
"""

import unittest

import numpy as np
import pandas as pd

from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


def _dataset():
    """Same fixed schema/seed as
    ``tests/test_property_based_chains.py::_dataset`` so the falsifying
    examples reproduce verbatim."""
    rng = np.random.default_rng(20260)
    n = 400
    return pd.DataFrame(
        {
            'cat': rng.choice(['A', 'B', 'C', 'D'], n),
            'sub': rng.choice(['x', 'y', 'z'], n),
            'v': rng.integers(0, 100, n),
            'w': rng.uniform(0, 100, n).round(3),
        }
    )


class TestEmptyGroupbyColumnDrift(unittest.TestCase):
    """Chain ``filter(v>0) -> filter(v>99) -> sort(w desc) ->
    groupby(cat).agg({'v':'sum'})`` produces an empty intermediate
    after the second filter, then aggregates an empty input. Pandas
    returns an empty DataFrame with just the aggregated column (``v``)
    and an empty group-key index. DataStore's executor short-circuits
    once an upstream segment returns zero rows and surfaces the
    source-bound intermediate columns (``['sub', 'v', 'w']``) instead
    of the agg's projected output.

    Discovered by Hypothesis in CI (PR #578, commit 64f38b8). Marked
    ``expectedFailure``; flip when the executor's empty-intermediate
    handling projects to the agg's output columns regardless of
    short-circuit.
    """

    @unittest.expectedFailure
    def test_empty_filter_then_sort_then_agg_drops_non_agg_columns(self):
        df = _dataset()

        pd_result = (
            df[df['v'] > 0][df[df['v'] > 0]['v'] > 99]
            .sort_values('w', ascending=False)
            .groupby('cat')
            .agg({'v': 'sum'})
        )

        ds = DataStore(df)
        ds_filt1 = ds[ds['v'] > 0]
        ds_filt2 = ds_filt1[ds_filt1['v'] > 99]
        ds_result = (
            ds_filt2.sort_values('w', ascending=False)
            .groupby('cat')
            .agg({'v': 'sum'})
        )

        assert_datastore_equals_pandas(
            ds_result, pd_result, check_index=True
        )


if __name__ == '__main__':
    unittest.main()
