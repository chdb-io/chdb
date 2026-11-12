"""
Column values read the state they were taken from, as pandas does, without
keeping the data of earlier states alive once the DataStore has moved on.
"""

import gc
import itertools
import weakref

import numpy as np
import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas

_counter = itertools.count()


@pytest.fixture(params=["dataframe", "file"])
def source(request, tmp_path):
    """Build a DataStore over ``df`` from the parametrized kind of source."""

    def make(df):
        if request.param == "dataframe":
            return DataStore(df.copy())
        path = str(tmp_path / f"t{next(_counter)}.parquet")
        df.to_parquet(path)
        return DataStore.from_file(path)

    return make


class TestCheckpointReleasesHistory:
    """Executing a pipeline releases the data of the ops it replaced."""

    @pytest.mark.parametrize("value", ["array", "expression"])
    def test_repeated_overwrite_keeps_no_earlier_data(self, source, value):
        df = pd.DataFrame({"a": np.arange(1000)})
        d = source(df)
        expected = df.copy()
        rounds = []
        for i in range(4):
            refs = []
            if value == "array":
                values = np.full(len(df), i)
                refs.append(weakref.ref(values))
                d["c"] = values
                del values
                expected["c"] = np.full(len(df), i)
            else:
                d["c"] = d["a"] + i
                expected["c"] = expected["a"] + i
            assert_datastore_equals_pandas(d, expected)
            # The DataFrame the pipeline was checkpointed to this round
            result = d.to_pandas()
            refs.append(weakref.ref(result))
            del result
            rounds.append(refs)
        gc.collect()
        alive = [i for i, refs in enumerate(rounds[:-1]) if any(r() is not None for r in refs)]
        assert alive == []

    def test_source_dataframe_released_after_checkpoint(self):
        df = pd.DataFrame({"a": np.arange(1000)})
        source_ref = weakref.ref(df)
        d = DataStore(df)
        del df
        for i in range(3):
            d["c"] = np.full(1000, i)
            assert len(d.to_pandas()) == 1000
        gc.collect()
        assert source_ref() is None
