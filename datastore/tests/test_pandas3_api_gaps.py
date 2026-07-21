"""
Tests for pandas 3.0 API-parity additions:

- Series.str.isascii()
- Series.str.replace({pat: repl}) dictionary form
- Series.map(func, **kwargs) keyword passthrough
- datastore.read_iceberg / DataStore.to_iceberg delegation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import datastore
from datastore import DataStore

PANDAS_3 = int(pd.__version__.split(".")[0]) >= 3


class TestStrIsascii:
    DATA = ["hello", "héllo", "中文", "", "mixed中x", "ascii only!"]

    def test_isascii_matches_python_semantics(self):
        pd_df = pd.DataFrame({"s": self.DATA})
        ds = DataStore.from_df(pd_df)

        result = [bool(v) for v in ds["s"].str.isascii()]

        assert result == [v.isascii() for v in self.DATA]

    @pytest.mark.skipif(not PANDAS_3, reason="Series.str.isascii is pandas 3.0+")
    def test_isascii_mirrors_pandas(self):
        # pandas operations
        pd_df = pd.DataFrame({"s": self.DATA})
        pd_result = pd_df["s"].str.isascii()

        # DataStore operations (mirror of pandas)
        ds = DataStore.from_df(pd_df)
        ds_result = ds["s"].str.isascii()

        assert [bool(v) for v in ds_result] == list(pd_result)

    def test_isascii_null_propagates(self):
        # object dtype keeps None handling engine-agnostic
        pd_df = pd.DataFrame({"s": pd.Series(["ok", None, "汉"], dtype=object)})
        ds = DataStore.from_df(pd_df)

        result = list(ds["s"].str.isascii())

        assert bool(result[0]) is True
        assert pd.isna(result[1])
        assert bool(result[2]) is False


class TestStrReplaceDict:
    DATA = ["cat hat", "bat", "no match", ""]

    @staticmethod
    def _python_chain_replace(values, mapping):
        out = []
        for v in values:
            for pat, repl in mapping.items():
                v = v.replace(pat, repl)
            out.append(v)
        return out

    def test_replace_dict_values(self):
        mapping = {"cat": "dog", "hat": "cap"}
        pd_df = pd.DataFrame({"s": self.DATA})
        ds = DataStore.from_df(pd_df)

        result = list(ds["s"].str.replace(mapping))

        assert result == self._python_chain_replace(self.DATA, mapping)

    @pytest.mark.skipif(not PANDAS_3, reason="dict pat form is pandas 3.0+")
    def test_replace_dict_mirrors_pandas(self):
        mapping = {"cat": "dog", "hat": "cap"}

        # pandas operations
        pd_df = pd.DataFrame({"s": self.DATA})
        pd_result = pd_df["s"].str.replace(mapping)

        # DataStore operations (mirror of pandas)
        ds = DataStore.from_df(pd_df)
        ds_result = ds["s"].str.replace(mapping)

        assert list(ds_result) == list(pd_result)

    @pytest.mark.skipif(not PANDAS_3, reason="dict pat form is pandas 3.0+")
    def test_replace_dict_sequential_application_mirrors_pandas(self):
        # First pair's output feeds the second pair: order semantics must match.
        mapping = {"a": "b", "b": "c"}
        data = ["aba", "abc"]

        pd_result = pd.DataFrame({"s": data})["s"].str.replace(mapping)

        ds = DataStore.from_df(pd.DataFrame({"s": data}))
        ds_result = ds["s"].str.replace(mapping)

        assert list(ds_result) == list(pd_result)

    def test_replace_dict_with_repl_raises(self):
        ds = DataStore.from_df(pd.DataFrame({"s": self.DATA}))

        with pytest.raises(ValueError, match="dictionary"):
            ds["s"].str.replace({"cat": "dog"}, "oops")

    def test_replace_dict_empty_is_identity(self):
        ds = DataStore.from_df(pd.DataFrame({"s": self.DATA}))

        result = list(ds["s"].str.replace({}))

        assert result == self.DATA

    def test_replace_two_arg_form_unchanged(self):
        # pandas operations
        pd_df = pd.DataFrame({"s": self.DATA})
        pd_result = pd_df["s"].str.replace("at", "AT")

        # DataStore operations (mirror of pandas)
        ds = DataStore.from_df(pd_df)
        ds_result = ds["s"].str.replace("at", "AT")

        assert list(ds_result) == list(pd_result)

    def test_replace_regex_form_unchanged(self):
        pd_df = pd.DataFrame({"s": self.DATA})
        pd_result = pd_df["s"].str.replace("[ch]at", "X", regex=True)

        ds = DataStore.from_df(pd_df)
        ds_result = ds["s"].str.replace("[ch]at", "X", regex=True)

        assert list(ds_result) == list(pd_result)


class TestMapKwargs:
    @pytest.mark.skipif(not PANDAS_3, reason="Series.map(**kwargs) is pandas 3.0+")
    def test_map_kwargs_mirrors_pandas(self):
        # pandas operations
        pd_df = pd.DataFrame({"a": [1, 2, 3]})
        pd_result = pd_df["a"].map(lambda v, add: v + add, add=10)

        # DataStore operations (mirror of pandas)
        ds = DataStore.from_df(pd_df)
        ds_result = ds["a"].map(lambda v, add: v + add, add=10)

        assert list(ds_result) == list(pd_result)
        assert list(ds_result) == [11, 12, 13]

    def test_map_dict_and_na_action_unchanged(self):
        pd_df = pd.DataFrame({"g": ["A", "B", "A"]})
        pd_result = pd_df["g"].map({"A": 4.0, "B": 3.0})

        ds = DataStore.from_df(pd_df)
        ds_result = ds["g"].map({"A": 4.0, "B": 3.0})

        assert list(ds_result) == list(pd_result)


class TestIcebergDelegation:
    """read_iceberg/to_iceberg delegate to pandas' PyIceberg-backed IO.

    The pandas functions are monkeypatched so the delegation contract
    (argument passthrough, DataStore wrapping) is tested without a live
    Iceberg catalog.
    """

    def test_read_iceberg_wraps_result_and_passes_args(self, monkeypatch):
        source_df = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        captured = {}

        def fake_read_iceberg(table_identifier, catalog_name=None, **kwargs):
            captured["table_identifier"] = table_identifier
            captured["catalog_name"] = catalog_name
            captured["kwargs"] = kwargs
            return source_df.copy()

        monkeypatch.setattr(pd, "read_iceberg", fake_read_iceberg, raising=False)

        ds = datastore.read_iceberg(
            "ns.orders", catalog_name="cat", snapshot_id=42, limit=10
        )

        assert isinstance(ds, DataStore)
        assert captured["table_identifier"] == "ns.orders"
        assert captured["catalog_name"] == "cat"
        assert captured["kwargs"] == {"snapshot_id": 42, "limit": 10}
        result_df = ds.to_df()
        assert list(result_df.columns) == ["x", "y"]
        assert list(result_df["x"]) == [1, 2]
        assert list(result_df["y"]) == ["a", "b"]

    def test_to_iceberg_delegates_materialized_frame(self, monkeypatch):
        captured = {}

        def fake_to_iceberg(self, table_identifier, catalog_name=None, **kwargs):
            captured["frame"] = self.copy()
            captured["table_identifier"] = table_identifier
            captured["catalog_name"] = catalog_name
            captured["kwargs"] = kwargs
            return None

        monkeypatch.setattr(pd.DataFrame, "to_iceberg", fake_to_iceberg, raising=False)

        source_df = pd.DataFrame({"x": [1, 2, 3]})
        ds = DataStore.from_df(source_df)
        ds.to_iceberg("ns.orders", catalog_name="cat", append=True)

        assert captured["table_identifier"] == "ns.orders"
        assert captured["catalog_name"] == "cat"
        assert captured["kwargs"] == {"append": True}
        assert list(captured["frame"]["x"]) == [1, 2, 3]

    @pytest.mark.skipif(PANDAS_3, reason="error contract only applies before pandas 3.0")
    def test_read_iceberg_requires_pandas3(self):
        with pytest.raises(NotImplementedError, match="pandas >= 3.0"):
            datastore.read_iceberg("ns.orders")

    @pytest.mark.skipif(PANDAS_3, reason="error contract only applies before pandas 3.0")
    def test_to_iceberg_requires_pandas3(self):
        ds = DataStore.from_df(pd.DataFrame({"x": [1]}))
        with pytest.raises(NotImplementedError, match="pandas >= 3.0"):
            ds.to_iceberg("ns.orders")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
