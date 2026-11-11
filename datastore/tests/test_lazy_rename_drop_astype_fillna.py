"""
rename(), drop(), astype() and fillna() must stay lazy (issue #699).

They used to call _get_df() while the pipeline was being built. As lazy ops
they now run as a pandas segment between SQL segments, so the chains below
also cover the multi-segment paths that this exposed: per-segment SELECT *,
from_df column baking, column pruning and empty intermediate results.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from datastore import DataStore
from tests.test_utils import assert_datastore_equals_pandas


def _frame():
    return pd.DataFrame(
        {
            'a': [1.0, 2.0, None, 4.0, 5.0],
            'b': [1.5, None, 3.0, 4.0, None],
            'c': ['x', 'y', 'z', None, 'w'],
            'd': [10, 20, 30, 40, 50],
        }
    )


@pytest.fixture
def parquet(tmp_path):
    path = str(tmp_path / 't.parquet')
    _frame().to_parquet(path, index=False)
    return path, pd.read_parquet(path)


@pytest.fixture(params=['file', 'from_df', 'ctor'])
def source(request, parquet):
    path, pd_df = parquet
    if request.param == 'file':
        return DataStore.from_file(path), pd_df.copy()
    if request.param == 'from_df':
        return DataStore.from_df(pd_df.copy()), pd_df.copy()
    return DataStore(pd_df.copy()), pd_df.copy()


BUILDERS = {
    'rename': lambda df: df.rename(columns={'a': 'aa'}),
    'drop': lambda df: df.drop(columns=['a']),
    'astype': lambda df: df.astype({'a': 'float64'}),
    'fillna': lambda df: df.fillna(0),
    'fillna_dict': lambda df: df.fillna({'a': 0}),
}


class TestStaysLazy:
    @pytest.mark.parametrize('name', sorted(BUILDERS))
    def test_no_execute_while_building(self, parquet, name):
        path, pd_df = parquet
        ds = DataStore.from_file(path)
        build = BUILDERS[name]

        with patch.object(DataStore, '_execute', autospec=True, side_effect=DataStore._execute) as mock_exec:
            ds_result = build(ds)
            assert mock_exec.call_count == 0

        assert isinstance(ds_result, DataStore)
        assert_datastore_equals_pandas(ds_result, build(pd_df))

    def test_chain_builds_without_execute(self, parquet):
        path, pd_df = parquet
        ds = DataStore.from_file(path)

        with patch.object(DataStore, '_execute', autospec=True, side_effect=DataStore._execute) as mock_exec:
            ds_result = ds.rename(columns={'a': 'aa'}).drop(columns=['c']).fillna(0).astype({'aa': 'int64'})
            ds_result = ds_result[ds_result['aa'] > 1]
            assert mock_exec.call_count == 0

        pd_result = pd_df.rename(columns={'a': 'aa'}).drop(columns=['c']).fillna(0).astype({'aa': 'int64'})
        pd_result = pd_result[pd_result['aa'] > 1]
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestChains:
    def test_rename_then_filter_on_new_name(self, source):
        ds, pd_df = source
        ds_result = ds.rename(columns={'a': 'aa'})
        ds_result = ds_result[ds_result['aa'] > 1].sort_values('d', ascending=False)
        pd_result = pd_df.rename(columns={'a': 'aa'})
        pd_result = pd_result[pd_result['aa'] > 1].sort_values('d', ascending=False)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_variants(self, source):
        ds, pd_df = source
        assert_datastore_equals_pandas(ds.rename(columns=str.upper), pd_df.rename(columns=str.upper))
        assert_datastore_equals_pandas(ds.rename({'a': 'A'}, axis=1), pd_df.rename({'a': 'A'}, axis=1))
        assert_datastore_equals_pandas(
            ds.rename(index={0: 100}), pd_df.rename(index={0: 100}), check_index=True
        )

    def test_drop_variants(self, source):
        ds, pd_df = source
        assert_datastore_equals_pandas(ds.drop('a', axis=1), pd_df.drop('a', axis=1))
        assert_datastore_equals_pandas(ds.drop(['a', 'b'], axis='columns'), pd_df.drop(['a', 'b'], axis='columns'))
        assert_datastore_equals_pandas(ds.drop([0, 2]), pd_df.drop([0, 2]), check_index=True)
        assert_datastore_equals_pandas(
            ds.drop(columns=['zz'], errors='ignore'), pd_df.drop(columns=['zz'], errors='ignore')
        )

    def test_drop_rows_then_assign_keeps_index(self, source):
        ds, pd_df = source
        ds_result = ds.drop([0, 2])
        ds_result = ds_result.assign(e=ds_result['d'] + 1)
        pd_result = pd_df.drop([0, 2])
        pd_result = pd_result.assign(e=pd_result['d'] + 1)
        assert_datastore_equals_pandas(ds_result, pd_result, check_index=True)

    def test_filter_assign_rename_select(self, source):
        ds, pd_df = source
        ds_result = ds[ds['d'] > 15]
        ds_result = ds_result.assign(e=ds_result['d'] + ds_result['a'])
        ds_result = ds_result.rename(columns={'e': 'total'})[['d', 'total']]
        pd_result = pd_df[pd_df['d'] > 15]
        pd_result = pd_result.assign(e=pd_result['d'] + pd_result['a'])
        pd_result = pd_result.rename(columns={'e': 'total'})[['d', 'total']]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_rename_then_assign(self, source):
        ds, pd_df = source
        ds_result = ds.rename(columns={'c': 'cc'})
        ds_result = ds_result.assign(x=ds_result['d'] * 2)
        pd_result = pd_df.rename(columns={'c': 'cc'})
        pd_result = pd_result.assign(x=pd_result['d'] * 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_drop_then_assign(self, source):
        ds, pd_df = source
        ds_result = ds.drop(columns=['c'])
        ds_result = ds_result.assign(x=ds_result['d'] * 2)
        pd_result = pd_df.drop(columns=['c'])
        pd_result = pd_result.assign(x=pd_result['d'] * 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_select_fillna_drop_assign(self, source):
        ds, pd_df = source
        ds_result = ds[['a', 'b', 'd']].fillna(0).drop(columns=['b'])
        ds_result = ds_result.assign(x=ds_result['a'] * 2)
        pd_result = pd_df[['a', 'b', 'd']].fillna(0).drop(columns=['b'])
        pd_result = pd_result.assign(x=pd_result['a'] * 2)
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_empty_filter_then_drop_then_head(self, source):
        ds, pd_df = source
        ds_result = ds[ds['d'] > 100].drop(columns=['a']).head(3)
        pd_result = pd_df[pd_df['d'] > 100].drop(columns=['a']).head(3)
        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_empty_filter_then_rename(self, source):
        ds, pd_df = source
        ds_result = ds[ds['d'] > 100].rename(columns={'a': 'aa'})
        pd_result = pd_df[pd_df['d'] > 100].rename(columns={'a': 'aa'})
        assert len(ds_result) == 0
        assert list(ds_result.columns) == list(pd_result.columns)

    def test_filter_fillna_filter(self, source):
        ds, pd_df = source
        ds_result = ds[ds['d'] > 10].fillna({'b': 0})
        ds_result = ds_result[ds_result['b'] >= 0]
        pd_result = pd_df[pd_df['d'] > 10].fillna({'b': 0})
        pd_result = pd_result[pd_result['b'] >= 0]
        assert_datastore_equals_pandas(ds_result, pd_result)

    def test_fillna_variants(self, source):
        ds, pd_df = source
        assert_datastore_equals_pandas(ds.fillna(0, limit=1), pd_df.fillna(0, limit=1))
        fill = pd.Series({'a': -1.0, 'b': -2.0})
        assert_datastore_equals_pandas(ds.fillna(fill), pd_df.fillna(fill))
        assert_datastore_equals_pandas(ds.fillna(method='ffill'), pd_df.ffill())

    def test_rename_then_groupby(self, source):
        ds, pd_df = source
        ds_result = ds.rename(columns={'c': 'cc'}).groupby('cc')['d'].sum()
        pd_result = pd_df.rename(columns={'c': 'cc'}).groupby('cc')['d'].sum()
        assert_datastore_equals_pandas(ds_result, pd_result)


class TestColumnAccessDtypes:
    """Column access after a pandas op must not round-trip through SQL."""

    @pytest.mark.parametrize(
        'build, col',
        [
            (lambda df: df.astype({'c': 'category'}), 'c'),
            (lambda df: df.astype({'d': object}), 'd'),
            (lambda df: df.astype({'d': 'int32'}), 'd'),
            (lambda df: df.fillna(0), 'c'),
            (lambda df: df.fillna({'c': 'q'}), 'c'),
            (lambda df: df.rename(columns={'a': 'aa'}), 'aa'),
        ],
    )
    def test_column_after_op(self, source, build, col):
        ds, pd_df = source
        ds_col = build(ds)[col]
        pd_col = build(pd_df)[col]
        assert_datastore_equals_pandas(ds_col, pd_col)
        assert ds_col.to_pandas().dtype == pd_col.dtype


class TestErrors:
    def test_fillna_rejects_bad_value_at_call_time(self, source):
        ds, pd_df = source
        with pytest.raises(TypeError):
            pd_df.fillna(value=[1, 2])
        with pytest.raises(TypeError):
            ds.fillna(value=[1, 2])
        with pytest.raises(ValueError):
            ds.fillna(method='nearest')

    def test_missing_label_raises_on_execution(self, source):
        ds, pd_df = source
        with pytest.raises(KeyError):
            pd_df.drop(columns=['zz'])
        with pytest.raises(KeyError):
            ds.drop(columns=['zz']).to_pandas()
        with pytest.raises(KeyError):
            ds.rename(columns={'zz': 'z'}, errors='raise').to_pandas()

    def test_inplace_still_rejected(self, source):
        ds, _ = source
        with pytest.raises(Exception, match='inplace'):
            ds.drop(columns=['a'], inplace=True)
        with pytest.raises(Exception, match='inplace'):
            ds.rename(columns={'a': 'aa'}, inplace=True)
        with pytest.raises(Exception, match='inplace'):
            ds.fillna(0, inplace=True)


def test_explain_lists_the_ops(parquet, capsys):
    path, pd_df = parquet
    ds = DataStore.from_file(path)
    ds_result = ds.rename(columns={'a': 'aa'}).drop(columns=['b']).fillna(0).astype({'d': 'float64'})
    ds_result.explain()
    plan = capsys.readouterr().out
    for text in ("Rename columns={'a': 'aa'}", "Drop columns=['b']", 'Fill NA with value 0', 'Cast types'):
        assert text in plan
    pd_result = pd_df.rename(columns={'a': 'aa'}).drop(columns=['b']).fillna(0).astype({'d': 'float64'})
    assert_datastore_equals_pandas(ds_result, pd_result)
