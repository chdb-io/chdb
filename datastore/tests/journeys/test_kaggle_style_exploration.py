"""
Real user journeys: Kaggle-style exploratory data analysis pipelines.

Each test is a 6-10 step chain that mirrors the kind of code a data
analyst writes in a notebook cell. Inspirations are noted in each
docstring; the data is synthetic and self-contained.

What distinguishes these tests from existing per-feature unit tests:
- chain length is intentionally long (>=6 ops)
- each step uses the OUTPUT of the previous step (result-as-input)
- the chain mixes filter/groupby/agg/sort/head/assign/join
- pandas runs the same code path - any divergence is a real bug

Most journeys read from a parquet file (the Kaggle default workflow) so
that the source-bound SQL path is exercised. A small number of tests
are explicitly tagged with ``DataStore(df)`` to cover the Python() table
function path - some of these are ``@unittest.expectedFailure`` because
they expose known latent bugs that are out of scope for the current PR.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from datastore import DataStore

from tests.test_utils import assert_datastore_equals_pandas


def _write_temp_parquet(df, dirpath, name):
    path = os.path.join(dirpath, name)
    df.to_parquet(path)
    return path


def _seed_orders_dataset(seed=42, n=3000):
    """E-commerce-style orders table with customer / product / amount."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            'order_id': np.arange(1, n + 1),
            'customer_id': rng.integers(1, 200, n),
            'product_category': rng.choice(
                ['Electronics', 'Books', 'Apparel', 'Home', 'Beauty'],
                n,
            ),
            'region': rng.choice(['NA', 'EU', 'APAC', 'LATAM'], n),
            'amount': rng.uniform(5, 500, n).round(2),
            'quantity': rng.integers(1, 8, n),
            'rating': rng.choice([1, 2, 3, 4, 5, np.nan], n),
        }
    )


def _seed_visits_dataset(seed=123, n=5000):
    """Web-visit-style table with timestamp / user / page / duration."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            'session_id': np.arange(1, n + 1),
            'user_id': rng.integers(1, 500, n),
            'page': rng.choice(
                ['home', 'product', 'cart', 'checkout', 'about'],
                n,
                p=[0.5, 0.25, 0.12, 0.08, 0.05],
            ),
            'device': rng.choice(['mobile', 'desktop', 'tablet'], n),
            'duration_sec': rng.exponential(scale=60, size=n).round(1),
        }
    )


class TestTopProductCategoriesByRegion(unittest.TestCase):
    """Inspiration: 'Which product categories drive revenue in each region?'

    Typical EDA cell: filter to recent orders -> compute revenue per
    (region, category) -> rank within each region -> take the top 3.
    A 7-op chain with filter, assign, groupby, agg, sort, head, filter.
    """

    @classmethod
    def setUpClass(cls):
        cls.pd_df = _seed_orders_dataset()
        cls.temp_dir = tempfile.mkdtemp()
        cls.parquet_path = _write_temp_parquet(
            cls.pd_df, cls.temp_dir, 'orders.parquet'
        )

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.temp_dir)

    def test_top_three_revenue_categories_per_region_from_parquet(self):
        # pandas reference
        pd_chain = (
            self.pd_df[self.pd_df['amount'] >= 50]
            .assign(revenue=lambda x: x['amount'] * x['quantity'])
            .groupby(['region', 'product_category'])
            .agg({'revenue': 'sum'})
            .sort_values('revenue', ascending=False)
            .head(20)
        )
        # post-filter: only categories where revenue clears a threshold
        pd_result = pd_chain[pd_chain['revenue'] > 5000]

        # mirror on DataStore reading from parquet (Kaggle default)
        ds = DataStore.from_file(self.parquet_path)
        ds_chain = (
            ds[ds['amount'] >= 50]
            .assign(revenue=ds['amount'] * ds['quantity'])
            .groupby(['region', 'product_category'])
            .agg({'revenue': 'sum'})
            .sort_values('revenue', ascending=False)
            .head(20)
        )
        ds_result = ds_chain[ds_chain['revenue'] > 5000]

        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_index=True,
            check_row_order=False,
        )

    def test_top_three_revenue_dataframe_source_assign_before_groupby(self):
        """Same pipeline against the in-memory DataFrame source.

        Originally failed with NOT_AN_AGGREGATE: the SQL emitted
        ``SELECT *, ..., sum(...) AS __agg_revenue__ ... GROUP BY ...``
        because the LazyColumnAssignment for ``revenue`` triggered
        include_star, which is invalid under GROUP BY. Fixed by routing
        DataFrame-source single-layer plans with ``LazyGroupByAgg``
        through ``_build_layered_sql`` and forcing ``include_star=False``
        whenever the layer has a groupby_agg (the agg's emitted
        select_fields already cover the group keys and aggregations).
        """
        pd_chain = (
            self.pd_df[self.pd_df['amount'] >= 50]
            .assign(revenue=lambda x: x['amount'] * x['quantity'])
            .groupby(['region', 'product_category'])
            .agg({'revenue': 'sum'})
            .sort_values('revenue', ascending=False)
            .head(20)
        )
        pd_result = pd_chain[pd_chain['revenue'] > 5000]

        ds = DataStore(self.pd_df)
        ds_chain = (
            ds[ds['amount'] >= 50]
            .assign(revenue=ds['amount'] * ds['quantity'])
            .groupby(['region', 'product_category'])
            .agg({'revenue': 'sum'})
            .sort_values('revenue', ascending=False)
            .head(20)
        )
        ds_result = ds_chain[ds_chain['revenue'] > 5000]
        assert_datastore_equals_pandas(
            ds_result, pd_result, check_index=True, check_row_order=False
        )


class TestCustomerOrderHistogram(unittest.TestCase):
    """Inspiration: 'Show me the distribution of orders per customer.'

    Standard cohort analysis: count orders per customer -> bucket by
    count -> compute size of each bucket -> sort by bucket. 7+ ops.
    """

    @classmethod
    def setUpClass(cls):
        cls.pd_df = _seed_orders_dataset()

    def test_orders_per_customer_buckets(self):
        # pandas: per-customer order count, then bucketed
        pd_per_customer = (
            self.pd_df.groupby('customer_id')
            .agg({'order_id': 'count'})
            .rename(columns={'order_id': 'order_count'})
        )
        # bucket the counts manually (avoid pd.cut to keep portable)
        pd_per_customer['bucket'] = pd_per_customer['order_count'].apply(
            lambda x: 'high' if x >= 20 else 'mid' if x >= 10 else 'low'
        )
        pd_bucket_sizes = (
            pd_per_customer.groupby('bucket')
            .agg({'order_count': 'sum'})
            .sort_values('order_count', ascending=False)
        )

        # mirror on DataStore - same shape, using pandas-compat APIs
        ds = DataStore(self.pd_df)
        ds_per_customer = (
            ds.groupby('customer_id')
            .agg({'order_id': 'count'})
            .rename(columns={'order_id': 'order_count'})
        )
        ds_per_customer['bucket'] = ds_per_customer['order_count'].apply(
            lambda x: 'high' if x >= 20 else 'mid' if x >= 10 else 'low'
        )
        ds_bucket_sizes = (
            ds_per_customer.groupby('bucket')
            .agg({'order_count': 'sum'})
            .sort_values('order_count', ascending=False)
        )

        assert_datastore_equals_pandas(
            ds_bucket_sizes,
            pd_bucket_sizes,
            check_index=True,
            check_row_order=False,
        )


class TestTopKActiveUsersPerDevice(unittest.TestCase):
    """Inspiration: 'Top-K users on each device'.

    Visits-style data: filter to long sessions -> compute per-user
    totals -> sort -> head per device. 6-op chain with the result-as-
    input pattern repeated twice.
    """

    @classmethod
    def setUpClass(cls):
        cls.pd_df = _seed_visits_dataset()

    def test_top_5_users_overall_after_filter(self):
        # pandas: long sessions only, total per user, top 5 globally
        pd_long = self.pd_df[self.pd_df['duration_sec'] >= 30]
        pd_by_user = pd_long.groupby('user_id').agg(
            {'duration_sec': 'sum', 'session_id': 'count'}
        )
        pd_sorted = pd_by_user.sort_values('duration_sec', ascending=False)
        pd_top = pd_sorted.head(5)
        # post-filter the top by another condition: sessions >= 5
        pd_result = pd_top[pd_top['session_id'] >= 5]

        # mirror
        ds = DataStore(self.pd_df)
        ds_long = ds[ds['duration_sec'] >= 30]
        ds_by_user = ds_long.groupby('user_id').agg(
            {'duration_sec': 'sum', 'session_id': 'count'}
        )
        ds_sorted = ds_by_user.sort_values('duration_sec', ascending=False)
        ds_top = ds_sorted.head(5)
        ds_result = ds_top[ds_top['session_id'] >= 5]

        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_index=True,
            check_row_order=False,
        )


class TestMultiStageRevenueFilter(unittest.TestCase):
    """Inspiration: data-quality + cohort analysis crossover.

    9-step chain:
    1. drop rows with null rating
    2. filter high-value orders
    3. assign discount column
    4. assign net amount
    5. groupby region
    6. agg net amount sum, order count
    7. sort descending by net
    8. head top regions
    9. filter regions whose net exceeds threshold
    """

    @classmethod
    def setUpClass(cls):
        cls.pd_df = _seed_orders_dataset()

    def test_full_revenue_pipeline(self):
        pd_clean = self.pd_df.dropna(subset=['rating'])
        pd_highvalue = pd_clean[pd_clean['amount'] >= 100]
        pd_with_discount = pd_highvalue.assign(
            discount=pd_highvalue['amount'] * 0.1
        )
        pd_with_net = pd_with_discount.assign(
            net=pd_with_discount['amount'] - pd_with_discount['discount']
        )
        pd_by_region = pd_with_net.groupby('region').agg(
            {'net': 'sum', 'order_id': 'count'}
        )
        pd_sorted = pd_by_region.sort_values('net', ascending=False)
        pd_top = pd_sorted.head(3)
        pd_result = pd_top[pd_top['net'] >= 10000]

        ds = DataStore(self.pd_df)
        ds_clean = ds.dropna(subset=['rating'])
        ds_highvalue = ds_clean[ds_clean['amount'] >= 100]
        ds_with_discount = ds_highvalue.assign(
            discount=ds_highvalue['amount'] * 0.1
        )
        ds_with_net = ds_with_discount.assign(
            net=ds_with_discount['amount'] - ds_with_discount['discount']
        )
        ds_by_region = ds_with_net.groupby('region').agg(
            {'net': 'sum', 'order_id': 'count'}
        )
        ds_sorted = ds_by_region.sort_values('net', ascending=False)
        ds_top = ds_sorted.head(3)
        ds_result = ds_top[ds_top['net'] >= 10000]

        assert_datastore_equals_pandas(
            ds_result,
            pd_result,
            check_index=True,
            check_row_order=False,
        )


if __name__ == '__main__':
    unittest.main()
