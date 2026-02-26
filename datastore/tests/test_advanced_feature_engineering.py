"""
Advanced ML Feature Engineering Tests - Mixed Column Reference Styles
======================================================================

Test objectives:
1. Mix three column reference styles: ds.col, ds["col"], col("name")
2. Use advanced ClickHouse functions: quantile, window functions, topK, argMin/argMax
3. Discover deep design and implementation issues
4. Verify data consistency in complex scenarios

Functions tested:
- Statistical: stddevSamp, varSamp, skewPop, kurtPop, corr, covarPop
- Quantile: quantile(0.25/0.5/0.75)
- Window: row_number, dense_rank, percent_rank, ntile, lag, lead
- TopK: topK, topKWeighted
- ArgMin/ArgMax: argMin, argMax
- String: length, multiSearchAny, splitByWhitespace
- DateTime: toYear, toMonth, dateDiff
- Funnel: windowFunnel
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore, col, F, Field
from datastore.column_expr import ColumnExpr


class TestAdvancedFeatureEngineering:
    """Advanced feature engineering tests using mixed column reference styles"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create test data"""
        np.random.seed(42)
        n_users = 500

        self.df = pd.DataFrame(
            {
                'user_id': range(1, n_users + 1),
                'signup_date': pd.date_range('2023-01-01', periods=n_users, freq='h'),
                'page_views': np.random.poisson(50, n_users),
                'session_duration': np.random.exponential(300, n_users),
                'clicks': np.random.poisson(20, n_users),
                'cart_additions': np.random.poisson(5, n_users),
                'purchases': np.random.poisson(2, n_users),
                'purchase_amount': np.random.exponential(100, n_users),
                'days_since_last_visit': np.random.exponential(7, n_users),
                'age': np.random.normal(35, 12, n_users).clip(18, 80).astype(int),
                'loyalty_points': np.random.poisson(1000, n_users),
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_users, p=[0.6, 0.3, 0.1]),
                'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'JP'], n_users),
                'referrer': np.random.choice(['google', 'facebook', 'direct', 'email'], n_users),
                'last_search_query': np.random.choice(
                    [
                        'laptop computer',
                        'running shoes',
                        'winter jacket',
                        'headphones wireless',
                        'coffee maker',
                        'yoga mat',
                    ],
                    n_users,
                ),
                'churned': np.random.choice([0, 1], n_users, p=[0.85, 0.15]),
            }
        )
        self.ds = DataStore.from_df(self.df)

    # ========== Statistical Aggregations ==========

    def test_statistical_aggregations_stddev_var(self):
        """
        Test stddevSamp and varSamp using mixed reference styles.

        Functions: stddev_samp(), var_samp()
        """
        # Use col() for aggregation expressions
        stats = (
            self.ds.groupby('device_type')
            .agg(
                std_views=col('page_views').stddev_samp(),
                var_views=col('page_views').var_samp(),
                std_amount=col('purchase_amount').stddev_samp(),
                var_amount=col('purchase_amount').var_samp(),
            )
            .to_df()
        )

        assert 'std_views' in stats.columns
        assert 'var_views' in stats.columns
        # stddev and variance must be non-negative
        assert all(stats['std_views'] >= 0)
        assert all(stats['var_views'] >= 0)
        # variance = stddev^2 (approximately)
        np.testing.assert_array_almost_equal(
            stats['var_views'].values,
            stats['std_views'].values ** 2,
            decimal=4,
        )

    def test_statistical_aggregations_skew_kurt(self):
        """
        Test skewPop and kurtPop using mixed reference styles.

        Functions: skew_pop(), kurt_pop()
        """
        stats = (
            self.ds.groupby('device_type')
            .agg(
                skew_views=col('page_views').skew_pop(),
                kurt_views=col('page_views').kurt_pop(),
                skew_amount=col('purchase_amount').skew_pop(),
                kurt_amount=col('purchase_amount').kurt_pop(),
            )
            .to_df()
        )

        assert 'skew_views' in stats.columns
        assert 'kurt_views' in stats.columns
        assert 'skew_amount' in stats.columns
        assert 'kurt_amount' in stats.columns
        # Kurtosis for normal distribution is ~3, exponential is ~6
        # Just verify values are finite
        assert all(np.isfinite(stats['skew_views']))
        assert all(np.isfinite(stats['kurt_views']))

    def test_correlation_features(self):
        """
        Test corr() for Pearson correlation coefficient.

        Functions: corr()
        """
        # Global correlation (no groupby)
        stats = self.ds.agg(
            pv_click_corr=col('page_views').corr(col('clicks')),
            click_cart_corr=col('clicks').corr(col('cart_additions')),
            cart_purchase_corr=col('cart_additions').corr(col('purchases')),
        ).to_df()

        assert len(stats) == 1
        # Correlation must be in [-1, 1]
        for c in ['pv_click_corr', 'click_cart_corr', 'cart_purchase_corr']:
            val = stats[c].iloc[0]
            if not pd.isna(val):
                assert -1 <= val <= 1, f"{c} out of range: {val}"

    def test_covariance_features(self):
        """
        Test covarPop() for population covariance.

        Functions: covar_pop()
        """
        stats = self.ds.agg(
            pv_click_covar=col('page_views').covar_pop(col('clicks')),
            amount_purchases_covar=col('purchase_amount').covar_pop(col('purchases')),
        ).to_df()

        assert 'pv_click_covar' in stats.columns
        assert 'amount_purchases_covar' in stats.columns

    # ========== Quantile Features ==========

    def test_quantile_features(self):
        """
        Test quantile() for percentile calculation.

        Functions: quantile(0.25), quantile(0.5), quantile(0.75)
        """
        stats = self.ds.agg(
            q25_amount=col('purchase_amount').quantile(0.25),
            q50_amount=col('purchase_amount').quantile(0.5),
            q75_amount=col('purchase_amount').quantile(0.75),
            q90_amount=col('purchase_amount').quantile(0.9),
        ).to_df()

        assert len(stats) == 1
        # Quantiles should be monotonically increasing
        assert stats['q25_amount'].iloc[0] <= stats['q50_amount'].iloc[0]
        assert stats['q50_amount'].iloc[0] <= stats['q75_amount'].iloc[0]
        assert stats['q75_amount'].iloc[0] <= stats['q90_amount'].iloc[0]

    def test_quantile_by_group(self):
        """
        Test quantile() with groupby.
        """
        stats = (
            self.ds.groupby('device_type')
            .agg(
                median_amount=col('purchase_amount').quantile(0.5),
                p90_views=col('page_views').quantile(0.9),
            )
            .to_df()
        )

        assert len(stats) == 3  # mobile, desktop, tablet
        assert 'median_amount' in stats.columns
        assert 'p90_views' in stats.columns

    # ========== TopK Features ==========

    def test_topk_features(self):
        """
        Test topK() for finding most frequent values.

        Functions: top_k()
        """
        stats = self.ds.agg(
            top3_devices=col('device_type').top_k(3),
            top5_countries=col('country').top_k(5),
            top3_referrers=col('referrer').top_k(3),
        ).to_df()

        assert len(stats) == 1
        assert 'top3_devices' in stats.columns
        assert 'top5_countries' in stats.columns
        # topK returns array
        top_devices = stats['top3_devices'].iloc[0]
        assert len(top_devices) <= 3

    def test_topk_weighted_features(self):
        """
        Test topKWeighted() for weighted top-K.

        Functions: top_k_weighted(weight, k)
        Syntax: col.top_k_weighted(weight_column, k=10)

        NOTE: topKWeighted requires integer weight. Float weights must be cast.
        """
        ds = DataStore.from_df(self.df.copy())

        # Find referrers with highest purchase count (integer weight)
        # topKWeighted requires integer weight
        stats = ds.agg(
            top_referrers_by_purchases=col('referrer').top_k_weighted(col('purchases'), k=3),
        ).to_df()

        assert 'top_referrers_by_purchases' in stats.columns

    # ========== ArgMin/ArgMax Features ==========

    def test_argmin_argmax_features(self):
        """
        Test argMin() and argMax() for finding context of extreme values.

        Functions: arg_min(), arg_max()
        """
        stats = (
            self.ds.groupby('device_type')
            .agg(
                top_spender_id=col('user_id').arg_max(col('purchase_amount')),
                lowest_spender_id=col('user_id').arg_min(col('purchase_amount')),
                max_amount=col('purchase_amount').max(),
                min_amount=col('purchase_amount').min(),
            )
            .to_df()
        )

        assert len(stats) == 3  # 3 device types
        assert 'top_spender_id' in stats.columns
        assert 'lowest_spender_id' in stats.columns
        # argMax/argMin should return valid user_ids
        assert all(stats['top_spender_id'] > 0)
        assert all(stats['lowest_spender_id'] > 0)

    # ========== Window Functions ==========

    def test_window_functions_row_number(self):
        """
        Test row_number() window function via SQL.

        Functions: row_number() OVER (PARTITION BY ... ORDER BY ...)
        """
        ds = DataStore.from_df(self.df.copy())

        # Window functions require SQL
        result = ds.sql(
            """
            SELECT 
                *,
                row_number() OVER (PARTITION BY device_type ORDER BY purchase_amount DESC) as spend_rank
            FROM __df__
        """
        ).to_df()

        assert 'spend_rank' in result.columns
        # Check each device_type starts with rank 1
        for device in ['mobile', 'desktop', 'tablet']:
            device_data = result[result['device_type'] == device]
            assert 1 in device_data['spend_rank'].values

    def test_window_functions_dense_rank_ntile(self):
        """
        Test dense_rank() and ntile() window functions.

        Functions: dense_rank(), ntile()
        """
        ds = DataStore.from_df(self.df.copy())

        result = ds.sql(
            """
            SELECT 
                user_id,
                device_type,
                purchase_amount,
                dense_rank() OVER (ORDER BY purchase_amount DESC) as global_rank,
                ntile(4) OVER (PARTITION BY device_type ORDER BY purchase_amount) as quartile
            FROM __df__
        """
        ).to_df()

        assert 'global_rank' in result.columns
        assert 'quartile' in result.columns
        # ntile(4) should produce values 1-4
        assert set(result['quartile'].unique()).issubset({1, 2, 3, 4})

    def test_window_functions_percent_rank(self):
        """
        Test percent_rank() window function.

        Functions: percent_rank()
        """
        ds = DataStore.from_df(self.df.copy())

        result = ds.sql(
            """
            SELECT 
                user_id,
                purchase_amount,
                percent_rank() OVER (ORDER BY purchase_amount) as pct_rank
            FROM __df__
        """
        ).to_df()

        assert 'pct_rank' in result.columns
        # percent_rank is between 0 and 1
        assert all(result['pct_rank'] >= 0)
        assert all(result['pct_rank'] <= 1)

    def test_window_functions_lag_lead(self):
        """
        Test lag() and lead() window functions.

        Functions: lagInFrame(), leadInFrame()
        """
        ds = DataStore.from_df(self.df.copy())

        result = ds.sql(
            """
            SELECT 
                user_id,
                signup_date,
                page_views,
                lagInFrame(page_views, 1) OVER (ORDER BY signup_date) as prev_views,
                leadInFrame(page_views, 1) OVER (ORDER BY signup_date) as next_views
            FROM __df__
            ORDER BY signup_date
            LIMIT 100
        """
        ).to_df()

        assert 'prev_views' in result.columns
        assert 'next_views' in result.columns

    # ========== String Features ==========

    def test_string_length_features(self):
        """
        Test string length using .str.len()
        """
        ds = DataStore.from_df(self.df.copy())

        # Mixed reference styles
        ds['query_len1'] = ds['last_search_query'].str.len()
        ds['query_len2'] = ds.last_search_query.str.len()

        result = ds.to_df()

        assert 'query_len1' in result.columns
        assert 'query_len2' in result.columns
        assert all(result['query_len1'] > 0)

    def test_string_case_features(self):
        """
        Test string case conversion using .str accessor
        """
        ds = DataStore.from_df(self.df.copy())

        ds['upper_query'] = ds['last_search_query'].str.upper()
        ds['lower_query'] = ds.last_search_query.str.lower()

        result = ds.to_df()

        assert 'upper_query' in result.columns
        assert 'lower_query' in result.columns

    def test_string_multisearch_any(self):
        """
        Test multiSearchAny() for keyword matching.

        Functions: multi_search_any()
        """
        ds = DataStore.from_df(self.df.copy())

        # multiSearchAny returns 1 if any keyword is found
        result = ds.sql(
            """
            SELECT 
                last_search_query,
                multiSearchAny(last_search_query, ['laptop', 'computer', 'wireless']) as has_tech_keyword,
                multiSearchAny(last_search_query, ['shoes', 'jacket', 'yoga']) as has_fitness_keyword
            FROM __df__
        """
        ).to_df()

        assert 'has_tech_keyword' in result.columns
        assert 'has_fitness_keyword' in result.columns
        # Results should be 0 or 1
        assert set(result['has_tech_keyword'].unique()).issubset({0, 1})

    def test_string_split_by_whitespace(self):
        """
        Test splitByWhitespace() for tokenization.

        Functions: splitByWhitespace()

        Note: ClickHouse doesn't support Nullable(Array), so we use ifNull()
        to convert NULL strings to empty string, which produces empty array [].
        """
        ds = DataStore.from_df(self.df.copy())

        # Use ifNull to handle Nullable columns - ClickHouse doesn't support Nullable(Array)
        result = ds.sql(
            """
            SELECT 
                last_search_query,
                splitByWhitespace(ifNull(last_search_query, '')) as tokens,
                length(splitByWhitespace(ifNull(last_search_query, ''))) as word_count
            FROM __df__
        """
        ).to_df()

        assert 'tokens' in result.columns
        assert 'word_count' in result.columns
        # Each query has at least 2 words
        assert all(result['word_count'] >= 2)

    # ========== DateTime Features ==========

    def test_datetime_extraction_features(self):
        """
        Test datetime component extraction using .dt accessor
        """
        ds = DataStore.from_df(self.df.copy())

        # Mixed reference styles
        ds['signup_year'] = ds['signup_date'].dt.year
        ds['signup_month'] = ds.signup_date.dt.month
        ds['signup_day'] = ds['signup_date'].dt.day
        ds['signup_dow'] = ds.signup_date.dt.dayofweek

        result = ds.to_df()

        assert 'signup_year' in result.columns
        assert 'signup_month' in result.columns
        assert 'signup_day' in result.columns
        assert 'signup_dow' in result.columns

    def test_datetime_datediff_features(self):
        """
        Test date difference calculation using age() function.

        Functions: age()

        NOTE: Using age() instead of dateDiff() because dateDiff's string
        argument ('day') gets transformed incorrectly by SQL processing.
        """
        # Create data with two date columns for reliable diff calculation
        df = pd.DataFrame(
            {
                'user_id': range(1, 11),
                'start_date': pd.date_range('2024-01-01', periods=10, freq='D'),
                'end_date': pd.date_range('2024-01-15', periods=10, freq='D'),
            }
        )
        ds = DataStore.from_df(df)

        # Calculate date differences using age() with toDate() conversion
        result = ds.sql(
            """
            SELECT 
                user_id,
                start_date,
                end_date,
                age('day', toDate(start_date), toDate(end_date)) as day_diff,
                age('week', toDate(start_date), toDate(end_date)) as week_diff
            FROM __df__
        """
        ).to_df()

        assert 'day_diff' in result.columns
        assert 'week_diff' in result.columns
        # end_date - start_date = 14 days
        assert all(result['day_diff'] == 14)
        assert all(result['week_diff'] == 2)

    # ========== Funnel Analysis ==========

    def test_window_funnel_analysis(self):
        """
        Test windowFunnel() for conversion funnel analysis.

        Functions: windowFunnel()
        """
        # Create event data with timestamps as integers to avoid overflow
        np.random.seed(42)
        n_events = 1000
        # Use integer timestamps (seconds since a reference point)
        base_timestamp = 1704067200  # 2024-01-01 00:00:00 UTC
        events_df = pd.DataFrame(
            {
                'user_id': np.random.choice(range(1, 101), n_events),
                'event_ts': base_timestamp + np.arange(n_events) * 300,  # 5 minute intervals
                'event_type': np.random.choice(
                    ['view', 'click', 'add_cart', 'purchase'], n_events, p=[0.5, 0.3, 0.15, 0.05]
                ),
            }
        )
        ds = DataStore.from_df(events_df)

        # Calculate funnel: view -> click -> add_cart -> purchase
        # Using integer timestamp directly to avoid conversion issues
        result = ds.sql(
            """
            SELECT 
                user_id,
                windowFunnel(3600)(
                    toUInt32(event_ts),
                    event_type = 'view',
                    event_type = 'click',
                    event_type = 'add_cart',
                    event_type = 'purchase'
                ) as funnel_level
            FROM __df__
            GROUP BY user_id
        """
        ).to_df()

        assert 'funnel_level' in result.columns
        # funnel_level should be between 0 and 4
        assert all(result['funnel_level'] >= 0)
        assert all(result['funnel_level'] <= 4)

    # ========== Retention Analysis ==========

    def test_retention_features(self):
        """
        Test retention calculation using sum and count.
        """
        stats = (
            self.ds.groupby('device_type')
            .agg(
                total_users=col('user_id').count(),
                active_users=col('page_views').count(),
                churned_count=col('churned').sum(),
            )
            .to_df()
        )

        assert len(stats) == 3
        assert 'total_users' in stats.columns
        assert 'churned_count' in stats.columns

        # Calculate retention rate (active users / total users)
        stats['retention_rate'] = 1 - (stats['churned_count'] / stats['total_users'])
        assert all(stats['retention_rate'] >= 0)
        assert all(stats['retention_rate'] <= 1)

    # ========== Mixed Reference Aggregations ==========

    def test_mixed_reference_groupby_agg(self):
        """
        Test groupby().agg() with col() expressions.

        Since ds['col'].mean() returns scalar (matching pandas), SQL agg() must use col().
        This test verifies col() works correctly for SQL aggregation building.
        """
        # Use col() for SQL aggregation building
        # ds['col'].mean() returns scalar, which cannot be used in agg()
        stats1 = (
            self.ds.groupby('device_type')
            .agg(
                avg_views=col('page_views').mean(),
                std_views=col('page_views').stddev_samp(),
                total_purchases=col('purchases').sum(),
            )
            .to_df()
            .sort_values('device_type')
            .reset_index(drop=True)
        )

        # Verify col() produces expected aggregations
        # Cross-check with pandas
        expected = (
            self.df.groupby('device_type')
            .agg(
                avg_views=('page_views', 'mean'),
                std_views=('page_views', 'std'),
                total_purchases=('purchases', 'sum'),
            )
            .reset_index()
            .sort_values('device_type')
            .reset_index(drop=True)
        )

        # Check values are close
        np.testing.assert_array_almost_equal(
            stats1['avg_views'].values,
            expected['avg_views'].values,
            decimal=3
        )


    def test_multi_level_groupby_with_advanced_stats(self):
        """
        Test multi-level groupby with advanced statistical functions.
        """
        stats = (
            self.ds.groupby('device_type', 'country')
            .agg(
                avg_amount=col('purchase_amount').mean(),
                std_amount=col('purchase_amount').stddev_samp(),
                skew_amount=col('purchase_amount').skew_pop(),
                median_amount=col('purchase_amount').quantile(0.5),
                user_count=col('user_id').count(),
            )
            .to_df()
        )

        # 3 devices * 5 countries = max 15 groups
        assert len(stats) <= 15
        assert all(stats['std_amount'] >= 0)

    # ========== Distribution Features ==========

    def test_distribution_features(self):
        """
        Test distribution statistics: skewPop, kurtPop, stddevSamp, varSamp.
        """
        stats = self.ds.agg(
            skew_pv=col('page_views').skew_pop(),
            kurt_pv=col('page_views').kurt_pop(),
            std_pv=col('page_views').stddev_samp(),
            var_pv=col('page_views').var_samp(),
            skew_amount=col('purchase_amount').skew_pop(),
            kurt_amount=col('purchase_amount').kurt_pop(),
        ).to_df()

        assert len(stats) == 1
        # Exponential distribution has skewness ~2
        assert stats['skew_amount'].iloc[0] > 0  # Right-skewed

    # ========== Complete ML Feature Pipeline ==========

    def test_complete_ml_feature_pipeline(self):
        """
        Complete ML feature engineering pipeline using advanced functions.
        """
        ds = DataStore.from_df(self.df.copy())

        # Phase 1: Basic derived features (mixed references)
        ds['recency_score'] = 1 / (ds["days_since_last_visit"] + 1)
        ds['frequency_score'] = ds.page_views + ds.clicks
        ds['monetary_score'] = col('purchase_amount')

        # Phase 2: Engagement features
        ds['engagement_score'] = col('page_views') * 0.3 + col('clicks') * 0.5 + col('cart_additions') * 0.2
        ds['conversion_rate'] = ds["purchases"] / (ds.page_views + 1)

        # Execute pipeline
        result = ds.to_df()

        expected_features = [
            'recency_score',
            'frequency_score',
            'monetary_score',
            'engagement_score',
            'conversion_rate',
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

        # Verify value reasonability
        assert all(result['recency_score'] > 0)
        assert all(result['conversion_rate'] >= 0)

    def test_grouped_advanced_statistics_pipeline(self):
        """
        Test a pipeline that computes advanced statistics by group.
        """
        # Compute comprehensive statistics by device type
        stats = (
            self.ds.groupby('device_type')
            .agg(
                # Central tendency
                mean_amount=col('purchase_amount').mean(),
                median_amount=col('purchase_amount').quantile(0.5),
                # Dispersion
                std_amount=col('purchase_amount').stddev_samp(),
                var_amount=col('purchase_amount').var_samp(),
                # Shape
                skew_amount=col('purchase_amount').skew_pop(),
                kurt_amount=col('purchase_amount').kurt_pop(),
                # Quantiles
                q25_amount=col('purchase_amount').quantile(0.25),
                q75_amount=col('purchase_amount').quantile(0.75),
                # Counts
                user_count=col('user_id').count(),
            )
            .to_df()
        )

        assert len(stats) == 3
        # IQR = Q3 - Q1 should be positive
        iqr = stats['q75_amount'] - stats['q25_amount']
        assert all(iqr >= 0)

    # ========== Filter + Aggregate ==========

    def test_filter_then_aggregate_with_advanced_stats(self):
        """
        Test filter followed by aggregation with advanced statistics.
        """
        ds = DataStore.from_df(self.df.copy())

        # Add derived feature
        ds['value_score'] = ds["purchase_amount"] * ds.purchases

        # Filter high value users
        high_value = ds.filter(ds.value_score > 100)

        # Aggregate with advanced stats
        stats = (
            high_value.groupby('device_type')
            .agg(
                user_count=col('user_id').count(),
                avg_value=col('value_score').mean(),
                std_value=col('value_score').stddev_samp(),
                median_value=col('value_score').quantile(0.5),
            )
            .to_df()
        )

        assert 'user_count' in stats.columns
        assert 'std_value' in stats.columns

    def test_filter_with_compound_conditions(self):
        """
        Test filter with compound conditions using mixed references.
        """
        ds = DataStore.from_df(self.df.copy())

        filtered = ds.filter((ds["page_views"] > 50) & (ds.purchases > 0) & (col('churned') == 0))

        result = filtered.to_df()

        assert all(result['page_views'] > 50)
        assert all(result['purchases'] > 0)
        assert all(result['churned'] == 0)

    # ========== Z-Score Standardization ==========

    def test_zscore_standardization(self):
        """
        Test Z-score standardization using mean and stddev.
        """
        ds = DataStore.from_df(self.df.copy())

        # Calculate Z-scores using SQL for window functions
        result = ds.sql(
            """
            SELECT 
                user_id,
                purchase_amount,
                (purchase_amount - avg(purchase_amount) OVER ()) / 
                    nullIf(stddevPop(purchase_amount) OVER (), 0) as zscore_amount,
                (page_views - avg(page_views) OVER ()) / 
                    nullIf(stddevPop(page_views) OVER (), 0) as zscore_views
            FROM __df__
        """
        ).to_df()

        assert 'zscore_amount' in result.columns
        assert 'zscore_views' in result.columns
        # Z-scores should have mean ~0
        assert abs(result['zscore_amount'].mean()) < 0.1
        assert abs(result['zscore_views'].mean()) < 0.1

    # ========== Edge Cases ==========

    def test_empty_dataframe_handling(self):
        """Test empty DataFrame handling."""
        empty_df = pd.DataFrame(
            {
                'x': pd.Series([], dtype=float),
                'y': pd.Series([], dtype=float),
                'category': pd.Series([], dtype=str),
            }
        )
        ds = DataStore.from_df(empty_df)
        ds['sum'] = ds['x'] + ds['y']

        result = ds.to_df()
        assert len(result) == 0
        assert 'sum' in result.columns

    def test_null_handling_with_aggregates(self):
        """Test NULL value handling with aggregate functions."""
        df_with_nulls = pd.DataFrame(
            {
                'value': [1.0, 2.0, np.nan, 4.0, np.nan],
                'category': ['A', 'A', 'B', 'B', 'B'],
            }
        )
        ds = DataStore.from_df(df_with_nulls)

        stats = (
            ds.groupby('category')
            .agg(
                mean_val=col('value').mean(),
                std_val=col('value').stddev_samp(),
                count_val=col('value').count(),
            )
            .to_df()
        )

        # ClickHouse should ignore NULLs
        assert 'mean_val' in stats.columns
        assert 'std_val' in stats.columns

    def test_wide_table_with_100_features(self):
        """Test wide table with 100+ features (common in ML)."""
        n_features = 100
        n_rows = 1000
        data = {'id': range(n_rows), 'category': np.random.choice(['A', 'B', 'C'], n_rows)}
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_rows)

        wide_df = pd.DataFrame(data)
        ds = DataStore.from_df(wide_df)

        # Compute statistics for multiple features
        stats = (
            ds.groupby('category')
            .agg(
                mean_f0=col('feature_0').mean(),
                std_f0=col('feature_0').stddev_samp(),
                mean_f1=col('feature_1').mean(),
                std_f1=col('feature_1').stddev_samp(),
                q50_f2=col('feature_2').quantile(0.5),
                skew_f3=col('feature_3').skew_pop(),
                kurt_f4=col('feature_4').kurt_pop(),
            )
            .to_df()
        )

        assert len(stats) == 3  # A, B, C
        assert all(stats['std_f0'] >= 0)
        assert 'skew_f3' in stats.columns
        assert 'kurt_f4' in stats.columns

    # ========== Lazy Execution Verification ==========

    def test_lazy_execution_no_premature_execution(self):
        """
        Verify lazy execution: operations should not execute prematurely.

        All operations are recorded as LazyOps until to_df() is called.
        """
        ds = DataStore.from_df(self.df.copy())

        initial_ops = len(ds._lazy_ops)

        # Add multiple operations
        ds['feat1'] = ds["page_views"] * 2
        ds['feat2'] = ds.clicks + ds.cart_additions
        ds['feat3'] = col('purchases') / (col('page_views') + 1)

        # Verify operations are recorded
        assert len(ds._lazy_ops) > initial_ops, "Operations should be recorded lazily"

        # Create aggregation
        stats = ds.groupby('device_type').agg(total=col('feat1').sum())
        assert len(stats._lazy_ops) > 0, "Aggregation should be lazy"

        # Only to_df() triggers execution
        result = stats.to_df()
        assert len(result) > 0

    def test_chained_lazy_operations(self):
        """Test chaining multiple lazy operations."""
        ds = DataStore.from_df(self.df.copy())

        ds['score'] = ds['page_views'] * 0.5 + ds.clicks * 0.5
        filtered = ds.filter(ds.score > 30)
        filtered['normalized'] = filtered['score'] / 100

        assert len(filtered._lazy_ops) > 0

        result = filtered.to_df()
        assert 'score' in result.columns
        assert 'normalized' in result.columns
        assert all(result['score'] > 30)


class TestReferenceTypeConsistency:
    """Tests for type and behavior consistency of three reference styles"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                'a': [1, 2, 3, 4, 5],
                'b': [10, 20, 30, 40, 50],
                'c': ['x', 'y', 'z', 'x', 'y'],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_type_differences(self):
        """Document type differences between three reference styles."""
        ref_getitem = self.ds["a"]
        ref_getattr = self.ds.a
        ref_col = col("a")

        # Verify types
        assert isinstance(ref_getitem, ColumnExpr)
        assert isinstance(ref_getattr, ColumnExpr)
        assert isinstance(ref_col, Field)

        # Verify SQL output is consistent
        sql_getitem = ref_getitem._expr.to_sql()
        sql_getattr = ref_getattr._expr.to_sql()
        sql_col = ref_col.to_sql()

        assert sql_getitem == sql_getattr == sql_col

    def test_aggregate_method_consistency(self):
        """Test that aggregate methods work consistently.
        
        Note: ds['col'].mean() returns scalar (matching pandas), while
        col('col').mean() returns Expression for SQL building.
        """
        import numpy as np
        import pandas as pd
        
        # col() returns Expression for SQL building
        mean1 = col('a').mean()
        assert 'avg' in mean1.to_sql().lower()
        
        # ds['col'].mean() returns scalar (matching pandas)
        mean2 = self.ds['a'].mean()
        assert isinstance(mean2, (int, float, np.integer, np.floating))
        
        # Verify scalar value matches pandas
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [10.0, 20.0, 30.0, 40.0, 50.0]})
        expected = df['a'].mean()
        assert mean2 == expected

    def test_statistical_method_consistency(self):
        """Test that statistical methods work via col() for SQL building."""
        # col() returns Expression for SQL building
        std1 = col('a').stddev_samp()
        var1 = col('a').var_samp()

        assert 'stddevSamp' in std1.to_sql()
        assert 'varSamp' in var1.to_sql()
        
        # ds['col'].std() returns scalar (matching pandas)
        import numpy as np
        std2 = self.ds['a'].std()
        var2 = self.ds['a'].var()
        
        assert isinstance(std2, (int, float, np.integer, np.floating))
        assert isinstance(var2, (int, float, np.integer, np.floating))

    def test_reserved_method_name_collision(self):
        """
        KNOWN ISSUE: When column name equals a method name,
        ds.column_name returns the method, not the column.

        WORKAROUND: Use ds["column_name"] or col("column_name")
        """
        df = pd.DataFrame(
            {
                'value': [1, 2, 3],
                'count': [10, 20, 30],  # 'count' conflicts with ds.count method
            }
        )
        ds = DataStore.from_df(df)

        # ds["count"] returns ColumnExpr (correct)
        ref_bracket = ds["count"]
        assert isinstance(ref_bracket, ColumnExpr)

        # ds.count returns method (NOT the column!)
        ref_attr = ds.count
        assert callable(ref_attr)

        # col("count") returns Field (safe)
        ref_col = col("count")
        assert isinstance(ref_col, Field)


class TestKnownIssuesDocumentation:
    """Document and test known issues"""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.df = pd.DataFrame(
            {
                'text': ['hello', 'world', 'test'],
                'value': [1, 2, 3],
            }
        )
        self.ds = DataStore.from_df(self.df)

    def test_row_order_is_preserved_basic(self):
        """
        Row order IS NOW preserved with the pre-added index column solution.

        Previously, chDB's Python() table function did not guarantee row order.
        Now DataStore automatically adds a __row_idx__ column and ORDER BY to preserve order.
        """
        ds = DataStore.from_df(self.df.copy())
        ds['len'] = ds['text'].str.len()
        result = ds.to_df()

        assert 'text' in result.columns
        assert 'len' in result.columns
        assert len(result) == 3

        # Verify exact row order is preserved
        assert result['text'].tolist() == ['hello', 'world', 'test']
        assert result['value'].tolist() == [1, 2, 3]
        # length of 'hello'=5, 'world'=5, 'test'=4
        assert result['len'].tolist() == [5, 5, 4]

    def test_row_order_unsorted_input(self):
        """Test row order preservation with unsorted input data."""
        df = pd.DataFrame(
            {
                'id': [100, 50, 75, 25, 10],
                'name': ['j', 'e', 'h', 'c', 'b'],
            }
        )
        ds = DataStore.from_df(df)
        result = ds.to_df()

        # Exact order must be preserved
        assert result['id'].tolist() == [100, 50, 75, 25, 10]
        assert result['name'].tolist() == ['j', 'e', 'h', 'c', 'b']

    def test_row_order_with_computed_columns(self):
        """Test row order preservation with computed columns."""
        df = pd.DataFrame(
            {
                'id': [100, 50, 75, 25, 10],
                'value': [1, 2, 3, 4, 5],
            }
        )
        ds = DataStore.from_df(df)
        ds['doubled'] = ds['value'] * 2
        result = ds.to_df()

        # Order and computation must be correct
        assert result['id'].tolist() == [100, 50, 75, 25, 10]
        assert result['doubled'].tolist() == [2, 4, 6, 8, 10]

    def test_row_order_consistency_multiple_runs(self):
        """Test row order is consistent across multiple runs."""
        df = pd.DataFrame(
            {
                'id': [5, 3, 8, 1, 6, 2, 9, 4, 7, 10],
                'value': [50, 30, 80, 10, 60, 20, 90, 40, 70, 100],
            }
        )
        original_ids = df['id'].tolist()

        for _ in range(5):
            ds = DataStore.from_df(df)
            ds['computed'] = ds['value'] * 2
            result = ds.to_df()
            assert result['id'].tolist() == original_ids

    def test_row_order_with_filter(self):
        """Test row order preservation after filtering."""
        df = pd.DataFrame(
            {
                'id': [5, 3, 8, 1, 6, 2, 9, 4, 7, 10],
                'value': [50, 30, 80, 10, 60, 20, 90, 40, 70, 100],
            }
        )
        ds = DataStore.from_df(df)
        ds['score'] = ds['value'] * 0.1
        filtered = ds.filter(ds.value > 50)
        result = filtered.to_df()

        # Expected: ids where value > 50, in original DataFrame order
        # Original order: 5(50), 3(30), 8(80)✓, 1(10), 6(60)✓, 2(20), 9(90)✓, 4(40), 7(70)✓, 10(100)✓
        expected_ids = [8, 6, 9, 7, 10]
        assert result['id'].tolist() == expected_ids

    def test_row_order_large_dataset(self):
        """Test row order preservation with large dataset."""
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame(
            {
                'id': np.random.permutation(n),  # Random order
                'value': np.random.randn(n),
            }
        )
        original_first_10 = df['id'].head(10).tolist()

        ds = DataStore.from_df(df)
        ds['computed'] = ds['value'] * 2
        result = ds.to_df()

        assert result['id'].head(10).tolist() == original_first_10
        assert len(result) == n

    def test_timezone_handling_issue(self):
        """
        KNOWN ISSUE: Timezone handling inconsistency.

        Pandas datetime64[ns] is timezone-naive, but chDB treats it as UTC.

        Workaround:
        - Use timezone-aware datetimes
        - Explicitly convert to UTC
        """
        import chdb

        df = pd.DataFrame(
            {
                'event_time': pd.date_range('2024-01-01 00:00', periods=3, freq='h'),
            }
        )

        tz_result = chdb.query("SELECT timezone()", 'CSV')
        ch_timezone = str(tz_result).strip().strip('"').strip()

        ds = DataStore.from_df(df)
        ds['hour'] = ds['event_time'].dt.hour
        result = ds.to_df()

        pandas_hours = df['event_time'].dt.hour.tolist()
        chdb_hours = result['hour'].tolist()

        print(f"\nClickHouse timezone: {ch_timezone}")
        print(f"Pandas hours: {pandas_hours}")
        print(f"chDB hours: {chdb_hours}")

        if pandas_hours != chdb_hours:
            print("WARNING: Timezone mismatch detected!")
            offset = (chdb_hours[0] - pandas_hours[0]) % 24
            print(f"Time offset: {offset} hours")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
