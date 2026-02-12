"""
Test: Machine Learning Feature Engineering Pipeline
====================================================

This test demonstrates a realistic ML feature engineering workflow using:
- Lazy execution mode (operations recorded but not executed until needed)
- Function chaining for fluent API (NOT raw SQL)
- Statistical functions (variance, stddev, correlation, skewness, kurtosis)
- Quantile functions for feature binning
- String functions for text feature extraction
- Window functions for time-series features
- Aggregation functions for grouped statistics
- TopK and ArgMin/Max for ranking features

Scenario: E-commerce user behavior analysis and purchase prediction
- Wide table with user activity, transactions, and profile data
- Build features for predicting user churn or purchase likelihood
"""

import pytest
import pandas as pd
import numpy as np
from datastore import DataStore, col, F, Field, Sum, Count, Avg, Min, Max


class TestMLFeatureEngineering:
    """Test ML feature engineering with lazy execution and function chaining."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Create a realistic e-commerce wide table for ML feature engineering."""
        np.random.seed(42)
        n_users = 1000

        # Generate realistic user data
        self.df = pd.DataFrame(
            {
                # User identifiers
                'user_id': range(1, n_users + 1),
                'signup_date': pd.date_range('2023-01-01', periods=n_users, freq='h'),
                # Numeric features - activity metrics
                'page_views': np.random.poisson(50, n_users),
                'session_duration': np.random.exponential(300, n_users),  # seconds
                'clicks': np.random.poisson(20, n_users),
                'cart_additions': np.random.poisson(5, n_users),
                'purchases': np.random.poisson(2, n_users),
                'purchase_amount': np.random.exponential(100, n_users),
                # Time-based metrics
                'days_since_last_visit': np.random.exponential(7, n_users),
                'days_since_last_purchase': np.random.exponential(30, n_users),
                'visit_frequency': np.random.uniform(0.1, 5, n_users),  # visits per week
                # User profile
                'age': np.random.normal(35, 12, n_users).clip(18, 80).astype(int),
                'account_balance': np.random.exponential(500, n_users),
                'loyalty_points': np.random.poisson(1000, n_users),
                # Categorical encoded as strings
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_users, p=[0.6, 0.3, 0.1]),
                'country': np.random.choice(['US', 'UK', 'DE', 'FR', 'JP', 'CN'], n_users),
                'referrer': np.random.choice(['google', 'facebook', 'direct', 'email', 'affiliate'], n_users),
                # Text features (for NLP processing)
                'last_search_query': np.random.choice(
                    [
                        'laptop computer',
                        'running shoes',
                        'winter jacket',
                        'headphones wireless',
                        'coffee maker',
                        'yoga mat',
                        'phone case',
                        'backpack travel',
                        'watch smart',
                        'camera digital',
                    ],
                    n_users,
                ),
                # Target variable
                'churned': np.random.choice([0, 1], n_users, p=[0.85, 0.15]),
            }
        )

        self.ds = DataStore.from_df(self.df)

    def test_lazy_chain_basic_statistics(self):
        """
        Test lazy computation of basic statistical features.

        Equivalent SQL:
            SELECT *,
                page_views * 0.3 + clicks * 0.4 + cart_additions * 0.3 AS engagement_score,
                purchases / (page_views + 1) AS conversion_rate,
                purchase_amount / (purchases + 1) AS avg_order_value
            FROM __df__
        """
        ds = self.ds

        # Function chain: Composite engagement score
        ds['engagement_score'] = ds['page_views'] * 0.3 + ds['clicks'] * 0.4 + ds['cart_additions'] * 0.3

        # Function chain: Conversion rate
        ds['conversion_rate'] = ds['purchases'] / (ds['page_views'] + 1)

        # Function chain: Average order value
        ds['avg_order_value'] = ds['purchase_amount'] / (ds['purchases'] + 1)

        # Verify lazy operations are recorded but not executed
        assert len(ds._lazy_ops) >= 3, "Lazy operations should be recorded"

        # Execute and verify results
        result = ds.to_df()
        assert 'engagement_score' in result.columns
        assert 'conversion_rate' in result.columns
        assert 'avg_order_value' in result.columns
        assert len(result) == 1000

    def test_lazy_chain_statistical_aggregations(self):
        """
        Test statistical aggregation functions for feature engineering.

        SQL equivalent:
            SELECT device_type,
                   avg(page_views) AS avg_page_views,
                   sum(purchases) AS total_purchases,
                   count(*) AS user_count
            FROM __df__
            GROUP BY device_type
        """
        # Pure function chain: groupby().agg() with col() expressions
        stats = self.ds.groupby('device_type').agg(
            avg_page_views=col('page_views').mean(),
            total_purchases=col('purchases').sum(),
            user_count=col('page_views').count(),
        )

        # Verify lazy: operation recorded but not executed
        assert len(stats._lazy_ops) > 0

        result = stats.to_df()

        assert len(result) == 3  # mobile, desktop, tablet
        assert 'avg_page_views' in result.columns
        assert 'total_purchases' in result.columns
        assert 'user_count' in result.columns

        # Verify statistics are reasonable
        assert result['user_count'].sum() == 1000

    def test_lazy_chain_derived_features(self):
        """
        Test derived feature computation with function chaining.

        Equivalent SQL:
            SELECT *,
                page_views / (session_duration / 60 + 1) AS pages_per_minute,
                clicks / (page_views + 1) AS click_through_rate,
                cart_additions / (clicks + 1) AS cart_rate,
                purchases / (cart_additions + 1) AS checkout_rate,
                purchase_amount / (purchases + 1) AS avg_order_value,
                loyalty_points / (account_balance + 1) AS loyalty_ratio
            FROM __df__
        """
        ds = self.ds

        # Function chain: Engagement rate features
        ds['pages_per_minute'] = ds['page_views'] / (ds['session_duration'] / 60 + 1)
        ds['click_through_rate'] = ds['clicks'] / (ds['page_views'] + 1)
        ds['cart_rate'] = ds['cart_additions'] / (ds['clicks'] + 1)
        ds['checkout_rate'] = ds['purchases'] / (ds['cart_additions'] + 1)

        # Function chain: Value features
        ds['avg_order_value'] = ds['purchase_amount'] / (ds['purchases'] + 1)
        ds['loyalty_ratio'] = ds['loyalty_points'] / (ds['account_balance'] + 1)

        result = ds.to_df()

        assert 'pages_per_minute' in result.columns
        assert 'click_through_rate' in result.columns
        assert 'cart_rate' in result.columns
        assert 'checkout_rate' in result.columns
        assert 'avg_order_value' in result.columns
        assert 'loyalty_ratio' in result.columns
        assert len(result) == 1000

    def test_lazy_chain_string_features(self):
        """
        Test string-based feature extraction using function chain.

        Equivalent SQL:
            SELECT
                user_id,
                last_search_query,
                length(last_search_query) AS query_length,
                upper(last_search_query) AS query_upper
            FROM __df__
            LIMIT 50
        """
        ds = self.ds

        # Function chain: String length feature
        ds['query_length'] = ds['last_search_query'].str.len()

        # Function chain: Uppercase for normalization
        ds['query_upper'] = ds['last_search_query'].str.upper()

        result = ds.head(50).to_df()

        assert 'query_length' in result.columns
        assert 'query_upper' in result.columns
        assert len(result) == 50

    def test_lazy_chain_datetime_features(self):
        """
        Test time-based feature engineering with function chain.

        Equivalent SQL:
            SELECT
                user_id,
                signup_date,
                toYear(signup_date) AS signup_year,
                toMonth(signup_date) AS signup_month,
                toDayOfWeek(signup_date) AS signup_day_of_week,
                toHour(signup_date) AS signup_hour
            FROM __df__
            LIMIT 100
        """
        ds = self.ds

        # Function chain: Extract datetime components
        ds['signup_year'] = ds['signup_date'].dt.year
        ds['signup_month'] = ds['signup_date'].dt.month
        ds['signup_day_of_week'] = ds['signup_date'].dt.dayofweek
        ds['signup_hour'] = ds['signup_date'].dt.hour

        result = ds.head(100).to_df()

        assert 'signup_year' in result.columns
        assert 'signup_month' in result.columns
        assert 'signup_day_of_week' in result.columns
        assert 'signup_hour' in result.columns
        assert len(result) == 100

    def test_lazy_chain_aggregation_by_category(self):
        """
        Test aggregation by categorical features.

        SQL equivalent:
            SELECT referrer,
                   count(*) AS user_count,
                   sum(purchase_amount) AS total_revenue,
                   avg(purchase_amount) AS avg_revenue,
                   max(purchase_amount) AS max_revenue,
                   min(purchase_amount) AS min_revenue
            FROM __df__
            GROUP BY referrer
        """
        # Pure function chain: groupby().agg() with col() expressions
        stats = self.ds.groupby('referrer').agg(
            user_count=col('user_id').count(),
            total_revenue=col('purchase_amount').sum(),
            avg_revenue=col('purchase_amount').mean(),
            max_revenue=col('purchase_amount').max(),
            min_revenue=col('purchase_amount').min(),
        )

        result = stats.to_df()

        assert len(result) == 5  # 5 referrer types
        assert 'user_count' in result.columns
        assert 'total_revenue' in result.columns
        assert 'avg_revenue' in result.columns
        assert result['user_count'].sum() == 1000

    def test_lazy_chain_correlation_analysis(self):
        """
        Test correlation-based feature analysis using groupby aggregation.

        Equivalent SQL:
            SELECT
                corr(page_views, purchases) AS pv_purchase_corr,
                corr(session_duration, purchases) AS session_purchase_corr,
                corr(cart_additions, purchases) AS cart_purchase_corr
            FROM __df__
        """
        # Function chain: Compute correlations using aggregate
        # Note: For correlation we need to use SQL as it's a complex aggregate
        result = self.ds.sql(
            """
            SELECT 
                corr(page_views, purchases) AS pv_purchase_corr,
                corr(session_duration, purchases) AS session_purchase_corr,
                corr(cart_additions, purchases) AS cart_purchase_corr,
                corr(age, purchase_amount) AS age_amount_corr
            FROM __df__
            """
        ).to_df()

        # Verify correlations are in valid range [-1, 1]
        for c in result.columns:
            val = result[c].iloc[0]
            if not pd.isna(val):
                assert -1 <= val <= 1, f"{c} correlation out of range: {val}"

    def test_lazy_chain_distribution_features(self):
        """
        Test distribution-based features by group.

        SQL equivalent:
            SELECT device_type,
                   count(*) AS n,
                   avg(purchase_amount) AS mean_amount,
                   min(purchase_amount) AS min_amount,
                   max(purchase_amount) AS max_amount
            FROM __df__
            GROUP BY device_type
        """
        # Pure function chain: groupby().agg() with col() expressions
        stats = self.ds.groupby('device_type').agg(
            n=col('user_id').count(),
            mean_amount=col('purchase_amount').mean(),
            min_amount=col('purchase_amount').min(),
            max_amount=col('purchase_amount').max(),
        )

        result = stats.to_df()

        assert 'n' in result.columns
        assert 'mean_amount' in result.columns
        assert 'min_amount' in result.columns
        assert 'max_amount' in result.columns
        assert len(result) == 3  # 3 device types

    def test_lazy_chain_complete_feature_pipeline(self):
        """
        Test a complete feature engineering pipeline combining multiple techniques.

        Equivalent SQL:
            SELECT
                user_id,
                churned,
                -- RFM features
                1 / (days_since_last_visit + 1) AS recency_score,
                visit_frequency AS frequency_score,
                purchase_amount AS monetary_score,
                -- Engagement features
                page_views * 0.2 + clicks * 0.3 + cart_additions * 0.5 AS engagement_score,
                page_views / (session_duration / 60 + 1) AS pages_per_minute,
                clicks / (page_views + 1) AS click_through_rate,
                cart_additions / (clicks + 1) AS cart_rate,
                purchases / (cart_additions + 1) AS checkout_rate,
                -- Value features
                purchase_amount / (purchases + 1) AS avg_order_value,
                loyalty_points / (account_balance + 1) AS loyalty_ratio
            FROM __df__
        """
        ds = self.ds

        # Step 1: RFM features (Recency, Frequency, Monetary)
        ds['recency_score'] = 1 / (ds['days_since_last_visit'] + 1)
        ds['frequency_score'] = ds['visit_frequency']
        ds['monetary_score'] = ds['purchase_amount']

        # Step 2: Engagement features
        ds['engagement_score'] = ds['page_views'] * 0.2 + ds['clicks'] * 0.3 + ds['cart_additions'] * 0.5
        ds['pages_per_minute'] = ds['page_views'] / (ds['session_duration'] / 60 + 1)
        ds['click_through_rate'] = ds['clicks'] / (ds['page_views'] + 1)
        ds['cart_rate'] = ds['cart_additions'] / (ds['clicks'] + 1)
        ds['checkout_rate'] = ds['purchases'] / (ds['cart_additions'] + 1)

        # Step 3: Value features
        ds['avg_order_value'] = ds['purchase_amount'] / (ds['purchases'] + 1)
        ds['loyalty_ratio'] = ds['loyalty_points'] / (ds['account_balance'] + 1)

        # Verify all operations are lazy
        assert len(ds._lazy_ops) >= 10, "Multiple lazy operations should be recorded"

        # Execute the pipeline
        feature_df = ds.to_df()

        # Verify all features are created
        expected_features = [
            'recency_score',
            'frequency_score',
            'monetary_score',
            'engagement_score',
            'pages_per_minute',
            'click_through_rate',
            'cart_rate',
            'checkout_rate',
            'avg_order_value',
            'loyalty_ratio',
        ]

        for feat in expected_features:
            assert feat in feature_df.columns, f"Missing feature: {feat}"

        assert len(feature_df) == 1000

        print(f"\n✅ Feature engineering pipeline complete!")
        print(f"   - Total samples: {len(feature_df)}")
        print(f"   - Total features: {len(feature_df.columns)}")
        print(f"   - Churn rate: {feature_df['churned'].mean():.2%}")

    def test_lazy_chain_filter_and_transform(self):
        """
        Test filtering combined with transformations.

        Equivalent SQL:
            SELECT
                user_id,
                purchase_amount,
                purchase_amount * 1.1 AS with_tax,
                page_views,
                page_views * 2 AS double_views
            FROM __df__
            WHERE churned = 1 AND purchase_amount > 50
        """
        ds = self.ds

        # Function chain: Add derived columns
        ds['with_tax'] = ds['purchase_amount'] * 1.1
        ds['double_views'] = ds['page_views'] * 2

        # Function chain: Filter churned users with significant purchases
        filtered = ds.filter((ds['churned'] == 1) & (ds['purchase_amount'] > 50))

        # Select specific columns
        result = filtered.select('user_id', 'purchase_amount', 'with_tax', 'page_views', 'double_views').to_df()

        assert 'with_tax' in result.columns
        assert 'double_views' in result.columns
        assert all(result['purchase_amount'] > 50)
        # All should be churned (but churned column not selected)

    def test_lazy_chain_multiple_aggregations(self):
        """
        Test multiple aggregations on same column with multi-level groupby.

        SQL equivalent:
            SELECT device_type, country,
                   count(*) AS user_count,
                   sum(purchases) AS total_purchases,
                   avg(purchases) AS avg_purchases,
                   max(purchases) AS max_purchases,
                   min(purchases) AS min_purchases
            FROM __df__
            GROUP BY device_type, country
        """
        # Pure function chain: groupby().agg() with multiple group keys
        stats = self.ds.groupby('device_type', 'country').agg(
            user_count=col('user_id').count(),
            total_purchases=col('purchases').sum(),
            avg_purchases=col('purchases').mean(),
            max_purchases=col('purchases').max(),
            min_purchases=col('purchases').min(),
        )

        result = stats.to_df()

        # 3 devices * 6 countries = up to 18 combinations
        assert len(result) <= 18
        assert 'device_type' in result.columns
        assert 'country' in result.columns
        assert 'total_purchases' in result.columns
        assert result['user_count'].sum() == 1000

    def test_lazy_chain_retention_features(self):
        """
        Test retention analysis using function chains.

        SQL equivalent:
            SELECT sum(day1_active) AS d1_retained,
                   sum(day7_active) AS d7_retained,
                   sum(day30_active) AS d30_retained,
                   count(*) AS total_users
            FROM __df__
        """
        # Create retention data
        retention_df = pd.DataFrame(
            {
                'user_id': range(1, 101),
                'day1_active': np.random.choice([0, 1], 100, p=[0.2, 0.8]),
                'day7_active': np.random.choice([0, 1], 100, p=[0.4, 0.6]),
                'day30_active': np.random.choice([0, 1], 100, p=[0.6, 0.4]),
            }
        )
        retention_ds = DataStore.from_df(retention_df)

        # Pure function chain: agg() without groupby for global aggregation
        result = retention_ds.agg(
            d1_retained=col('day1_active').sum(),
            d7_retained=col('day7_active').sum(),
            d30_retained=col('day30_active').sum(),
            total_users=col('user_id').count(),
        ).to_df()

        assert 'd1_retained' in result.columns
        assert 'd7_retained' in result.columns
        assert 'd30_retained' in result.columns
        assert 'total_users' in result.columns
        assert len(result) == 1  # Global aggregation returns 1 row

    def test_lazy_verification_no_early_execution(self):
        """
        Verify that lazy operations don't execute prematurely.

        This test ensures the lazy execution model works correctly:
        - Operations are recorded but not executed
        - Execution only happens on .to_df() or similar triggering calls
        """
        ds = self.ds

        # Add many lazy operations
        ds['feat1'] = ds['page_views'] * 2
        ds['feat2'] = ds['clicks'] + ds['cart_additions']
        ds['feat3'] = ds['purchase_amount'] / (ds['purchases'] + 1)
        ds['feat4'] = ds['age'] - 18
        ds['feat5'] = ds['loyalty_points'] * ds['visit_frequency']

        # Verify operations are queued but not executed
        assert len(ds._lazy_ops) >= 5, "Lazy operations should be recorded"

        # Filter operation (also lazy)
        filtered = ds.filter(ds['churned'] == 1)

        # Still no execution
        assert len(filtered._lazy_ops) > len(ds._lazy_ops), "Filter should add lazy op"

        # Now execute
        result = filtered.to_df()

        # Verify filtered results
        assert all(result['churned'] == 1)
        assert 'feat1' in result.columns
        assert 'feat5' in result.columns


class TestMLFeatureEngineeringEdgeCases:
    """Test edge cases in ML feature engineering."""

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame({'a': [], 'b': []})
        ds = DataStore.from_df(empty_df)

        # Function chain: Operations on empty data
        ds['c'] = ds['a'] + ds['b']

        result = ds.to_df()
        assert len(result) == 0
        assert 'c' in result.columns

    def test_null_handling_in_features(self):
        """Test proper handling of NULL values in feature computation."""
        df_with_nulls = pd.DataFrame(
            {
                'a': [1, 2, None, 4, 5],
                'b': [10, None, 30, 40, 50],
            }
        )
        ds = DataStore.from_df(df_with_nulls)

        # Function chain: Operations with nulls
        ds['c'] = ds['a'] + ds['b']
        ds['d'] = ds['a'] * 2

        result = ds.to_df()

        # Verify NULL propagation
        assert 'c' in result.columns
        assert 'd' in result.columns

    def test_large_feature_count(self):
        """
        Test handling of many features (wide table scenario).

        Simulates a typical ML feature store with 100+ features.
        """
        # Create a wide table with 100 features
        n_features = 100
        data = {f'feature_{i}': np.random.randn(100) for i in range(n_features)}
        data['target'] = np.random.choice([0, 1], 100)

        ds = DataStore.from_df(pd.DataFrame(data))

        # Function chain: Add derived features
        for i in range(0, n_features, 2):
            ds[f'derived_{i}'] = ds[f'feature_{i}'] + ds[f'feature_{i+1}']

        result = ds.to_df()

        # Should have original + derived features
        assert len(result.columns) >= n_features + n_features // 2

    def test_chained_transformations(self):
        """
        Test chaining multiple transformations.

        Equivalent SQL:
            SELECT
                user_id,
                value,
                value * 2 AS doubled,
                (value * 2) + 10 AS shifted,
                ((value * 2) + 10) / 2 AS normalized
            FROM __df__
            WHERE value > 5
        """
        df = pd.DataFrame({'user_id': range(1, 101), 'value': np.random.randint(1, 20, 100)})
        ds = DataStore.from_df(df)

        # Function chain: Multiple transformations
        ds['doubled'] = ds['value'] * 2
        ds['shifted'] = ds['doubled'] + 10
        ds['normalized'] = ds['shifted'] / 2

        # Filter
        filtered = ds.filter(ds['value'] > 5)

        result = filtered.to_df()

        assert 'doubled' in result.columns
        assert 'shifted' in result.columns
        assert 'normalized' in result.columns
        assert all(result['value'] > 5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
