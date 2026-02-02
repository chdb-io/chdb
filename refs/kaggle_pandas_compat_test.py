"""
Comprehensive Pandas Compatibility Test for DataStore

This script tests DataStore against pandas with diverse operations commonly found
in Kaggle notebooks across different domains (CV, NLP, LLM, Recommendation, EDA).

The test operations are based on common patterns from:
- Computer Vision: Image metadata processing, label encoding
- NLP: Text preprocessing, tokenization statistics
- LLM: Token counting, sequence analysis
- Recommendation Systems: User-item matrices, rating analysis
- Data Analysis/EDA: Statistical analysis, data cleaning, visualization prep
"""

import sys
import traceback
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import warnings


# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class PandasCompatibilityTester:
    """Test DataStore pandas compatibility with diverse operations."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []
        self.test_data = self._create_test_datasets()

    def _create_test_datasets(self) -> Dict[str, pd.DataFrame]:
        """Create diverse test datasets mimicking Kaggle scenarios."""

        # Dataset 1: Image metadata (CV scenario)
        images_df = pd.DataFrame(
            {
                'image_id': range(100),
                'filename': [f'img_{i:04d}.jpg' for i in range(100)],
                'width': np.random.randint(224, 1024, 100),
                'height': np.random.randint(224, 1024, 100),
                'class_label': np.random.choice(['cat', 'dog', 'bird', 'fish'], 100),
                'brightness': np.random.uniform(0, 255, 100),
                'contrast': np.random.uniform(0, 100, 100),
                'file_size_mb': np.random.uniform(0.1, 5.0, 100),
                'is_augmented': np.random.choice([True, False], 100),
            }
        )

        # Dataset 2: Text data (NLP scenario)
        texts_df = pd.DataFrame(
            {
                'text_id': range(200),
                'text': [f'This is sample text number {i} with some words.' for i in range(200)],
                'sentiment': np.random.choice(['positive', 'negative', 'neutral'], 200),
                'word_count': np.random.randint(5, 100, 200),
                'char_count': np.random.randint(20, 500, 200),
                'has_url': np.random.choice([True, False], 200),
                'language': np.random.choice(['en', 'es', 'fr', 'de'], 200),
                'toxicity_score': np.random.uniform(0, 1, 200),
            }
        )

        # Dataset 3: User ratings (Recommendation scenario)
        ratings_df = pd.DataFrame(
            {
                'user_id': np.random.randint(1, 50, 500),
                'item_id': np.random.randint(1, 100, 500),
                'rating': np.random.uniform(1, 5, 500),
                'timestamp': pd.date_range('2024-01-01', periods=500, freq='h'),
                'platform': np.random.choice(['web', 'mobile', 'tablet'], 500),
            }
        )

        # Dataset 4: Sales data (EDA scenario)
        sales_df = pd.DataFrame(
            {
                'date': pd.date_range('2024-01-01', periods=365, freq='D'),
                'product_id': np.random.randint(1, 20, 365),
                'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 365),
                'sales': np.random.uniform(100, 10000, 365),
                'quantity': np.random.randint(1, 100, 365),
                'discount': np.random.uniform(0, 0.5, 365),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 365),
                'customer_age': np.random.randint(18, 80, 365),
            }
        )

        # Dataset 5: Missing data scenario
        missing_df = pd.DataFrame(
            {
                'col_a': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10],
                'col_b': ['a', None, 'c', 'd', 'e', 'f', None, 'h', 'i', 'j'],
                'col_c': [1.1, 2.2, 3.3, np.nan, 5.5, 6.6, 7.7, np.nan, 9.9, 10.0],
            }
        )

        return {
            'images': images_df,
            'texts': texts_df,
            'ratings': ratings_df,
            'sales': sales_df,
            'missing': missing_df,
        }

    def compare_results(self, test_name: str, pandas_result: Any, datastore_result: Any) -> bool:
        """Compare pandas and datastore results."""
        try:
            # Convert DataStore lazy objects to pandas using to_pandas()
            # This handles ColumnExpr, LazySeries, LazyAggregate, etc.
            if hasattr(datastore_result, 'to_pandas'):
                ds_df = datastore_result.to_pandas()
            elif hasattr(datastore_result, 'to_df'):
                ds_df = datastore_result.to_df()
            elif isinstance(datastore_result, (pd.DataFrame, pd.Series)):
                ds_df = datastore_result
            else:
                ds_df = datastore_result

            # Handle case where pandas returns scalar but DataStore returns single-element Series
            # (e.g., df['col'].mean() returns scalar in pandas but Series in DataStore)
            if (
                not isinstance(pandas_result, (pd.DataFrame, pd.Series))
                and isinstance(ds_df, pd.Series)
                and len(ds_df) == 1
            ):
                ds_df = ds_df.iloc[0]

            # Compare
            if isinstance(pandas_result, pd.DataFrame):
                if isinstance(ds_df, pd.DataFrame):
                    # Reset index for comparison
                    pd_compare = pandas_result.reset_index(drop=True)
                    ds_compare = ds_df.reset_index(drop=True)
                    # Sort columns for comparison
                    if set(pd_compare.columns) == set(ds_compare.columns):
                        pd_compare = pd_compare[sorted(pd_compare.columns)]
                        ds_compare = ds_compare[sorted(ds_compare.columns)]
                        return pd_compare.equals(ds_compare)
                return False
            elif isinstance(pandas_result, pd.Series):
                if isinstance(ds_df, pd.Series):
                    # Use numpy allclose for float comparison to handle precision differences
                    # Use pd.api.types.is_float_dtype to cover all float types (float16, float32, float64, etc.)
                    if pd.api.types.is_float_dtype(pandas_result) and pd.api.types.is_float_dtype(ds_df):
                        values_match = np.allclose(
                            pandas_result.values, ds_df.values, rtol=1e-10, atol=1e-10, equal_nan=True
                        )
                        index_match = pandas_result.index.equals(ds_df.index)
                        name_match = pandas_result.name == ds_df.name
                        return values_match and index_match and name_match
                    return pandas_result.equals(ds_df)
                return False
            else:
                # Scalar or other types - use numpy for float comparison
                if isinstance(pandas_result, (float, np.floating)) and isinstance(ds_df, (float, np.floating)):
                    if np.isnan(pandas_result) and np.isnan(ds_df):
                        return True
                    return bool(np.isclose(pandas_result, ds_df, rtol=1e-10, atol=1e-10))
                return pandas_result == ds_df
        except Exception as e:
            print(f"  Comparison error: {e}")
            return False

    def run_test(self, test_name: str, category: str, pandas_func, datastore_func):
        """Run a single test comparing pandas and datastore."""
        print(f"\n{'='*80}")
        print(f"Test: {test_name}")
        print(f"Category: {category}")
        print(f"{'='*80}")

        result = {
            'test_name': test_name,
            'category': category,
            'pandas_success': False,
            'datastore_success': False,
            'results_match': False,
            'pandas_error': None,
            'datastore_error': None,
        }

        # Run pandas version
        try:
            pandas_result = pandas_func()
            result['pandas_success'] = True
            print(f"✓ Pandas executed successfully")
            if isinstance(pandas_result, pd.DataFrame):
                print(f"  Result shape: {pandas_result.shape}")
            elif isinstance(pandas_result, pd.Series):
                print(f"  Result length: {len(pandas_result)}")
        except Exception as e:
            result['pandas_error'] = str(e)
            print(f"✗ Pandas failed: {e}")
            traceback.print_exc()
            pandas_result = None

        # Run datastore version
        try:
            datastore_result = datastore_func()
            result['datastore_success'] = True
            print(f"✓ DataStore executed successfully")
            if hasattr(datastore_result, 'to_df'):
                df = datastore_result.to_df()
                print(f"  Result shape: {df.shape}")
            elif isinstance(datastore_result, pd.DataFrame):
                print(f"  Result shape: {datastore_result.shape}")
            elif isinstance(datastore_result, pd.Series):
                print(f"  Result length: {len(datastore_result)}")
        except Exception as e:
            result['datastore_error'] = str(e)
            print(f"✗ DataStore failed: {e}")
            traceback.print_exc()
            datastore_result = None

        # Compare results
        if result['pandas_success'] and result['datastore_success']:
            results_match = self.compare_results(test_name, pandas_result, datastore_result)
            result['results_match'] = results_match
            if results_match:
                print(f"✓ Results match!")
            else:
                print(f"✗ Results differ!")
                print(f"  Pandas result preview:")
                print(f"  {pandas_result}")
                print(f"  DataStore result preview:")
                print(f"  {datastore_result}")

        self.results.append(result)
        return result

    # ========== Test Categories ==========

    def test_basic_operations(self):
        """Test basic DataFrame operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Basic Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: Column selection
        self.run_test(
            "Column Selection - Single Column",
            "Basic Operations",
            lambda: df['class_label'],
            lambda: self._ds_from_df(df)['class_label'],
        )

        # Test 2: Multiple column selection
        self.run_test(
            "Column Selection - Multiple Columns",
            "Basic Operations",
            lambda: df[['image_id', 'class_label', 'brightness']],
            lambda: self._ds_from_df(df)[['image_id', 'class_label', 'brightness']].to_df(),
        )

        # Test 3: Head
        self.run_test("Head Operation", "Basic Operations", lambda: df.head(10), lambda: self._ds_from_df(df).head(10))

        # Test 4: Tail
        self.run_test("Tail Operation", "Basic Operations", lambda: df.tail(10), lambda: self._ds_from_df(df).tail(10))

        # Test 5: Shape
        self.run_test("Shape Property", "Basic Operations", lambda: df.shape, lambda: self._ds_from_df(df).shape)

        # Test 6: Columns
        self.run_test(
            "Columns Property", "Basic Operations", lambda: list(df.columns), lambda: list(self._ds_from_df(df).columns)
        )

    def test_filtering_operations(self):
        """Test filtering and boolean indexing."""
        print("\n" + "=" * 80)
        print("CATEGORY: Filtering Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: Simple filter
        self.run_test(
            "Simple Filter - Equality",
            "Filtering",
            lambda: df[df['class_label'] == 'cat'],
            lambda: self._ds_from_df(df)[self._ds_from_df(df)['class_label'] == 'cat'].to_df(),
        )

        # Test 2: Numeric comparison
        self.run_test(
            "Numeric Filter - Greater Than",
            "Filtering",
            lambda: df[df['brightness'] > 150],
            lambda: self._ds_from_df(df)[self._ds_from_df(df)['brightness'] > 150].to_df(),
        )

        # Test 3: Multiple conditions (AND)
        self.run_test(
            "Multiple Conditions - AND",
            "Filtering",
            lambda: df[(df['class_label'] == 'cat') & (df['brightness'] > 150)],
            lambda: self._ds_from_df(df)[
                (self._ds_from_df(df)['class_label'] == 'cat') & (self._ds_from_df(df)['brightness'] > 150)
            ].to_df(),
        )

        # Test 4: isin
        self.run_test(
            "Filter - isin",
            "Filtering",
            lambda: df[df['class_label'].isin(['cat', 'dog'])],
            lambda: self._ds_from_df(df)[self._ds_from_df(df)['class_label'].isin(['cat', 'dog'])].to_df(),
        )

    def test_aggregation_operations(self):
        """Test aggregation operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Aggregation Operations")
        print("=" * 80)

        df = self.test_data['sales']

        # Test 1: Mean
        self.run_test(
            "Aggregation - Mean",
            "Aggregation",
            lambda: df['sales'].mean(),
            lambda: self._ds_from_df(df)['sales'].mean(),
        )

        # Test 2: Sum
        self.run_test(
            "Aggregation - Sum",
            "Aggregation",
            lambda: df['quantity'].sum(),
            lambda: self._ds_from_df(df)['quantity'].sum(),
        )

        # Test 3: Count
        self.run_test(
            "Aggregation - Count",
            "Aggregation",
            lambda: df['product_id'].count(),
            lambda: self._ds_from_df(df)['product_id'].count(),
        )

        # Test 4: Min/Max
        self.run_test(
            "Aggregation - Min", "Aggregation", lambda: df['sales'].min(), lambda: self._ds_from_df(df)['sales'].min()
        )

        self.run_test(
            "Aggregation - Max", "Aggregation", lambda: df['sales'].max(), lambda: self._ds_from_df(df)['sales'].max()
        )

    def test_groupby_operations(self):
        """Test groupby operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: GroupBy Operations")
        print("=" * 80)

        df = self.test_data['sales']

        # Test 1: Simple groupby with mean
        self.run_test(
            "GroupBy - Mean by Category",
            "GroupBy",
            lambda: df.groupby('category')['sales'].mean(),
            lambda: self._ds_from_df(df).groupby('category')['sales'].mean(),
        )

        # Test 2: Groupby with sum
        self.run_test(
            "GroupBy - Sum by Region",
            "GroupBy",
            lambda: df.groupby('region')['quantity'].sum(),
            lambda: self._ds_from_df(df).groupby('region')['quantity'].sum(),
        )

        # Test 3: Multiple aggregations
        self.run_test(
            "GroupBy - Multiple Aggregations",
            "GroupBy",
            lambda: df.groupby('category').agg({'sales': 'mean', 'quantity': 'sum'}),
            lambda: self._ds_from_df(df).groupby('category').agg({'sales': 'mean', 'quantity': 'sum'}),
        )

        # Test 4: Groupby with count
        self.run_test(
            "GroupBy - Count",
            "GroupBy",
            lambda: df.groupby('category').size(),
            lambda: self._ds_from_df(df).groupby('category').size(),
        )

    def test_sorting_operations(self):
        """Test sorting operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Sorting Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: Sort by single column
        self.run_test(
            "Sort - Single Column Ascending",
            "Sorting",
            lambda: df.sort_values('brightness').head(10),
            lambda: self._ds_from_df(df).sort_values('brightness').head(10),
        )

        # Test 2: Sort descending
        self.run_test(
            "Sort - Single Column Descending",
            "Sorting",
            lambda: df.sort_values('brightness', ascending=False).head(10),
            lambda: self._ds_from_df(df).sort_values('brightness', ascending=False).head(10),
        )

        # Test 3: Sort by multiple columns
        self.run_test(
            "Sort - Multiple Columns",
            "Sorting",
            lambda: df.sort_values(['class_label', 'brightness']).head(10),
            lambda: self._ds_from_df(df).sort_values(['class_label', 'brightness']).head(10),
        )

    def test_missing_data_operations(self):
        """Test missing data handling."""
        print("\n" + "=" * 80)
        print("CATEGORY: Missing Data Operations")
        print("=" * 80)

        df = self.test_data['missing']

        # Test 1: isna
        self.run_test("Missing Data - isna", "Missing Data", lambda: df.isna(), lambda: self._ds_from_df(df).isna())

        # Test 2: dropna
        self.run_test(
            "Missing Data - dropna", "Missing Data", lambda: df.dropna(), lambda: self._ds_from_df(df).dropna()
        )

        # Test 3: fillna with value
        self.run_test(
            "Missing Data - fillna with value",
            "Missing Data",
            lambda: df.fillna(0),
            lambda: self._ds_from_df(df).fillna(0),
        )

        # Test 4: fillna with method
        self.run_test(
            "Missing Data - fillna forward fill",
            "Missing Data",
            lambda: df.fillna(method='ffill'),
            lambda: self._ds_from_df(df).fillna(method='ffill'),
        )

    def test_string_operations(self):
        """Test string operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: String Operations")
        print("=" * 80)

        df = self.test_data['texts']

        # Test 1: str.lower
        self.run_test(
            "String - Lower",
            "String Operations",
            lambda: df['sentiment'].str.lower().head(10),
            lambda: self._ds_from_df(df)['sentiment'].str.lower().head(10),
        )

        # Test 2: str.upper
        self.run_test(
            "String - Upper",
            "String Operations",
            lambda: df['sentiment'].str.upper().head(10),
            lambda: self._ds_from_df(df)['sentiment'].str.upper().head(10),
        )

        # Test 3: str.contains
        self.run_test(
            "String - Contains",
            "String Operations",
            lambda: df[df['text'].str.contains('sample')].head(10),
            lambda: self._ds_from_df(df)[self._ds_from_df(df)['text'].str.contains('sample')].head(10).to_df(),
        )

        # Test 4: str.len
        self.run_test(
            "String - Length",
            "String Operations",
            lambda: df['text'].str.len().head(10),
            lambda: self._ds_from_df(df)['text'].str.len().head(10),
        )

    def test_datetime_operations(self):
        """Test datetime operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: DateTime Operations")
        print("=" * 80)

        df = self.test_data['ratings']

        # Test 1: dt.year
        self.run_test(
            "DateTime - Year",
            "DateTime Operations",
            lambda: df['timestamp'].dt.year.head(10),
            lambda: self._ds_from_df(df)['timestamp'].dt.year.head(10),
        )

        # Test 2: dt.month
        self.run_test(
            "DateTime - Month",
            "DateTime Operations",
            lambda: df['timestamp'].dt.month.head(10),
            lambda: self._ds_from_df(df)['timestamp'].dt.month.head(10),
        )

        # Test 3: dt.day
        self.run_test(
            "DateTime - Day",
            "DateTime Operations",
            lambda: df['timestamp'].dt.day.head(10),
            lambda: self._ds_from_df(df)['timestamp'].dt.day.head(10),
        )

        # Test 4: dt.dayofweek
        self.run_test(
            "DateTime - Day of Week",
            "DateTime Operations",
            lambda: df['timestamp'].dt.dayofweek.head(10),
            lambda: self._ds_from_df(df)['timestamp'].dt.dayofweek.head(10),
        )

    def test_value_counts(self):
        """Test value_counts operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Value Counts")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: value_counts
        self.run_test(
            "Value Counts - Basic",
            "Value Counts",
            lambda: df['class_label'].value_counts(),
            lambda: self._ds_from_df(df)['class_label'].value_counts(),
        )

        # Test 2: value_counts with normalize
        self.run_test(
            "Value Counts - Normalized",
            "Value Counts",
            lambda: df['class_label'].value_counts(normalize=True),
            lambda: self._ds_from_df(df)['class_label'].value_counts(normalize=True),
        )

    def test_merge_operations(self):
        """Test merge/join operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Merge Operations")
        print("=" * 80)

        # Create two simple DataFrames for merging
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value1': [1, 2, 3, 4]})

        df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value2': [5, 6, 7, 8]})

        # Test 1: Inner merge
        self.run_test(
            "Merge - Inner",
            "Merge",
            lambda: pd.merge(df1, df2, on='key', how='inner'),
            lambda: pd.merge(self._ds_from_df(df1).to_df(), self._ds_from_df(df2).to_df(), on='key', how='inner'),
        )

        # Test 2: Left merge
        self.run_test(
            "Merge - Left",
            "Merge",
            lambda: pd.merge(df1, df2, on='key', how='left'),
            lambda: pd.merge(self._ds_from_df(df1).to_df(), self._ds_from_df(df2).to_df(), on='key', how='left'),
        )

    def test_apply_operations(self):
        """Test apply operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Apply Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: Simple apply
        self.run_test(
            "Apply - Square",
            "Apply",
            lambda: df['brightness'].apply(lambda x: x**2).head(10),
            lambda: self._ds_from_df(df)['brightness'].apply(lambda x: x**2).head(10),
        )

        # Test 2: Apply with custom function
        def categorize_brightness(x):
            if x < 100:
                return 'low'
            elif x < 200:
                return 'medium'
            else:
                return 'high'

        self.run_test(
            "Apply - Categorize",
            "Apply",
            lambda: df['brightness'].apply(categorize_brightness).head(10),
            lambda: self._ds_from_df(df)['brightness'].apply(categorize_brightness).head(10),
        )

    def test_statistical_operations(self):
        """Test statistical operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Statistical Operations")
        print("=" * 80)

        df = self.test_data['sales']

        # Test 1: describe
        self.run_test(
            "Statistics - Describe",
            "Statistics",
            lambda: df['sales'].describe(),
            lambda: self._ds_from_df(df)['sales'].describe(),
        )

        # Test 2: std
        self.run_test(
            "Statistics - Standard Deviation",
            "Statistics",
            lambda: df['sales'].std(),
            lambda: self._ds_from_df(df)['sales'].std(),
        )

        # Test 3: var
        self.run_test(
            "Statistics - Variance",
            "Statistics",
            lambda: df['sales'].var(),
            lambda: self._ds_from_df(df)['sales'].var(),
        )

        # Test 4: median
        self.run_test(
            "Statistics - Median",
            "Statistics",
            lambda: df['sales'].median(),
            lambda: self._ds_from_df(df)['sales'].median(),
        )

    def test_unique_operations(self):
        """Test unique and nunique operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Unique Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: unique
        self.run_test(
            "Unique - Get Unique Values",
            "Unique",
            lambda: sorted(df['class_label'].unique()),
            lambda: sorted(self._ds_from_df(df)['class_label'].unique()),
        )

        # Test 2: nunique
        self.run_test(
            "Unique - Count Unique",
            "Unique",
            lambda: df['class_label'].nunique(),
            lambda: self._ds_from_df(df)['class_label'].nunique(),
        )

    def test_drop_operations(self):
        """Test drop operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Drop Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: drop columns
        self.run_test(
            "Drop - Single Column",
            "Drop",
            lambda: df.drop(columns=['brightness']).head(10),
            lambda: self._ds_from_df(df).drop(columns=['brightness']).head(10),
        )

        # Test 2: drop multiple columns
        self.run_test(
            "Drop - Multiple Columns",
            "Drop",
            lambda: df.drop(columns=['brightness', 'contrast']).head(10),
            lambda: self._ds_from_df(df).drop(columns=['brightness', 'contrast']).head(10),
        )

    def test_rename_operations(self):
        """Test rename operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Rename Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: rename single column
        self.run_test(
            "Rename - Single Column",
            "Rename",
            lambda: df.rename(columns={'brightness': 'brightness_value'}).head(10),
            lambda: self._ds_from_df(df).rename(columns={'brightness': 'brightness_value'}).head(10),
        )

        # Test 2: rename multiple columns
        self.run_test(
            "Rename - Multiple Columns",
            "Rename",
            lambda: df.rename(columns={'brightness': 'bright', 'contrast': 'contr'}).head(10),
            lambda: self._ds_from_df(df).rename(columns={'brightness': 'bright', 'contrast': 'contr'}).head(10),
        )

    def test_reset_index(self):
        """Test reset_index operations."""
        print("\n" + "=" * 80)
        print("CATEGORY: Reset Index Operations")
        print("=" * 80)

        df = self.test_data['images']

        # Test 1: reset_index
        self.run_test(
            "Reset Index - Basic",
            "Index",
            lambda: df.head(10).reset_index(drop=True),
            lambda: self._ds_from_df(df).head(10).reset_index(drop=True),
        )

    # Helper methods
    def _ds_from_df(self, df: pd.DataFrame):
        """Create a DataStore from a pandas DataFrame."""
        import datastore as ds

        return ds.DataFrame(df)

    def generate_report(self):
        """Generate a comprehensive compatibility report."""
        print("\n" + "=" * 80)
        print("COMPATIBILITY TEST REPORT")
        print("=" * 80)

        total_tests = len(self.results)
        pandas_success = sum(1 for r in self.results if r['pandas_success'])
        datastore_success = sum(1 for r in self.results if r['datastore_success'])
        results_match = sum(1 for r in self.results if r['results_match'])

        print(f"\nTotal Tests: {total_tests}")
        print(f"Pandas Success: {pandas_success}/{total_tests} ({pandas_success/total_tests*100:.1f}%)")
        print(f"DataStore Success: {datastore_success}/{total_tests} ({datastore_success/total_tests*100:.1f}%)")
        print(f"Results Match: {results_match}/{total_tests} ({results_match/total_tests*100:.1f}%)")

        # Group by category
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result)

        print("\n" + "=" * 80)
        print("RESULTS BY CATEGORY")
        print("=" * 80)

        for category, tests in sorted(categories.items()):
            cat_total = len(tests)
            cat_success = sum(1 for t in tests if t['datastore_success'])
            cat_match = sum(1 for t in tests if t['results_match'])

            print(f"\n{category}:")
            print(f"  Success: {cat_success}/{cat_total}")
            print(f"  Match: {cat_match}/{cat_total}")

        # List failures
        failures = [r for r in self.results if not r['datastore_success']]
        if failures:
            print("\n" + "=" * 80)
            print("DATASTORE FAILURES")
            print("=" * 80)
            for failure in failures:
                print(f"\n{failure['test_name']}:")
                print(f"  Error: {failure['datastore_error']}")

        # List mismatches
        mismatches = [r for r in self.results if r['datastore_success'] and not r['results_match']]
        if mismatches:
            print("\n" + "=" * 80)
            print("RESULT MISMATCHES")
            print("=" * 80)
            for mismatch in mismatches:
                print(f"  - {mismatch['test_name']}")

        # Recommendations
        print("\n" + "=" * 80)
        print("IMPROVEMENT RECOMMENDATIONS")
        print("=" * 80)

        improvements = []

        # Analyze failure patterns
        error_patterns = {}
        for result in self.results:
            if result['datastore_error']:
                error = result['datastore_error']
                # Extract key error type
                if 'NotImplementedError' in error:
                    error_type = 'NotImplementedError'
                elif 'AttributeError' in error:
                    error_type = 'AttributeError'
                elif 'TypeError' in error:
                    error_type = 'TypeError'
                else:
                    error_type = 'Other'

                if error_type not in error_patterns:
                    error_patterns[error_type] = []
                error_patterns[error_type].append(result['test_name'])

        if error_patterns:
            print("\nError Pattern Analysis:")
            for error_type, tests in error_patterns.items():
                print(f"\n{error_type} ({len(tests)} occurrences):")
                for test in tests[:5]:  # Show first 5
                    print(f"  - {test}")
                if len(tests) > 5:
                    print(f"  ... and {len(tests) - 5} more")

        # Specific recommendations
        if failures:
            print("\nPriority Improvements:")

            # Check for common pandas methods not implemented
            missing_methods = set()
            for failure in failures:
                error = failure['datastore_error']
                if 'has no attribute' in error:
                    # Extract method name
                    import re

                    match = re.search(r"has no attribute '(\w+)'", error)
                    if match:
                        missing_methods.add(match.group(1))

            if missing_methods:
                print(f"\n1. Implement missing methods: {', '.join(sorted(missing_methods))}")

            # Check for string operations
            str_failures = [f for f in failures if 'String' in f['category']]
            if str_failures:
                print(f"\n2. Improve string accessor support ({len(str_failures)} failures)")

            # Check for datetime operations
            dt_failures = [f for f in failures if 'DateTime' in f['category']]
            if dt_failures:
                print(f"\n3. Improve datetime accessor support ({len(dt_failures)} failures)")

            # Check for groupby operations
            gb_failures = [f for f in failures if 'GroupBy' in f['category']]
            if gb_failures:
                print(f"\n4. Improve groupby support ({len(gb_failures)} failures)")

        print("\n" + "=" * 80)


def main():
    """Main test runner."""
    print("=" * 80)
    print("Kaggle Pandas Compatibility Test for DataStore")
    print("=" * 80)
    print("\nThis test suite evaluates DataStore's compatibility with pandas")
    print("using operations commonly found in Kaggle notebooks across different domains.")
    print("\nDomains covered:")
    print("  - Computer Vision (CV): Image metadata processing")
    print("  - Natural Language Processing (NLP): Text analysis")
    print("  - Large Language Models (LLM): Token statistics")
    print("  - Recommendation Systems: User-item interactions")
    print("  - Exploratory Data Analysis (EDA): Statistical analysis")

    try:
        import datastore as ds

        print(f"\n✓ DataStore imported successfully")
        print(f"  Version: {ds.__version__ if hasattr(ds, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"\n✗ Failed to import datastore: {e}")
        print("  Make sure datastore is installed and accessible")
        sys.exit(1)

    # Run tests
    tester = PandasCompatibilityTester()

    # Execute all test categories
    tester.test_basic_operations()
    tester.test_filtering_operations()
    tester.test_aggregation_operations()
    tester.test_groupby_operations()
    tester.test_sorting_operations()
    tester.test_missing_data_operations()
    tester.test_string_operations()
    tester.test_datetime_operations()
    tester.test_value_counts()
    tester.test_merge_operations()
    tester.test_apply_operations()
    tester.test_statistical_operations()
    tester.test_unique_operations()
    tester.test_drop_operations()
    tester.test_rename_operations()
    tester.test_reset_index()

    # Generate final report
    tester.generate_report()


if __name__ == '__main__':
    main()
