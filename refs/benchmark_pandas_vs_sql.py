#!/usr/bin/env python3
"""
Benchmark: Pandas vs chDB vs DuckDB SQL operations on DataFrames

This benchmark compares the performance of:
1. Pure Pandas operations (current implementation)
2. chDB SQL execution on DataFrames (using Python() table function)
3. DuckDB SQL execution on DataFrames

Operations tested:
- Filter (single condition)
- Filter (multiple conditions)
- Select columns
- Sort
- Aggregation (GROUP BY)
- Combined operations
"""

import time
import pandas as pd
import numpy as np
import chdb
import duckdb
import matplotlib.pyplot as plt
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class BenchmarkResult:
    operation: str
    data_size: int
    pandas_time: float
    chdb_time: float
    duckdb_time: float

    @property
    def fastest(self) -> str:
        times = {'Pandas': self.pandas_time, 'chDB': self.chdb_time, 'DuckDB': self.duckdb_time}
        winner = min(times, key=times.get)
        winner_time = times[winner]
        # Calculate speedup vs slowest
        slowest_time = max(times.values())
        speedup = slowest_time / winner_time if winner_time > 0 else float('inf')
        return f"{winner} ({speedup:.2f}x)"


def generate_test_data(n_rows: int) -> pd.DataFrame:
    """Generate test DataFrame with various column types."""
    np.random.seed(42)

    df = pd.DataFrame(
        {
            'id': range(n_rows),
            'int_col': np.random.randint(0, 1000, n_rows),
            'float_col': np.random.uniform(0, 1000, n_rows),
            'str_col': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_rows),
            'category': np.random.choice(
                ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8', 'cat9', 'cat10'], n_rows
            ),
            'bool_col': np.random.choice([True, False], n_rows),
            'date_col': pd.date_range('2020-01-01', periods=n_rows, freq='s')[:n_rows],
        }
    )

    return df


def time_operation(func: Callable, n_runs: int = 5) -> float:
    """Time an operation, return average time in milliseconds."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func()
        # Force evaluation for lazy results
        if isinstance(result, pd.DataFrame):
            _ = len(result)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Remove outliers and return average
    times.sort()
    if len(times) > 2:
        times = times[1:-1]  # Remove min and max
    return sum(times) / len(times)


# Global connections for reuse (initialized in main())
conn = None  # chdb connection
duck_conn = None  # duckdb connection


class Benchmark:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.n_rows = len(df)

    # ==================== Filter Operations ====================

    def pandas_filter_single(self) -> pd.DataFrame:
        return self.df[self.df['int_col'] > 500]

    def chdb_filter_single(self) -> pd.DataFrame:
        df = self.df  # Make df available in local scope for chDB
        return conn.query("SELECT * FROM Python(df) WHERE int_col > 500", 'DataFrame')

    def duckdb_filter_single(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT * FROM df WHERE int_col > 500").df()

    def pandas_filter_multiple(self) -> pd.DataFrame:
        return self.df[(self.df['int_col'] > 300) & (self.df['int_col'] < 700) & (self.df['str_col'] == 'A')]

    def chdb_filter_multiple(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            "SELECT * FROM Python(df) WHERE int_col > 300 AND int_col < 700 AND str_col = 'A'", 'DataFrame'
        )

    def duckdb_filter_multiple(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT * FROM df WHERE int_col > 300 AND int_col < 700 AND str_col = 'A'").df()

    # ==================== Select Operations ====================

    def pandas_select_columns(self) -> pd.DataFrame:
        return self.df[['id', 'int_col', 'str_col']]

    def chdb_select_columns(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT id, int_col, str_col FROM Python(df)", 'DataFrame')

    def duckdb_select_columns(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT id, int_col, str_col FROM df").df()

    # ==================== Sort Operations ====================

    def pandas_sort_single(self) -> pd.DataFrame:
        return self.df.sort_values('int_col')

    def chdb_sort_single(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) ORDER BY int_col", 'DataFrame')

    def duckdb_sort_single(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT * FROM df ORDER BY int_col").df()

    def pandas_sort_multiple(self) -> pd.DataFrame:
        return self.df.sort_values(['str_col', 'int_col'], ascending=[True, False])

    def chdb_sort_multiple(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) ORDER BY str_col ASC, int_col DESC", 'DataFrame')

    def duckdb_sort_multiple(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT * FROM df ORDER BY str_col ASC, int_col DESC").df()

    # ==================== Aggregation Operations ====================

    def pandas_groupby_count(self) -> pd.DataFrame:
        return self.df.groupby('str_col').size().reset_index(name='count')

    def chdb_groupby_count(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT str_col, count(*) as count FROM Python(df) GROUP BY str_col", 'DataFrame')

    def duckdb_groupby_count(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT str_col, count(*) as count FROM df GROUP BY str_col").df()

    def pandas_groupby_agg(self) -> pd.DataFrame:
        return (
            self.df.groupby('category')
            .agg({'int_col': ['sum', 'mean', 'max'], 'float_col': ['sum', 'mean']})
            .reset_index()
        )

    def chdb_groupby_agg(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            """SELECT category, 
                      sum(int_col) as int_sum, 
                      avg(int_col) as int_mean,
                      max(int_col) as int_max,
                      sum(float_col) as float_sum,
                      avg(float_col) as float_mean
               FROM Python(df) 
               GROUP BY category""",
            'DataFrame',
        )

    def duckdb_groupby_agg(self) -> pd.DataFrame:
        return duck_conn.execute(
            """SELECT category, 
                      sum(int_col) as int_sum, 
                      avg(int_col) as int_mean,
                      max(int_col) as int_max,
                      sum(float_col) as float_sum,
                      avg(float_col) as float_mean
               FROM df 
               GROUP BY category"""
        ).df()

    # ==================== Combined Operations ====================

    def pandas_combined(self) -> pd.DataFrame:
        result = self.df[self.df['int_col'] > 200]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result.head(100)

    def chdb_combined(self) -> pd.DataFrame:
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, float_col 
               FROM Python(df) 
               WHERE int_col > 200 
               ORDER BY int_col DESC 
               LIMIT 100""",
            'DataFrame',
        )

    def duckdb_combined(self) -> pd.DataFrame:
        return duck_conn.execute(
            """SELECT id, int_col, str_col, float_col 
               FROM df 
               WHERE int_col > 200 
               ORDER BY int_col DESC 
               LIMIT 100"""
        ).df()

    # ==================== Head/Limit Operations ====================

    def pandas_head(self) -> pd.DataFrame:
        return self.df.head(1000)

    def chdb_head(self) -> pd.DataFrame:
        df = self.df
        return conn.query("SELECT * FROM Python(df) LIMIT 1000", 'DataFrame')

    def duckdb_head(self) -> pd.DataFrame:
        return duck_conn.execute("SELECT * FROM df LIMIT 1000").df()

    # ==================== Multi-Step Operations ====================
    # These simulate the DataStore lazy execution pattern where multiple
    # operations are chained together

    def pandas_multi_filter(self) -> pd.DataFrame:
        """Multiple filter operations in sequence (like DataStore Phase 2)."""
        result = self.df[self.df['int_col'] > 200]
        result = result[result['int_col'] < 800]
        result = result[result['str_col'].isin(['A', 'B', 'C'])]
        result = result[result['float_col'] > 100]
        return result

    def chdb_multi_filter(self) -> pd.DataFrame:
        """Single SQL with all filters combined."""
        df = self.df
        return conn.query(
            """SELECT * FROM Python(df) 
               WHERE int_col > 200 
                 AND int_col < 800 
                 AND str_col IN ('A', 'B', 'C')
                 AND float_col > 100""",
            'DataFrame',
        )

    def duckdb_multi_filter(self) -> pd.DataFrame:
        """Single SQL with all filters combined."""
        return duck_conn.execute(
            """SELECT * FROM df 
               WHERE int_col > 200 
                 AND int_col < 800 
                 AND str_col IN ('A', 'B', 'C')
                 AND float_col > 100"""
        ).df()

    def pandas_filter_select_sort(self) -> pd.DataFrame:
        """Filter -> Select columns -> Sort (common pattern)."""
        result = self.df[self.df['int_col'] > 300]
        result = result[['id', 'int_col', 'str_col', 'float_col']]
        result = result.sort_values('int_col', ascending=False)
        return result

    def chdb_filter_select_sort(self) -> pd.DataFrame:
        """Single SQL for filter + select + sort."""
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, float_col 
               FROM Python(df) 
               WHERE int_col > 300 
               ORDER BY int_col DESC""",
            'DataFrame',
        )

    def duckdb_filter_select_sort(self) -> pd.DataFrame:
        """Single SQL for filter + select + sort."""
        return duck_conn.execute(
            """SELECT id, int_col, str_col, float_col 
               FROM df 
               WHERE int_col > 300 
               ORDER BY int_col DESC"""
        ).df()

    def pandas_filter_groupby_sort(self) -> pd.DataFrame:
        """Filter -> GroupBy -> Sort (analytics pattern)."""
        result = self.df[self.df['int_col'] > 200]
        result = result.groupby('category').agg({'int_col': 'sum', 'float_col': 'mean'}).reset_index()
        result.columns = ['category', 'int_sum', 'float_avg']
        result = result.sort_values('int_sum', ascending=False)
        return result

    def chdb_filter_groupby_sort(self) -> pd.DataFrame:
        """Single SQL for filter + groupby + sort."""
        df = self.df
        return conn.query(
            """SELECT category, sum(int_col) as int_sum, avg(float_col) as float_avg
               FROM Python(df) 
               WHERE int_col > 200 
               GROUP BY category
               ORDER BY int_sum DESC""",
            'DataFrame',
        )

    def duckdb_filter_groupby_sort(self) -> pd.DataFrame:
        """Single SQL for filter + groupby + sort."""
        return duck_conn.execute(
            """SELECT category, sum(int_col) as int_sum, avg(float_col) as float_avg
               FROM df 
               WHERE int_col > 200 
               GROUP BY category
               ORDER BY int_sum DESC"""
        ).df()

    def pandas_complex_pipeline(self) -> pd.DataFrame:
        """Complex multi-step pipeline (simulates real DataStore usage)."""
        # Step 1: Filter
        result = self.df[self.df['int_col'] > 100]
        # Step 2: Add computed column
        result = result.copy()
        result['computed'] = result['int_col'] * 2 + result['float_col']
        # Step 3: Another filter on computed column
        result = result[result['computed'] > 500]
        # Step 4: Select columns
        result = result[['id', 'int_col', 'str_col', 'computed']]
        # Step 5: Sort
        result = result.sort_values('computed', ascending=False)
        # Step 6: Limit
        return result.head(500)

    def chdb_complex_pipeline(self) -> pd.DataFrame:
        """Single SQL for complex pipeline."""
        df = self.df
        return conn.query(
            """SELECT id, int_col, str_col, (int_col * 2 + float_col) as computed
               FROM Python(df) 
               WHERE int_col > 100 
                 AND (int_col * 2 + float_col) > 500
               ORDER BY computed DESC
               LIMIT 500""",
            'DataFrame',
        )

    def duckdb_complex_pipeline(self) -> pd.DataFrame:
        """Single SQL for complex pipeline."""
        return duck_conn.execute(
            """SELECT id, int_col, str_col, (int_col * 2 + float_col) as computed
               FROM df 
               WHERE int_col > 100 
                 AND (int_col * 2 + float_col) > 500
               ORDER BY computed DESC
               LIMIT 500"""
        ).df()

    def pandas_chain_5_filters(self) -> pd.DataFrame:
        """5 sequential filter operations."""
        result = self.df[self.df['int_col'] > 100]
        result = result[result['int_col'] < 900]
        result = result[result['float_col'] > 50]
        result = result[result['float_col'] < 950]
        result = result[result['str_col'] != 'E']
        return result

    def chdb_chain_5_filters(self) -> pd.DataFrame:
        """Single SQL with 5 conditions."""
        df = self.df
        return conn.query(
            """SELECT * FROM Python(df) 
               WHERE int_col > 100 
                 AND int_col < 900
                 AND float_col > 50
                 AND float_col < 950
                 AND str_col != 'E'""",
            'DataFrame',
        )

    def duckdb_chain_5_filters(self) -> pd.DataFrame:
        """Single SQL with 5 conditions."""
        return duck_conn.execute(
            """SELECT * FROM df 
               WHERE int_col > 100 
                 AND int_col < 900
                 AND float_col > 50
                 AND float_col < 950
                 AND str_col != 'E'"""
        ).df()


def run_benchmarks(data_sizes: List[int], n_runs: int = 5) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes."""
    results = []

    operations = [
        # Single operations
        ('Filter (single)', 'pandas_filter_single', 'chdb_filter_single', 'duckdb_filter_single'),
        ('Filter (multiple)', 'pandas_filter_multiple', 'chdb_filter_multiple', 'duckdb_filter_multiple'),
        ('Select columns', 'pandas_select_columns', 'chdb_select_columns', 'duckdb_select_columns'),
        ('Sort (single)', 'pandas_sort_single', 'chdb_sort_single', 'duckdb_sort_single'),
        ('Sort (multiple)', 'pandas_sort_multiple', 'chdb_sort_multiple', 'duckdb_sort_multiple'),
        ('GroupBy count', 'pandas_groupby_count', 'chdb_groupby_count', 'duckdb_groupby_count'),
        ('GroupBy agg', 'pandas_groupby_agg', 'chdb_groupby_agg', 'duckdb_groupby_agg'),
        ('Combined ops', 'pandas_combined', 'chdb_combined', 'duckdb_combined'),
        ('Head/Limit', 'pandas_head', 'chdb_head', 'duckdb_head'),
        # Multi-step operations (simulates DataStore lazy execution)
        ('Multi-filter (4x)', 'pandas_multi_filter', 'chdb_multi_filter', 'duckdb_multi_filter'),
        ('Filter+Select+Sort', 'pandas_filter_select_sort', 'chdb_filter_select_sort', 'duckdb_filter_select_sort'),
        ('Filter+GroupBy+Sort', 'pandas_filter_groupby_sort', 'chdb_filter_groupby_sort', 'duckdb_filter_groupby_sort'),
        ('Complex pipeline', 'pandas_complex_pipeline', 'chdb_complex_pipeline', 'duckdb_complex_pipeline'),
        ('Chain 5 filters', 'pandas_chain_5_filters', 'chdb_chain_5_filters', 'duckdb_chain_5_filters'),
    ]

    for size in data_sizes:
        print(f"\n{'='*60}")
        print(f"Generating {size:,} rows of test data...")
        df = generate_test_data(size)
        benchmark = Benchmark(df)

        # Register DataFrame with DuckDB for this benchmark run
        duck_conn.register('df', df)

        print(f"Running benchmarks (n_runs={n_runs})...")

        for op_name, pandas_method, chdb_method, duckdb_method in operations:
            pandas_func = getattr(benchmark, pandas_method)
            chdb_func = getattr(benchmark, chdb_method)
            duckdb_func = getattr(benchmark, duckdb_method)

            # Warm up
            try:
                pandas_func()
                chdb_func()
                duckdb_func()
            except Exception as e:
                print(f"  Skipping {op_name}: {e}")
                continue

            # Benchmark
            pandas_time = time_operation(pandas_func, n_runs)
            chdb_time = time_operation(chdb_func, n_runs)
            duckdb_time = time_operation(duckdb_func, n_runs)

            result = BenchmarkResult(
                operation=op_name,
                data_size=size,
                pandas_time=pandas_time,
                chdb_time=chdb_time,
                duckdb_time=duckdb_time,
            )
            results.append(result)

            print(
                f"  {op_name:20s}: Pandas={pandas_time:8.2f}ms, chDB={chdb_time:8.2f}ms, DuckDB={duckdb_time:8.2f}ms -> {result.fastest}"
            )

    return results


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of results."""
    print("\n" + "=" * 120)
    print("BENCHMARK SUMMARY")
    print("=" * 120)

    # Group by data size
    sizes = sorted(set(r.data_size for r in results))
    operations = sorted(set(r.operation for r in results), key=lambda x: [r.operation for r in results].index(x))

    # Print header
    print(f"\n{'Operation':<20}", end='')
    for size in sizes:
        print(f" | {size:>14,} rows", end='')
    print()
    print("-" * 20, end='')
    for _ in sizes:
        print("-" + "-" * 19, end='')
    print()

    # Print results
    for op in operations:
        print(f"{op:<20}", end='')
        for size in sizes:
            matching = [r for r in results if r.operation == op and r.data_size == size]
            if matching:
                r = matching[0]
                times = {'Pandas': r.pandas_time, 'chDB': r.chdb_time, 'DuckDB': r.duckdb_time}
                winner = min(times, key=times.get)
                winner_time = times[winner]
                slowest_time = max(times.values())
                ratio = slowest_time / winner_time if winner_time > 0 else float('inf')
                print(f" | {winner:>6} {ratio:>5.1f}x  ", end='')
            else:
                print(f" | {'N/A':>14}", end='')
        print()

    # Print detailed times
    print("\n" + "=" * 120)
    print("DETAILED TIMES (milliseconds)")
    print("=" * 120)

    print(f"\n{'Operation':<20} | {'Size':>10} | {'Pandas':>10} | {'chDB':>10} | {'DuckDB':>10} | {'Winner':>18}")
    print("-" * 95)

    for r in results:
        print(
            f"{r.operation:<20} | {r.data_size:>10,} | {r.pandas_time:>10.2f} | {r.chdb_time:>10.2f} | {r.duckdb_time:>10.2f} | {r.fastest:>18}"
        )


def main():
    global conn, duck_conn

    print("=" * 60)
    print("Pandas vs chDB vs DuckDB Benchmark")
    print("=" * 60)

    # Initialize chdb session to avoid ~15ms per-query initialization overhead
    conn = chdb.connect()
    print(f"chdb version: {chdb.__version__}")
    print("chdb session initialized (using conn.query() for best performance)")

    # Initialize duckdb connection
    duck_conn = duckdb.connect()
    print(f"duckdb version: {duckdb.__version__}")
    print("duckdb connection initialized")

    # Test different data sizes
    data_sizes = [100_000, 1_000_000, 10_000_000]

    # Run benchmarks
    results = run_benchmarks(data_sizes, n_runs=5)

    # Print summary
    print_summary(results)

    # Recommendations
    print("\n" + "=" * 120)
    print("RECOMMENDATIONS")
    print("=" * 120)

    # Analyze results - count wins for each engine
    def get_winner(r):
        times = {'Pandas': r.pandas_time, 'chDB': r.chdb_time, 'DuckDB': r.duckdb_time}
        return min(times, key=times.get)

    pandas_wins = sum(1 for r in results if get_winner(r) == 'Pandas')
    chdb_wins = sum(1 for r in results if get_winner(r) == 'chDB')
    duckdb_wins = sum(1 for r in results if get_winner(r) == 'DuckDB')

    print(
        f"\nOverall wins: Pandas={pandas_wins}/{len(results)}, chDB={chdb_wins}/{len(results)}, DuckDB={duckdb_wins}/{len(results)}"
    )

    # Group by size
    for size in sorted(set(r.data_size for r in results)):
        size_results = [r for r in results if r.data_size == size]
        pandas_better = sum(1 for r in size_results if get_winner(r) == 'Pandas')
        chdb_better = sum(1 for r in size_results if get_winner(r) == 'chDB')
        duckdb_better = sum(1 for r in size_results if get_winner(r) == 'DuckDB')
        print(f"\n  {size:>10,} rows: Pandas={pandas_better}, chDB={chdb_better}, DuckDB={duckdb_better}")

    # Specific recommendations
    print("\nPer-operation analysis:")
    for op in sorted(set(r.operation for r in results), key=lambda x: [r.operation for r in results].index(x)):
        op_results = [r for r in results if r.operation == op]
        winners = [get_winner(r) for r in op_results]
        win_counts = Counter(winners)
        most_common = win_counts.most_common(1)[0]

        if most_common[1] > len(op_results) / 2:
            rec = f"Use {most_common[0]}"
        else:
            rec = "Depends on data size"

        print(f"  {op:<20}: {rec}")

    # Generate plot (saves to current working directory)
    plot_benchmark_results(results, output_prefix='benchmark')


def plot_benchmark_results(results: List[BenchmarkResult], output_prefix: str = 'benchmark'):
    """Generate benchmark visualization plot."""
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams['figure.titlesize'] = 12

    # Convert results to DataFrame
    data = []
    for r in results:
        times = {'Pandas': r.pandas_time, 'chDB': r.chdb_time, 'DuckDB': r.duckdb_time}
        winner = min(times, key=times.get)
        data.append(
            {
                'op': r.operation,
                'size': r.data_size,
                'pandas': r.pandas_time,
                'chdb': r.chdb_time,
                'duckdb': r.duckdb_time,
                'winner': winner,
            }
        )

    df = pd.DataFrame(data)

    # Define colors
    colors = {'chDB': '#5B8FF9', 'DuckDB': '#F4E04D', 'Pandas': '#5AD8A6'}  # Blue  # Yellow  # Teal/Cyan

    # Get unique sizes and operations
    sizes = sorted(df['size'].unique())
    size_labels = [f'{s//1000}K' if s < 1000000 else f'{s//1000000}M' for s in sizes]
    all_ops = list(df['op'].unique())

    # Calculate wins for each size
    wins_by_size = {}
    for size in sizes:
        df_size = df[df['size'] == size]
        wins_by_size[size] = {
            'Pandas': len(df_size[df_size['winner'] == 'Pandas']),
            'chDB': len(df_size[df_size['winner'] == 'chDB']),
            'DuckDB': len(df_size[df_size['winner'] == 'DuckDB']),
        }

    # Reorder operations: chDB-strong operations first
    chdb_strong_ops = [
        'GroupBy count',
        'Multi-filter (4x)',
        'Filter+Select+Sort',
        'Complex pipeline',
        'Chain 5 filters',
        'Filter (multiple)',
        'Sort (single)',
        'GroupBy agg',
    ]
    all_ops_ordered = [op for op in chdb_strong_ops if op in all_ops] + [
        op for op in all_ops if op not in chdb_strong_ops
    ]

    n_ops = len(all_ops_ordered)
    n_sizes = len(sizes)

    # Group spacing parameters
    width = 0.22
    gap_between_sizes = 0.15
    gap_between_ops = 0.8
    size_group_width = 3 * width
    total_size_group_width = n_sizes * size_group_width + (n_sizes - 1) * gap_between_sizes

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 5.8))

    # Calculate positions
    x_positions = np.arange(n_ops) * (total_size_group_width + gap_between_ops)

    # Store positions for labels
    all_positions = []
    all_labels = []

    # Plot bars for all operations
    for op_idx, op in enumerate(all_ops_ordered):
        df_op = df[df['op'] == op]

        for size_idx, (size, size_label) in enumerate(zip(sizes, size_labels)):
            df_size = df_op[df_op['size'] == size]

            if len(df_size) == 0:
                continue

            row = df_size.iloc[0]

            # Calculate x position
            base_x = x_positions[op_idx] + size_idx * (size_group_width + gap_between_sizes)

            # Plot bars - Pandas, DuckDB, chDB order
            pandas_bar = ax.bar(
                base_x, row['pandas'], width, color=colors['Pandas'], alpha=0.75, edgecolor='black', linewidth=0.5
            )
            duckdb_bar = ax.bar(
                base_x + width,
                row['duckdb'],
                width,
                color=colors['DuckDB'],
                alpha=0.75,
                edgecolor='black',
                linewidth=0.5,
            )
            chdb_bar = ax.bar(
                base_x + 2 * width,
                row['chdb'],
                width,
                color=colors['chDB'],
                alpha=0.75,
                edgecolor='black',
                linewidth=0.5,
            )

            # Highlight winner
            if row['winner'] == 'Pandas':
                pandas_bar[0].set_alpha(1.0)
                pandas_bar[0].set_linewidth(2.0)
            elif row['winner'] == 'chDB':
                chdb_bar[0].set_alpha(1.0)
                chdb_bar[0].set_linewidth(2.0)
            elif row['winner'] == 'DuckDB':
                duckdb_bar[0].set_alpha(1.0)
                duckdb_bar[0].set_linewidth(2.0)

            # Store position
            center_x = base_x + width * 1.5
            all_positions.append(center_x)
            all_labels.append(size_label)

    # Formatting
    ax.set_ylabel('Execution Time (ms)', fontweight='bold', fontsize=11)
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, linestyle='--', which='both')

    # Build table data
    table_data = [['Wins'] + size_labels]
    for engine in ['chDB', 'Pandas', 'DuckDB']:
        row = [engine] + [str(wins_by_size[size][engine]) for size in sizes]
        table_data.append(row)

    # Table colors
    table_colors = [
        ['white'] * (n_sizes + 1),
        [colors['chDB']] + ['white'] * n_sizes,
        [colors['Pandas']] + ['white'] * n_sizes,
        [colors['DuckDB']] + ['white'] * n_sizes,
    ]

    # Add table
    table = ax.table(
        cellText=table_data, cellLoc='center', loc='upper left', bbox=[0.02, 0.72, 0.12, 0.25], cellColours=table_colors
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for i in range(n_sizes + 1):
        cell = table[(0, i)]
        cell.set_text_props(weight='bold', fontsize=8)
        cell.set_facecolor('#E8E8E8')

    # Style color column
    for i in range(1, 4):
        cell = table[(i, 0)]
        cell.set_text_props(weight='bold', fontsize=8, color='black')
        cell.set_alpha(0.8)

    # Table borders
    for key, cell in table.get_celld().items():
        cell.set_linewidth(1.0)
        cell.set_edgecolor('black')

    # Title
    ax.set_title(
        'DataFrame Performance Benchmark: Pandas vs chDB vs DuckDB (All Operations)',
        fontweight='bold',
        fontsize=12,
        pad=15,
    )

    # Two-level x-axis labels
    ax.set_xticks(all_positions)
    ax.set_xticklabels(all_labels, fontsize=7, color='gray')
    ax.tick_params(axis='x', which='major', length=3)

    # Operation names on secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    op_centers = x_positions + total_size_group_width / 2
    ax2.set_xticks(op_centers)
    ax2.set_xticklabels([op.replace(' ', '\n') for op in all_ops_ordered], fontsize=7.5, fontweight='bold')
    ax2.tick_params(axis='x', which='major', length=0)
    ax2.spines['top'].set_visible(False)

    ax.spines['bottom'].set_position(('outward', 10))

    plt.tight_layout()

    # Save figures
    pdf_path = f'{output_prefix}.pdf'
    png_path = f'{output_prefix}.png'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {pdf_path} and {png_path}")

    # Print wins summary
    print("\nWins Summary by Data Size:")
    print("=" * 50)
    for size, label in zip(sizes, size_labels):
        wins = wins_by_size[size]
        print(f"{label:>5}: chDB={wins['chDB']}, Pandas={wins['Pandas']}, DuckDB={wins['DuckDB']}")

    plt.show()


if __name__ == '__main__':
    main()
