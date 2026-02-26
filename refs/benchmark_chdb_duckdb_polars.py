#!/usr/bin/env python3
"""
Benchmark: chDB vs DuckDB vs Polars

This benchmark compares the performance of three SQL-capable DataFrame engines:
1. chDB - ClickHouse embedded
2. DuckDB - Analytical SQL database
3. Polars - Fast DataFrame library with SQL support

Operations tested:
- Simple filter (WHERE)
- Aggregation (GROUP BY)
- Complex filter + sort + limit
"""

import os
import json
import subprocess
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import duckdb
import polars as pl
import chdb
import time
import matplotlib.pyplot as plt


def get_git_info() -> Dict[str, str]:
    """Get current git branch, commit hash, and other info."""
    git_info = {
        'branch': 'unknown',
        'commit_hash': 'unknown',
        'commit_short': 'unknown',
        'commit_message': 'unknown',
        'is_dirty': False,
    }
    try:
        # Get branch name
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['branch'] = result.stdout.strip()

        # Get full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['commit_hash'] = result.stdout.strip()
            git_info['commit_short'] = result.stdout.strip()[:7]

        # Get commit message (first line)
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%s'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['commit_message'] = result.stdout.strip()[:80]

        # Check if working directory is dirty
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info['is_dirty'] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not get git info: {e}")

    return git_info


def create_benchmark_output_dir(benchmark_name: str, base_dir: str = None) -> tuple:
    """
    Create a timestamped output directory for benchmark results.

    Args:
        benchmark_name: Name of the benchmark (e.g., 'chdb_duckdb_polars')
        base_dir: Base directory for all benchmark results

    Returns:
        tuple: (output_dir_path, metadata_dict)
    """
    if base_dir is None:
        # Default to refs/benchmark_results/ relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, 'benchmark_results', benchmark_name)

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Get timestamp and git info
    timestamp = datetime.now()
    git_info = get_git_info()

    # Create directory name: YYYYMMDD_HHMMSS_branch_commit
    branch_safe = git_info['branch'].replace('/', '_').replace('\\', '_')[:20]
    dirty_marker = '_dirty' if git_info['is_dirty'] else ''
    dir_name = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{branch_safe}_{git_info['commit_short']}{dirty_marker}"

    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare metadata
    metadata = {
        'timestamp': timestamp.isoformat(),
        'timestamp_unix': timestamp.timestamp(),
        'git_branch': git_info['branch'],
        'git_commit_hash': git_info['commit_hash'],
        'git_commit_short': git_info['commit_short'],
        'git_commit_message': git_info['commit_message'],
        'git_is_dirty': git_info['is_dirty'],
        'output_dir': output_dir,
    }

    return output_dir, metadata


@dataclass
class BenchmarkResult:
    operation: str
    data_size: int
    duckdb_time: float
    polars_time: float
    chdb_time: float

    @property
    def fastest(self) -> str:
        times = {'DuckDB': self.duckdb_time, 'Polars': self.polars_time, 'chDB': self.chdb_time}
        winner = min(times, key=times.get)
        winner_time = times[winner]
        loser_time = max(times.values())
        speedup = loser_time / winner_time if winner_time > 0 else float('inf')
        return f"{winner} ({speedup:.2f}x)"


def run_benchmarks(sizes: List[int], n_runs: int = 3) -> List[BenchmarkResult]:
    """Run all benchmarks for different data sizes."""
    results = []

    duck_conn = duckdb.connect()
    chdb_conn = chdb.connect()

    operations = [
        ('Simple Filter', 
         'SELECT a, c FROM Python(df) WHERE a > 500',
         'SELECT a, c FROM df WHERE a > 500'),
        ('Aggregation',
         'SELECT b, SUM(a) as sum_a, AVG(c) as avg_c FROM Python(df) GROUP BY b',
         'SELECT b, SUM(a) as sum_a, AVG(c) as avg_c FROM df GROUP BY b'),
        ('Complex Filter+Sort',
         'SELECT a, b, c FROM Python(df) WHERE a > 500 AND d < 50 ORDER BY c DESC LIMIT 1000',
         'SELECT a, b, c FROM df WHERE a > 500 AND d < 50 ORDER BY c DESC LIMIT 1000'),
    ]

    for op_name, chdb_query, duck_query in operations:
        print(f'\nBenchmark: {op_name}')
        print('=' * 70)
        print(f'{"Rows":>15} {"DuckDB":>12} {"Polars":>12} {"chDB":>12}')
        print('-' * 70)

        for N in sizes:
            np.random.seed(42)
            df = pd.DataFrame({
                'a': np.random.randint(1, 1000, N),
                'b': np.random.randint(1, 100, N),
                'c': np.random.randn(N),
                'd': np.random.randint(1, 100, N),
            })

            # Warmup
            duck_conn.sql(duck_query).df()
            pl.sql(duck_query, eager=True)
            chdb_conn.query(chdb_query, 'DataFrame')

            # Run multiple times and take average
            duck_times, pl_times, chdb_times = [], [], []

            for _ in range(n_runs):
                # DuckDB
                start = time.perf_counter()
                duck_res = duck_conn.sql(duck_query).df()
                duck_times.append((time.perf_counter() - start) * 1000)

                # Polars
                start = time.perf_counter()
                pl_res = pl.sql(duck_query, eager=True).to_pandas()
                pl_times.append((time.perf_counter() - start) * 1000)

                # chDB
                start = time.perf_counter()
                chdb_res = chdb_conn.query(chdb_query, 'DataFrame')
                chdb_times.append((time.perf_counter() - start) * 1000)

            # Average times
            duck_time = sum(duck_times) / len(duck_times)
            pl_time = sum(pl_times) / len(pl_times)
            chdb_time = sum(chdb_times) / len(chdb_times)

            result = BenchmarkResult(
                operation=op_name,
                data_size=N,
                duckdb_time=duck_time,
                polars_time=pl_time,
                chdb_time=chdb_time,
            )
            results.append(result)

            print(f'{N:>15,} {duck_time:>10.2f}ms {pl_time:>10.2f}ms {chdb_time:>10.2f}ms  -> {result.fastest}')

    return results


def save_benchmark_results(
    results: List[BenchmarkResult], output_dir: str, metadata: Dict, data_sizes: List[int], n_runs: int
):
    """Save benchmark results to CSV and metadata to JSON."""
    # Convert results to serializable format
    results_data = []
    for r in results:
        results_data.append({
            'operation': r.operation,
            'data_size': r.data_size,
            'duckdb_time_ms': r.duckdb_time,
            'polars_time_ms': r.polars_time,
            'chdb_time_ms': r.chdb_time,
            'fastest': r.fastest,
        })

    # Save CSV (flat format for easy analysis)
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df = pd.DataFrame(results_data)
    df['timestamp'] = metadata['timestamp']
    df['git_branch'] = metadata['git_branch']
    df['git_commit'] = metadata['git_commit_short']
    df.to_csv(csv_path, index=False)
    print(f"Results saved: {csv_path}")

    # Save metadata separately
    meta_path = os.path.join(output_dir, 'metadata.json')
    full_metadata = {
        **metadata,
        'config': {
            'data_sizes': data_sizes,
            'n_runs': n_runs,
        },
    }
    with open(meta_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    return csv_path


def plot_benchmark_results(results: List[BenchmarkResult], output_dir: str = None, output_prefix: str = 'benchmark_chdb_duckdb_polars'):
    """Generate benchmark visualization plot."""
    # Set style
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 9
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 11

    # Convert results to DataFrame
    data = []
    for r in results:
        times = {'DuckDB': r.duckdb_time, 'Polars': r.polars_time, 'chDB': r.chdb_time}
        winner = min(times, key=times.get)
        data.append({
            'op': r.operation,
            'size': r.data_size,
            'DuckDB': r.duckdb_time,
            'Polars': r.polars_time,
            'chDB': r.chdb_time,
            'winner': winner,
        })

    df = pd.DataFrame(data)

    # Define colors
    colors = {'DuckDB': '#FFC107', 'Polars': '#2196F3', 'chDB': '#4CAF50'}

    # Get unique sizes and operations
    sizes = sorted(df['size'].unique())
    size_labels = [f'{s//1000}K' if s < 1000000 else f'{s//1000000}M' for s in sizes]
    all_ops = list(df['op'].unique())

    n_ops = len(all_ops)
    n_sizes = len(sizes)
    n_engines = 3

    # Create figure with subplots for each operation
    fig, axes = plt.subplots(1, n_ops, figsize=(4 * n_ops, 5), sharey=False)
    if n_ops == 1:
        axes = [axes]

    for ax, op in zip(axes, all_ops):
        df_op = df[df['op'] == op]
        
        x = np.arange(n_sizes)
        width = 0.25

        for i, (engine, color) in enumerate(colors.items()):
            values = [df_op[df_op['size'] == s][engine].values[0] for s in sizes]
            bars = ax.bar(x + i * width, values, width, label=engine, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Data Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title(op)
        ax.set_xticks(x + width)
        ax.set_xticklabels(size_labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.suptitle('SQL Engine Benchmark: chDB vs DuckDB vs Polars', fontweight='bold', fontsize=12)
    plt.tight_layout()

    # Determine output path
    if output_dir:
        pdf_path = os.path.join(output_dir, f'{output_prefix}.pdf')
    else:
        pdf_path = f'{output_prefix}.pdf'

    # Save figure (PDF only)
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved: {pdf_path}")

    plt.show()


def print_summary(results: List[BenchmarkResult]):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Count wins
    wins = {'DuckDB': 0, 'Polars': 0, 'chDB': 0}
    for r in results:
        times = {'DuckDB': r.duckdb_time, 'Polars': r.polars_time, 'chDB': r.chdb_time}
        winner = min(times, key=times.get)
        wins[winner] += 1

    print(f"\nOverall wins: DuckDB={wins['DuckDB']}, Polars={wins['Polars']}, chDB={wins['chDB']}")

    # Group by operation
    ops = sorted(set(r.operation for r in results))
    for op in ops:
        op_results = [r for r in results if r.operation == op]
        op_wins = {'DuckDB': 0, 'Polars': 0, 'chDB': 0}
        for r in op_results:
            times = {'DuckDB': r.duckdb_time, 'Polars': r.polars_time, 'chDB': r.chdb_time}
            winner = min(times, key=times.get)
            op_wins[winner] += 1
        print(f"  {op}: DuckDB={op_wins['DuckDB']}, Polars={op_wins['Polars']}, chDB={op_wins['chDB']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark chDB vs DuckDB vs Polars')
    parser.add_argument(
        '--sizes', type=str, default='1000000,5000000,10000000,20000000',
        help='Comma-separated data sizes (default: 1000000,5000000,10000000,20000000)'
    )
    parser.add_argument('--runs', type=int, default=3, help='Number of runs per operation')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true', help='Skip saving results to file')
    parser.add_argument('--output-dir', type=str, help='Custom output directory for results')
    args = parser.parse_args()

    print("=" * 70)
    print("chDB vs DuckDB vs Polars Benchmark")
    print("=" * 70)

    # Create output directory for results
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        git_info = get_git_info()
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'timestamp_unix': datetime.now().timestamp(),
            'git_branch': git_info['branch'],
            'git_commit_hash': git_info['commit_hash'],
            'git_commit_short': git_info['commit_short'],
            'git_commit_message': git_info['commit_message'],
            'git_is_dirty': git_info['is_dirty'],
            'output_dir': output_dir,
        }
    else:
        output_dir, metadata = create_benchmark_output_dir('chdb_duckdb_polars')

    print(f"Output directory: {output_dir}")
    print(f"Git: {metadata.get('git_branch', 'unknown')}@{metadata.get('git_commit_short', 'unknown')}")
    if metadata.get('git_is_dirty'):
        print("Warning: Working directory has uncommitted changes")

    # Parse data sizes
    data_sizes = [int(s.strip()) for s in args.sizes.split(',')]
    print(f"Data sizes: {data_sizes}")
    print(f"Runs per operation: {args.runs}")

    # Run benchmarks
    results = run_benchmarks(data_sizes, n_runs=args.runs)

    # Print summary
    print_summary(results)

    # Save results to output directory
    if not args.no_save:
        save_benchmark_results(results, output_dir, metadata, data_sizes, args.runs)

    # Generate plot (unless --no-plot)
    if not args.no_plot:
        plot_benchmark_results(results, output_dir=output_dir)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
