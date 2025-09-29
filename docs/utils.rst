Utilities
=========

chDB provides various utility functions and classes for data processing, type conversion, performance monitoring, and debugging.

Overview
--------

The utils module includes:

- **Type Conversion**: Convert between Python and ClickHouse data types
- **Data Processing**: Flatten dictionaries, columnar conversion, type inference
- **Tracing and Debugging**: Performance monitoring and query tracing
- **Helper Functions**: Various utility functions for data manipulation

API Reference
-------------

For the complete utilities API reference, see :doc:`api`.

Data Processing Utilities
-------------------------

**Dictionary Flattening**

.. code-block:: python

   from chdb.utils import flatten_dict
   
   # Flatten nested dictionaries
   nested_data = {
       'user': {
           'profile': {
               'name': 'Alice',
               'age': 30
           },
           'preferences': {
               'theme': 'dark',
               'language': 'en'
           }
       },
       'account': {
           'status': 'active'
       }
   }
   
   flattened = flatten_dict(nested_data)
   print(flattened)
   # Output: {
   #     'user.profile.name': 'Alice',
   #     'user.profile.age': 30,
   #     'user.preferences.theme': 'dark',
   #     'user.preferences.language': 'en',
   #     'account.status': 'active'
   # }

**Columnar Data Conversion**

.. code-block:: python

   from chdb.utils import convert_to_columnar
   
   # Convert row-based data to columnar format
   row_data = [
       {'name': 'Alice', 'age': 30, 'city': 'NYC'},
       {'name': 'Bob', 'age': 25, 'city': 'LA'},  
       {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
   ]
   
   columnar_data = convert_to_columnar(row_data)
   print(columnar_data)
   # Output: {
   #     'name': ['Alice', 'Bob', 'Charlie'],
   #     'age': [30, 25, 35], 
   #     'city': ['NYC', 'LA', 'Chicago']
   # }

**Data Type Inference**

.. code-block:: python

   from chdb.utils import infer_data_type, infer_data_types
   
   # Infer single data type
   data_type = infer_data_type([1, 2, 3, 4, 5])
   print(data_type)  # 'UInt64'
   
   data_type = infer_data_type(['hello', 'world', 'test'])
   print(data_type)  # 'String'
   
   data_type = infer_data_type([1.5, 2.7, 3.14])
   print(data_type)  # 'Float64'
   
   # Infer types for multiple columns
   data = {
       'id': [1, 2, 3],
       'name': ['Alice', 'Bob', 'Charlie'],
       'score': [85.5, 92.0, 88.5],
       'active': [True, False, True]
   }
   
   types = infer_data_types(data)
   print(types)
   # Output: {
   #     'id': 'UInt64',
   #     'name': 'String', 
   #     'score': 'Float64',
   #     'active': 'Bool'
   # }

Type Conversion
---------------

**Working with ClickHouse Types**

.. code-block:: python

   import chdb
   from chdb.utils import types
   
   # Convert Python types to ClickHouse types
   ch_type = types.python_to_clickhouse(int)
   print(ch_type)  # Int64
   
   ch_type = types.python_to_clickhouse(str)
   print(ch_type)  # String
   
   ch_type = types.python_to_clickhouse(float)
   print(ch_type)  # Float64
   
   # Convert ClickHouse types to Python types
   py_type = types.clickhouse_to_python("String")
   print(py_type)  # str
   
   py_type = types.clickhouse_to_python("UInt32")
   print(py_type)  # int
   
   py_type = types.clickhouse_to_python("Float64")
   print(py_type)  # float

**Advanced Type Mapping**

.. code-block:: python

   # Complex type mappings
   mappings = [
       ('Array(String)', list),
       ('Tuple(String, UInt64)', tuple),
       ('Nullable(String)', str),
       ('DateTime', 'datetime'),
       ('Date', 'date'),
       ('UUID', str),
       ('Decimal(10,2)', 'decimal')
   ]
   
   for clickhouse_type, expected in mappings:
       python_type = types.clickhouse_to_python(clickhouse_type)
       print(f"{clickhouse_type} -> {python_type}")

**Type Validation and Conversion**

.. code-block:: python

   def validate_and_convert_data(data, schema):
       """Validate data against schema and convert types"""
       from chdb.utils import infer_data_type
       
       converted_data = {}
       
       for column, values in data.items():
           if column in schema:
               expected_type = schema[column]
               inferred_type = infer_data_type(values)
               
               if inferred_type != expected_type:
                   print(f"Warning: {column} expected {expected_type}, got {inferred_type}")
               
               converted_data[column] = values
           else:
               print(f"Warning: Unknown column {column}")
       
       return converted_data
   
   # Example usage
   data = {
       'id': [1, 2, 3],
       'name': ['A', 'B', 'C'],
       'score': [1.1, 2.2, 3.3]
   }
   
   schema = {
       'id': 'UInt64',
       'name': 'String',
       'score': 'Float64'
   }
   
   validated_data = validate_and_convert_data(data, schema)

Tracing and Debugging
---------------------

**Query Tracing**

.. code-block:: python

   from chdb.utils import trace
   import chdb
   
   # Enable query tracing
   trace.enable_trace()
   
   # Run query with tracing
   result = chdb.query("SELECT count() FROM numbers(100000)")
   
   # View trace information
   trace_info = trace.get_trace()
   print(trace_info)
   
   # Disable tracing
   trace.disable_trace()

**Advanced Tracing with Context Manager**

.. code-block:: python

   from chdb.utils import trace
   import chdb
   
   class QueryTracer:
       def __enter__(self):
           trace.enable_trace()
           return self
       
       def __exit__(self, exc_type, exc_val, exc_tb):
           trace.disable_trace()
       
       def get_trace(self):
           return trace.get_trace()
   
   # Usage
   with QueryTracer() as tracer:
       result1 = chdb.query("SELECT count() FROM numbers(10000)")
       result2 = chdb.query("SELECT avg(number) FROM numbers(10000)")
       
       trace_info = tracer.get_trace()
       print(f"Traced operations: {trace_info}")

Performance Monitoring
----------------------

**Built-in Query Metrics**

.. code-block:: python

   import chdb
   
   # Execute query and get detailed metrics
   result = chdb.query("SELECT count() FROM numbers(1000000)")
   
   # Access performance metrics
   print(f"Rows read: {result.rows_read():,}")
   print(f"Bytes read: {result.bytes_read():,}")
   print(f"Storage rows read: {result.storage_rows_read():,}")
   print(f"Storage bytes read: {result.storage_bytes_read():,}")
   print(f"Query execution time: {result.elapsed():.4f} seconds")

**Custom Performance Monitor**

.. code-block:: python

   import chdb
   import time
   from contextlib import contextmanager
   
   @contextmanager
   def performance_monitor(query_name="Query"):
       """Context manager for monitoring query performance"""
       print(f"Starting {query_name}...")
       start_time = time.perf_counter()
       start_memory = get_memory_usage()
       
       try:
           yield
       finally:
           end_time = time.perf_counter()
           end_memory = get_memory_usage()
           
           print(f"{query_name} completed:")
           print(f"  Execution time: {end_time - start_time:.4f} seconds")
           print(f"  Memory delta: {end_memory - start_memory:.2f} MB")
   
   def get_memory_usage():
       """Get current memory usage in MB"""
       try:
           import psutil
           import os
           process = psutil.Process(os.getpid())
           return process.memory_info().rss / 1024 / 1024
       except ImportError:
           return 0.0
   
   # Usage
   with performance_monitor("Large aggregation"):
       result = chdb.query("""
           SELECT 
               number % 1000 as bucket,
               count(*) as count,
               avg(number) as avg_value
           FROM numbers(10000000)
           GROUP BY bucket
           ORDER BY bucket
       """)

**Benchmark Utilities**

.. code-block:: python

   import chdb
   import time
   from statistics import mean, median
   
   def benchmark_query(query, iterations=5, warmup=1):
       """Benchmark a query with multiple iterations"""
       
       # Warmup runs
       for _ in range(warmup):
           chdb.query(query)
       
       # Actual benchmark runs
       times = []
       for i in range(iterations):
           start_time = time.perf_counter()
           result = chdb.query(query)
           end_time = time.perf_counter()
           
           execution_time = end_time - start_time
           times.append(execution_time)
           
           print(f"Run {i+1}: {execution_time:.4f}s "
                 f"({result.rows_read():,} rows, {result.bytes_read():,} bytes)")
       
       print(f"\nBenchmark Results:")
       print(f"  Mean execution time: {mean(times):.4f}s")
       print(f"  Median execution time: {median(times):.4f}s")
       print(f"  Min execution time: {min(times):.4f}s") 
       print(f"  Max execution time: {max(times):.4f}s")
       
       return {
           'times': times,
           'mean': mean(times),
           'median': median(times),
           'min': min(times),
           'max': max(times)
       }
   
   # Usage
   benchmark_results = benchmark_query(
       "SELECT count() FROM numbers(1000000) WHERE number % 2 = 0",
       iterations=3
   )

Memory Usage Monitoring
-----------------------

**Memory Profiling**

.. code-block:: python

   import chdb
   import gc
   
   def memory_profile_query(query, description="Query"):
       """Profile memory usage of a query"""
       try:
           import psutil
           import os
           
           process = psutil.Process(os.getpid())
           
           # Force garbage collection
           gc.collect()
           
           # Get initial memory usage
           memory_before = process.memory_info().rss / 1024 / 1024
           
           # Execute query
           print(f"Executing {description}...")
           result = chdb.query(query)
           
           # Get final memory usage
           memory_after = process.memory_info().rss / 1024 / 1024
           memory_delta = memory_after - memory_before
           
           print(f"Memory Profile for {description}:")
           print(f"  Initial memory: {memory_before:.2f} MB")
           print(f"  Final memory: {memory_after:.2f} MB")
           print(f"  Memory delta: {memory_delta:+.2f} MB")
           print(f"  Rows processed: {result.rows_read():,}")
           print(f"  Bytes processed: {result.bytes_read():,}")
           print(f"  Memory per row: {(memory_delta * 1024 * 1024) / max(result.rows_read(), 1):.2f} bytes")
           
           return {
               'memory_before': memory_before,
               'memory_after': memory_after,
               'memory_delta': memory_delta,
               'rows_read': result.rows_read(),
               'bytes_read': result.bytes_read()
           }
           
       except ImportError:
           print("psutil not available. Install with: pip install psutil")
           return None
   
   # Usage
   memory_stats = memory_profile_query(
       "SELECT * FROM numbers(1000000) WHERE number % 1000 = 0",
       "Filtering large dataset"
   )

Data Processing Helpers
-----------------------

**Batch Processing Utilities**

.. code-block:: python

   def process_data_in_batches(data, batch_size=1000, processor_func=None):
       """Process large datasets in batches"""
       import chdb
       
       if processor_func is None:
           processor_func = lambda batch: chdb.query(f"SELECT * FROM Python(batch)")
       
       results = []
       
       for i in range(0, len(data), batch_size):
           batch = data[i:i + batch_size]
           print(f"Processing batch {i//batch_size + 1} "
                 f"({len(batch)} items)")
           
           batch_result = processor_func(batch)
           results.append(batch_result)
       
       return results
   
   # Example usage with pandas DataFrame
   def process_large_dataframe():
       import pandas as pd
       import chdb
       
       # Create large dataset
       large_df = pd.DataFrame({
           'id': range(10000),
           'value': [i * 2 for i in range(10000)],
           'category': [f'cat_{i % 10}' for i in range(10000)]
       })
       
       def analyze_batch(batch_df):
           return chdb.query("""
               SELECT 
                   category,
                   count(*) as count,
                   avg(value) as avg_value
               FROM Python(batch_df)
               GROUP BY category
           """)
       
       # Split DataFrame into batches
       batch_size = 2000
       df_batches = [
           large_df[i:i+batch_size] 
           for i in range(0, len(large_df), batch_size)
       ]
       
       # Process batches
       batch_results = process_data_in_batches(
           df_batches, 
           processor_func=analyze_batch
       )
       
       return batch_results

Utility Functions
-----------------

**Data Format Conversion**

.. code-block:: python

   def convert_result_format(result, output_format='dict'):
       """Convert chDB result to different formats"""
       
       if output_format == 'dict':
           # Convert to list of dictionaries
           lines = result.strip().split('\n')[1:]  # Skip header
           header = result.strip().split('\n')[0].split('\t')
           
           dict_result = []
           for line in lines:
               values = line.split('\t')
               row_dict = dict(zip(header, values))
               dict_result.append(row_dict)
           
           return dict_result
           
       elif output_format == 'pandas':
           # Convert to pandas DataFrame
           try:
               import pandas as pd
               from io import StringIO
               
               return pd.read_csv(StringIO(result), sep='\t')
           except ImportError:
               print("pandas not available. Install with: pip install pandas")
               return None
               
       else:
           return result
   
   # Usage
   result = chdb.query("SELECT number, number*2 as double FROM numbers(5)", "TSV")
   
   dict_data = convert_result_format(result, 'dict')
   print("Dictionary format:", dict_data)
   
   df_data = convert_result_format(result, 'pandas')
   print("DataFrame format:")
   print(df_data)

See Also
--------

- :doc:`api` - Complete API reference including utilities module
- :doc:`troubleshooting` - Troubleshooting and performance tips
- :doc:`examples` - More utility examples and use cases