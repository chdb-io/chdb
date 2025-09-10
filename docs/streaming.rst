Streaming Queries
=================

chDB supports streaming queries that allow you to process large datasets with constant memory usage through chunked streaming. This is particularly useful when working with datasets that are too large to fit in memory or when you need to process results as they become available.

Overview
--------

Streaming queries provide several key benefits:

- **Memory Efficiency**: Process large result sets without loading all data into memory
- **Progressive Processing**: Start processing results as soon as the first chunk arrives
- **Early Termination**: Stop processing mid-stream to save resources
- **Integration**: Works seamlessly with external libraries like PyArrow and Delta Lake

The streaming functionality is built around the :class:`StreamingResult` class, which acts as an iterator over result chunks.

Basic Streaming Examples
------------------------

**Context Manager Approach (Recommended)**

.. code-block:: python

   from chdb import session as chs
   
   sess = chs.Session()
   
   # Basic streaming with automatic resource cleanup
   rows_cnt = 0
   with sess.send_query("SELECT * FROM numbers(200000)", "CSV") as stream_result:
       for chunk in stream_result:
           rows_cnt += chunk.rows_read()
   
   print(f"Processed {rows_cnt} rows")  # 200000

**Manual Iteration with fetch()**

.. code-block:: python

   # Manual control over streaming
   rows_cnt = 0
   stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
   
   while True:
       chunk = stream_result.fetch()
       if chunk is None:
           break
       rows_cnt += chunk.rows_read()
   
   print(f"Processed {rows_cnt} rows")  # 200000
   
   # Important: Don't forget to close when done
   stream_result.close()

**Early Termination**

.. code-block:: python

   # Process only part of the result set
   rows_cnt = 0
   stream_result = sess.send_query("SELECT * FROM numbers(200000)", "CSV")
   
   while True:
       chunk = stream_result.fetch()
       if chunk is None:
           break
       
       rows_cnt += chunk.rows_read()
       
       # Stop after processing first chunk
       if rows_cnt > 0:
           stream_result.close()  # Must call close() for early termination
           break
   
   print(f"Early termination after {rows_cnt} rows")  # ~65409

Advanced Streaming Operations
-----------------------------

**Processing Large Tables**

.. code-block:: python

   sess = chs.Session("analytics.db")
   
   # Create and populate a large table
   sess.query("""
       CREATE TABLE large_dataset (
           id UInt64,
           timestamp DateTime,
           value Float64,
           category String
       ) ENGINE = MergeTree() ORDER BY id
   """)
   
   sess.query("""
       INSERT INTO large_dataset 
       SELECT 
           number as id,
           now() - interval number second as timestamp,
           randCanonical() * 100 as value,
           ['A', 'B', 'C'][number % 3 + 1] as category
       FROM numbers(10000000)
   """)
   
   # Stream process the large table
   category_sums = {'A': 0, 'B': 0, 'C': 0}
   processed_rows = 0
   
   with sess.send_query("""
       SELECT category, value 
       FROM large_dataset 
       ORDER BY timestamp DESC
   """, "CSV") as stream_result:
       
       for chunk in stream_result:
           # Process chunk data
           lines = chunk.data().strip().split('\n')
           for line in lines:
               if line:
                   category, value = line.split(',')
                   category_sums[category] += float(value)
                   processed_rows += 1
           
           # Optional: Progress reporting
           if processed_rows % 100000 == 0:
               print(f"Processed {processed_rows} rows...")
   
   print("Category sums:", category_sums)
   print(f"Total processed: {processed_rows}")

**Streaming Aggregations**

.. code-block:: python

   # Stream processing with running aggregates
   sess = chs.Session()
   
   running_sum = 0
   running_count = 0
   max_value = float('-inf')
   min_value = float('inf')
   
   with sess.send_query("""
       SELECT number, number * randCanonical() as random_value
       FROM numbers(1000000)
   """, "CSV") as stream_result:
       
       for chunk in stream_result:
           lines = chunk.data().strip().split('\n')
           for line in lines:
               if line:
                   number, random_value = map(float, line.split(','))
                   running_sum += random_value
                   running_count += 1
                   max_value = max(max_value, random_value)
                   min_value = min(min_value, random_value)
           
           # Calculate running average
           if running_count > 0:
               running_avg = running_sum / running_count
           
           # Optional: Report progress every chunk
           print(f"Running avg: {running_avg:.2f}, min: {min_value:.2f}, max: {max_value:.2f}")

PyArrow Integration
-------------------

**Arrow Format Streaming**

.. code-block:: python

   import pyarrow as pa
   from chdb import session as chs
   
   sess = chs.Session()
   
   # Get streaming result in Arrow format
   stream_result = sess.send_query("SELECT * FROM numbers(100000)", "Arrow")
   
   # Create RecordBatchReader with custom batch size
   batch_reader = stream_result.record_batch(rows_per_batch=10000)
   
   # Process Arrow batches
   total_rows = 0
   for batch in batch_reader:
       print(f"Batch: {batch.num_rows} rows, {batch.num_columns} columns")
       total_rows += batch.num_rows
       
       # Convert to pandas for analysis if needed
       df = batch.to_pandas()
       print(f"  Sum: {df['number'].sum()}")
   
   print(f"Total processed: {total_rows} rows")
   stream_result.close()

**Integration with Delta Lake**

.. code-block:: python

   import pyarrow as pa
   from chdb import session as chs
   
   sess = chs.Session()
   
   # Generate sample data
   sess.query("""
       CREATE TABLE source_data (
           id UInt64,
           name String,
           value Float64,
           timestamp DateTime
       ) ENGINE = Memory
   """)
   
   sess.query("""
       INSERT INTO source_data
       SELECT 
           number as id,
           concat('user_', toString(number)) as name,
           randCanonical() * 1000 as value,
           now() - interval number hour as timestamp
       FROM numbers(50000)
   """)
   
   # Stream to Delta Lake (requires deltalake package)
   stream_result = sess.send_query("SELECT * FROM source_data ORDER BY timestamp", "Arrow")
   batch_reader = stream_result.record_batch(rows_per_batch=5000)
   
   # Note: Uncomment the following lines if you have deltalake installed
   # from deltalake import write_deltalake
   # write_deltalake(
   #     table_or_uri="./my_delta_table",
   #     data=batch_reader,
   #     mode="overwrite"
   # )
   
   # Alternative: Process batches manually
   batch_count = 0
   for batch in batch_reader:
       batch_count += 1
       print(f"Batch {batch_count}: {batch.num_rows} rows")
       # Your custom processing logic here
   
   stream_result.close()

File Processing with Streaming
------------------------------

**Streaming Large CSV Files**

.. code-block:: python

   sess = chs.Session()
   
   # Stream process large CSV file
   with sess.send_query("""
       SELECT 
           column1,
           column2,
           toFloat64OrZero(column3) as numeric_value
       FROM file('large_data.csv', 'CSV')
       WHERE numeric_value > 100
   """, "CSV") as stream_result:
       
       filtered_count = 0
       total_value = 0
       
       for chunk in stream_result:
           lines = chunk.data().strip().split('\n')
           for line in lines:
               if line:
                   parts = line.split(',')
                   if len(parts) >= 3:
                       try:
                           value = float(parts[2])
                           total_value += value
                           filtered_count += 1
                       except ValueError:
                           continue
       
       if filtered_count > 0:
           avg_value = total_value / filtered_count
           print(f"Processed {filtered_count} filtered records")
           print(f"Average value: {avg_value:.2f}")

**Multi-format File Processing**

.. code-block:: python

   # Process different file formats in streaming fashion
   file_configs = [
       ("data.parquet", "Parquet"),
       ("data.json", "JSONEachRow"),
       ("data.csv", "CSV")
   ]
   
   for file_path, format_type in file_configs:
       print(f"Processing {file_path} ({format_type})...")
       
       try:
           with sess.send_query(f"""
               SELECT count(*) as record_count
               FROM file('{file_path}', '{format_type}')
           """, "CSV") as stream_result:
               
               for chunk in stream_result:
                   count = int(chunk.data().strip())
                   print(f"  {file_path}: {count} records")
       
       except Exception as e:
           print(f"  Error processing {file_path}: {e}")

Error Handling and Resource Management
--------------------------------------

**Robust Streaming with Error Handling**

.. code-block:: python

   def safe_streaming_query(session, query, format_type="CSV"):
       """Execute streaming query with comprehensive error handling"""
       stream_result = None
       try:
           stream_result = session.send_query(query, format_type)
           
           for chunk in stream_result:
               try:
                   # Process chunk
                   yield chunk
               except Exception as chunk_error:
                   print(f"Error processing chunk: {chunk_error}")
                   continue
                   
       except Exception as e:
           print(f"Streaming query failed: {e}")
           raise
       finally:
           if stream_result:
               stream_result.close()
   
   # Usage example
   sess = chs.Session()
   
   try:
       for chunk in safe_streaming_query(sess, "SELECT * FROM numbers(100000)"):
           rows_in_chunk = chunk.rows_read()
           print(f"Processing chunk with {rows_in_chunk} rows")
           
           # Your chunk processing logic here
           if rows_in_chunk == 0:
               break
               
   except Exception as e:
       print(f"Stream processing failed: {e}")
   finally:
       sess.close()

**Resource Monitoring**

.. code-block:: python

   import time
   import psutil
   import os
   
   def monitor_streaming_query(session, query):
       """Monitor memory usage during streaming query"""
       process = psutil.Process(os.getpid())
       initial_memory = process.memory_info().rss / 1024 / 1024  # MB
       
       print(f"Initial memory usage: {initial_memory:.2f} MB")
       
       start_time = time.time()
       total_rows = 0
       
       with session.send_query(query, "CSV") as stream_result:
           for i, chunk in enumerate(stream_result):
               total_rows += chunk.rows_read()
               
               # Monitor every 10 chunks
               if i % 10 == 0:
                   current_memory = process.memory_info().rss / 1024 / 1024
                   elapsed = time.time() - start_time
                   
                   print(f"Chunk {i}: {current_memory:.2f} MB "
                         f"(+{current_memory - initial_memory:.2f} MB), "
                         f"{total_rows} rows, {elapsed:.2f}s")
       
       final_memory = process.memory_info().rss / 1024 / 1024
       total_time = time.time() - start_time
       
       print(f"Final: {final_memory:.2f} MB, {total_rows} rows, {total_time:.2f}s")
       print(f"Memory increase: {final_memory - initial_memory:.2f} MB")
   
   # Example usage
   sess = chs.Session()
   monitor_streaming_query(sess, "SELECT * FROM numbers(1000000)")

Performance Optimization
------------------------

**Optimal Chunk Sizes**

.. code-block:: python

   import time
   
   def benchmark_chunk_sizes(session, query):
       """Benchmark different Arrow batch sizes"""
       chunk_sizes = [1000, 5000, 10000, 50000, 100000]
       
       for chunk_size in chunk_sizes:
           start_time = time.time()
           total_rows = 0
           
           stream_result = session.send_query(query, "Arrow")
           batch_reader = stream_result.record_batch(rows_per_batch=chunk_size)
           
           for batch in batch_reader:
               total_rows += batch.num_rows
           
           stream_result.close()
           elapsed = time.time() - start_time
           
           print(f"Chunk size {chunk_size}: {total_rows} rows in {elapsed:.2f}s "
                 f"({total_rows/elapsed:.0f} rows/sec)")
   
   # Example
   sess = chs.Session()
   benchmark_chunk_sizes(sess, "SELECT * FROM numbers(500000)")

**Memory-Efficient Processing Patterns**

.. code-block:: python

   def efficient_aggregation_stream(session, query):
       """Process large datasets with minimal memory footprint"""
       
       # Use generators to avoid storing intermediate results
       def chunk_processor():
           with session.send_query(query, "CSV") as stream_result:
               for chunk in stream_result:
                   lines = chunk.data().strip().split('\n')
                   for line in lines:
                       if line:
                           yield line.split(',')
       
       # Process one record at a time
       count = 0
       running_sum = 0
       
       for record in chunk_processor():
           if len(record) >= 2:
               try:
                   value = float(record[1])
                   running_sum += value
                   count += 1
                   
                   # Yield control periodically
                   if count % 10000 == 0:
                       print(f"Processed {count} records, avg: {running_sum/count:.2f}")
                       
               except ValueError:
                   continue
       
       return count, running_sum / count if count > 0 else 0
   
   # Usage
   sess = chs.Session()
   total_count, average = efficient_aggregation_stream(
       sess, "SELECT number, randCanonical() FROM numbers(1000000)"
   )
   print(f"Final: {total_count} records, average: {average:.2f}")

Best Practices
--------------

**Resource Management**

1. **Always Close Streams**: Use context managers or explicitly call ``close()``
2. **Handle Early Termination**: Call ``close()`` when breaking out of streaming loops
3. **Monitor Memory**: Keep track of memory usage for large datasets
4. **Use Appropriate Formats**: Choose CSV for text processing, Arrow for numerical data

**Performance Tips**

1. **Batch Size Tuning**: Experiment with different batch sizes for optimal performance
2. **Format Selection**: Arrow format is typically faster for large numerical datasets
3. **Query Optimization**: Use WHERE clauses to filter data at the source
4. **Progressive Processing**: Start processing as soon as the first chunk arrives

**Error Handling**

1. **Graceful Degradation**: Handle chunk processing errors without stopping the entire stream
2. **Resource Cleanup**: Ensure streams are closed even when errors occur
3. **Timeout Handling**: Implement timeouts for long-running streaming operations

.. code-block:: python

   # Example of comprehensive best practices
   import time
   from contextlib import contextmanager
   
   @contextmanager
   def streaming_query_with_timeout(session, query, format_type="CSV", timeout=300):
       """Context manager for streaming with timeout and error handling"""
       stream_result = None
       start_time = time.time()
       
       try:
           stream_result = session.send_query(query, format_type)
           
           class TimeoutStreamWrapper:
               def __init__(self, stream, timeout, start_time):
                   self.stream = stream
                   self.timeout = timeout
                   self.start_time = start_time
               
               def __iter__(self):
                   return self
               
               def __next__(self):
                   if time.time() - self.start_time > self.timeout:
                       raise TimeoutError(f"Streaming query exceeded {self.timeout}s timeout")
                   
                   chunk = self.stream.fetch()
                   if chunk is None:
                       raise StopIteration
                   return chunk
               
               def close(self):
                   return self.stream.close()
           
           yield TimeoutStreamWrapper(stream_result, timeout, start_time)
           
       except Exception as e:
           print(f"Streaming error: {e}")
           raise
       finally:
           if stream_result:
               stream_result.close()
   
   # Usage
   sess = chs.Session()
   
   try:
       with streaming_query_with_timeout(sess, "SELECT * FROM numbers(100000)", timeout=60) as stream:
           for chunk in stream:
               # Process chunk with automatic timeout protection
               print(f"Chunk: {chunk.rows_read()} rows")
               
   except TimeoutError as e:
       print(f"Query timed out: {e}")
   except Exception as e:
       print(f"Processing error: {e}")

.. note::
   - Streaming is most beneficial for large result sets (>100MB)
   - Context managers (``with`` statements) are recommended for automatic cleanup
   - Arrow format streaming is typically 2-3x faster than CSV for numerical data
   - Early termination requires explicit ``close()`` calls to prevent resource leaks

.. warning::
   - Always close ``StreamingResult`` objects when terminating early
   - Large batch sizes may increase memory usage
   - Streaming queries may block other queries if not properly closed
   - Network interruptions can cause streaming to fail mid-process

See Also
--------

- :doc:`session` - Session management for streaming queries
- :doc:`examples` - More streaming examples and patterns
- :doc:`api` - Complete API reference
- `PyArrow Documentation <https://arrow.apache.org/docs/python/>`_ - For Arrow integration details