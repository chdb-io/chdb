Examples
========

This section provides comprehensive examples showcasing chDB's capabilities across different use cases.

Basic Queries
-------------

**Simple Calculations**

.. code-block:: python

   import chdb
   
   result = chdb.query("SELECT 1 + 1 as result", "CSV")
   print(result)

   result = chdb.query("""
       SELECT 
           'Hello' || ' ' || 'World' as greeting,
           length('chDB') as name_length,
           upper('clickhouse') as uppercase
   """, "Pretty")
   print(result)

**Date and Time**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           now() as current_timestamp,
           today() as current_date,
           formatDateTime(now(), '%Y-%m-%d %H:%M:%S') as formatted_datetime
   """, "DataFrame")
   print(result)

**Number Generation**

.. code-block:: python

   import chdb
   result = chdb.query("""
       SELECT 
           count() as total_count,
           min(number) as minimum,
           max(number) as maximum,
           avg(number) as average,
           median(number) as median_value,
           stddevPop(number) as std_deviation
       FROM numbers(1000)
   """, "DataFrame")
   print(result)

File Processing Examples
------------------------

**CSV File Analysis**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           column1,
           count(*) as row_count,
           avg(toFloat64OrZero(column2)) as avg_value,
           min(column2) as min_value,
           max(column2) as max_value
       FROM file('sales_data.csv', 'CSV')
       WHERE column2 != ''
       GROUP BY column1
       HAVING row_count > 5
       ORDER BY avg_value DESC
   """, "DataFrame")
   print(result)

**JSON File Processing**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           JSONExtractString(json, 'user.name') as user_name,
           JSONExtractInt(json, 'user.age') as age,
           JSONExtractString(json, 'event.type') as event_type,
           JSONExtract(json, 'metadata.tags', 'Array(String)') as tags
       FROM file('events.json', 'JSONEachRow') as t(json String)
       WHERE JSONExtractInt(json, 'user.age') >= 18
       LIMIT 100
   """)
   print(result)

**Parquet File Querying**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           department,
           job_title,
           count(*) as employee_count,
           avg(salary) as avg_salary,
           percentile(salary, 0.5) as median_salary,
           percentile(salary, 0.95) as salary_95th_percentile
       FROM file('employees.parquet', 'Parquet')
       WHERE hire_date >= '2020-01-01'
       GROUP BY department, job_title
       ORDER BY avg_salary DESC
   """, "DataFrame")
   print(result)

Connection-Based API
---------------------

**Recommended Approach for Database-like Operations**

.. code-block:: python

   import chdb
   
   conn = chdb.connect(":memory:")
   cur = conn.cursor()
   
   cur.execute("SELECT number, toString(number) as str FROM system.numbers LIMIT 3")
   
   print(cur.fetchone())
   print(cur.fetchmany(2))
   
   print(cur.column_names())
   print(cur.column_types())
   cur.execute("SELECT number FROM system.numbers LIMIT 3")
   for row in cur:
       print(row)
   
   cur.close()
   conn.close()

**Memory-Efficient Processing with Batches**

.. code-block:: python

   conn = chdb.connect()
   cur = conn.cursor()
   
   cur.execute("""
       SELECT user_id, action, timestamp, details
       FROM file('large_log_file.csv', 'CSV')
       WHERE timestamp >= '2024-01-01'
       ORDER BY timestamp
   """)
   
   batch_size = 1000
   batch_count = 0
   
   while True:
       rows = cur.fetchmany(batch_size)
       if not rows:
           break
       
       batch_count += 1
       print(f"Processing batch {batch_count}: {len(rows)} rows")
       
       for row in rows:
           user_id, action, timestamp, details = row
   
   cur.close()
   conn.close()

DataFrame Integration
---------------------

**Table Class with __table__ Syntax**

.. code-block:: python

   import chdb.dataframe as cdf
   import pandas as pd
   
   df1 = pd.DataFrame({'a': [1, 2, 3], 'b': ["one", "two", "three"]})
   df2 = pd.DataFrame({'c': [1, 2, 3], 'd': ["1", "2", "3"]})
   
   ret_tbl = cdf.query(sql="select * from __tbl1__ t1 join __tbl2__ t2 on t1.a = t2.c",
                     tbl1=df1, tbl2=df2)
   print("Join Results:")
   print(ret_tbl)
   
   summary = ret_tbl.query('select b, sum(a) from __table__ group by b')
   print("\nAggregation Results:")
   print(summary)
   
   result = chdb.query("SELECT * FROM Python(df1) t1 JOIN Python(df2) t2 ON t1.a = t2.c")
   print("\nDirect Python() engine approach:")
   print(result)

**Working with Table Class Features**

.. code-block:: python

   df = pd.DataFrame({
       'id': [1, 2, 3, 4, 5],
       'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
       'salary': [50000, 60000, 75000, 80000, 95000],
       'department': ['Engineering', 'Marketing', 'Engineering', 'Sales', 'Engineering']
   })
   
   table = cdf.Table(dataframe=df)
   
   result = table.query("SELECT * FROM __table__ WHERE salary > 70000")
   print("High earners:")
   print(result.to_pandas())
   
   summary = table.query("""
       SELECT 
           department,
           COUNT(*) as employee_count,
           AVG(salary) as avg_salary,
           MIN(salary) as min_salary,
           MAX(salary) as max_salary
       FROM __table__ 
       GROUP BY department 
       ORDER BY avg_salary DESC
   """)
   print("\nDepartment Summary:")
   print(summary.to_pandas())
   
   print(f"\nQuery Statistics:")
   print(f"Rows read: {summary.rows_read()}")
   print(f"Bytes read: {summary.bytes_read()}")
   print(f"Elapsed time: {summary.elapsed():.4f} seconds")

**Advanced DataFrame Operations**

.. code-block:: python

   import pandas as pd
   import chdb
   
   sales_df = pd.DataFrame({
       'product_id': [1, 2, 3, 1, 2, 3, 1, 2],
       'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Mouse'],
       'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
       'price': [999.99, 29.99, 79.99, 899.99, 24.99, 69.99, 1099.99, 34.99],
       'quantity': [2, 5, 3, 1, 8, 2, 1, 6],
       'sale_date': pd.date_range('2024-01-01', periods=8, freq='D')
   })
   
   result = chdb.query("""
       SELECT 
           category,
           product_name,
           count(*) as transaction_count,
           sum(price * quantity) as total_revenue,
           avg(price) as avg_price,
           sum(quantity) as total_units_sold,
           min(price) as min_price,
           max(price) as max_price
       FROM Python(sales_df)
       GROUP BY category, product_name
       ORDER BY total_revenue DESC
   """, "DataFrame")
   
   print("Sales Analysis:")
   print(result)
   
   daily_sales = chdb.query("""
       SELECT 
           sale_date,
           sum(price * quantity) as daily_revenue,
           count(*) as transaction_count,
           avg(price * quantity) as avg_transaction_value
       FROM Python(sales_df)
       GROUP BY sale_date
       ORDER BY sale_date
   """, "DataFrame")
   
   print("\nDaily Sales Trends:")
   print(daily_sales)

**Multiple DataFrame Joins**

.. code-block:: python

   products = pd.DataFrame({
       'product_id': [1, 2, 3, 4],
       'product_name': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
       'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics'],
       'cost': [750.00, 15.00, 45.00, 200.00]
   })
   
   orders = pd.DataFrame({
       'order_id': [1001, 1002, 1003, 1004, 1005],
       'product_id': [1, 2, 1, 3, 2],
       'quantity': [2, 5, 1, 3, 8],
       'order_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-17', '2024-01-18']
   })
   
   result = chdb.query("""
       SELECT 
           p.product_name,
           p.category,
           o.order_date,
           o.quantity,
           p.cost * o.quantity as total_cost,
           (p.cost * o.quantity * 1.4) as expected_revenue,
           ((p.cost * o.quantity * 1.4) - (p.cost * o.quantity)) as expected_profit
       FROM Python(orders) o
       JOIN Python(products) p ON o.product_id = p.product_id
       ORDER BY o.order_date, expected_profit DESC
   """, "DataFrame")
   
   print("Order Profitability Analysis:")
   print(result)

Text and String Processing
--------------------------

**String Analysis and Manipulation**

.. code-block:: python

   text_data = pd.DataFrame({
       'id': range(1, 6),
       'text': [
           'The quick brown fox jumps over the lazy dog',
           'Python is a powerful programming language',
           'Data analysis with chDB is fast and efficient',
           'ClickHouse provides excellent analytical capabilities',
           'Machine learning requires clean and structured data'
       ],
       'category': ['Animals', 'Programming', 'Analytics', 'Database', 'ML']
   })
   
   result = chdb.query("""
       SELECT 
           id,
           category,
           text,
           length(text) as text_length,
           arrayJoin(splitByString(' ', text)) as word,
           length(arrayJoin(splitByString(' ', text))) as word_length
       FROM Python(text_data)
       WHERE length(arrayJoin(splitByString(' ', text))) > 4
       ORDER BY word_length DESC, category
   """, "DataFrame")
   
   print("Text Analysis - Long Words:")
   print(result.head(10))

**Pattern Matching and Regular Expressions**

.. code-block:: python

   contact_data = pd.DataFrame({
       'id': [1, 2, 3, 4, 5],
       'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown', 'Charlie Wilson'],
       'contact_info': [
           'john.doe@email.com phone:123-456-7890',
           'jane.smith@company.org mobile:987-654-3210',
           'bob@invalid-email office:555-0123',
           'alice.brown@university.edu',
           'charlie.wilson@startup.io tel:+1-800-555-0199'
       ]
   })
   
   result = chdb.query("""
       SELECT 
           name,
           contact_info,
           extractAll(contact_info, '[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}')[1] as email,
           extractAll(contact_info, '\\d{3}-\\d{3}-\\d{4}')[1] as phone,
           match(contact_info, '.*\\.edu.*') as is_university,
           match(contact_info, '.*\\.com.*') as is_commercial
       FROM Python(contact_data)
       WHERE email != ''
   """, "DataFrame")
   
   print("Contact Information Extraction:")
   print(result)

Advanced Analytics
------------------

**Window Functions and Time Series**

.. code-block:: python

   import pandas as pd
   import numpy as np
   dates = pd.date_range('2024-01-01', periods=30, freq='D')
   ts_data = pd.DataFrame({
       'date': dates,
       'sales': [100 + i*5 + (i%7)*10 + np.random.randint(-20, 20) for i in range(30)],
       'visitors': [1000 + i*20 + (i%7)*50 + np.random.randint(-100, 100) for i in range(30)]
   })
   
   result = chdb.query("""
       SELECT 
           date,
           sales,
           visitors,
           avg(sales) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as sales_7day_avg,
           avg(visitors) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as visitors_7day_avg,
           
           sum(sales) OVER (ORDER BY date) as sales_cumulative,
           
           lag(sales, 1) OVER (ORDER BY date) as prev_day_sales,
           sales - lag(sales, 1) OVER (ORDER BY date) as daily_sales_change,
           
           -- Percentiles
           percent_rank() OVER (ORDER BY sales) as sales_percentile,
           row_number() OVER (ORDER BY sales DESC) as sales_rank
       FROM Python(ts_data)
       ORDER BY date
   """, "DataFrame")
   
   print("Time Series Analysis with Window Functions:")
   print(result.head(10))

**Statistical Analysis**

.. code-block:: python

   result = chdb.query("""
       WITH stats AS (
           SELECT 
               sales,
               visitors,
               sales / visitors * 1000 as conversion_rate
           FROM Python(ts_data)
       )
       SELECT 
           count(*) as n_observations,
           
           avg(sales) as sales_mean,
           median(sales) as sales_median,
           stddevPop(sales) as sales_std,
           min(sales) as sales_min,
           max(sales) as sales_max,
           
           avg(visitors) as visitors_mean,
           median(visitors) as visitors_median,
           stddevPop(visitors) as visitors_std,
           
           avg(conversion_rate) as avg_conversion_rate,
           stddevPop(conversion_rate) as conversion_rate_std,
           
           -- Percentiles
           quantile(0.25)(sales) as sales_q25,
           quantile(0.75)(sales) as sales_q75,
           quantile(0.95)(sales) as sales_q95,
           corr(sales, visitors) as sales_visitors_correlation
       FROM stats
   """, "DataFrame")
   
   print("Statistical Summary:")
   print(result.T)

Complex Data Transformations
-----------------------------

**Array Operations**

.. code-block:: python

   array_data = pd.DataFrame({
       'user_id': [1, 2, 3, 4, 5],
       'interests': [
           'sports,music,travel',
           'technology,gaming,programming',
           'cooking,reading,gardening',
           'fitness,photography,art',
           'movies,books,writing'
       ],
       'scores': [
           '85,92,78',
           '95,88,91',
           '77,83,89',
           '92,79,85',
           '88,94,82'
       ]
   })
   
   result = chdb.query("""
       SELECT 
           user_id,
           interests,
           splitByString(',', interests) as interests_array,
           arrayJoin(splitByString(',', interests)) as individual_interest,
           length(splitByString(',', interests)) as num_interests,
           
           scores,
           arrayMap(x -> toFloat64(x), splitByString(',', scores)) as scores_array,
           arrayReduce('avg', arrayMap(x -> toFloat64(x), splitByString(',', scores))) as avg_score,
           arrayReduce('max', arrayMap(x -> toFloat64(x), splitByString(',', scores))) as max_score
       FROM Python(array_data)
   """, "DataFrame")
   
   print("Array Operations Example:")
   print(result)

**Conditional Logic and Case Statements**

.. code-block:: python

   employee_data = pd.DataFrame({
       'employee_id': range(1, 11),
       'department': ['Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering', 
                     'HR', 'Marketing', 'Engineering', 'Sales', 'HR'],
       'salary': [45000, 85000, 55000, 48000, 90000, 52000, 58000, 95000, 47000, 54000],
       'years_experience': [2, 8, 4, 3, 10, 5, 6, 12, 2, 7],
       'performance_score': [3.2, 4.8, 3.9, 3.5, 4.9, 4.1, 4.2, 4.7, 3.1, 4.3]
   })
   
   result = chdb.query("""
       SELECT 
           employee_id,
           department,
           salary,
           years_experience,
           performance_score,
           
           CASE 
               WHEN salary >= 80000 THEN 'Senior'
               WHEN salary >= 60000 THEN 'Mid-level'
               WHEN salary >= 40000 THEN 'Junior'
               ELSE 'Entry-level'
           END as salary_band,
           
           CASE 
               WHEN performance_score >= 4.5 THEN 'Exceptional'
               WHEN performance_score >= 4.0 THEN 'Excellent'
               WHEN performance_score >= 3.5 THEN 'Good'
               ELSE 'Needs Improvement'
           END as performance_category,
           
           CASE 
               WHEN performance_score >= 4.5 AND salary >= 80000 THEN salary * 0.15
               WHEN performance_score >= 4.0 THEN salary * 0.10
               WHEN performance_score >= 3.5 THEN salary * 0.05
               ELSE 0
           END as bonus_amount,
           
           multiIf(
               department = 'Engineering' AND years_experience >= 8, 'Senior Engineer',
               department = 'Sales' AND performance_score >= 4.0, 'Top Performer',
               department = 'Marketing' AND years_experience >= 5, 'Marketing Lead',
               'Regular Employee'
           ) as role_classification
           
       FROM Python(employee_data)
       ORDER BY salary DESC, performance_score DESC
   """, "DataFrame")
   
   print("Employee Analysis with Complex Logic:")
   print(result)

Performance Optimization Examples
---------------------------------

**Large Dataset Processing**

.. code-block:: python

   result = chdb.query("""
       SELECT 
           toYYYYMM(date_column) as year_month,
           category,
           count(*) as record_count,
           sum(amount) as total_amount,
           avg(amount) as avg_amount
       FROM file('large_dataset.csv', 'CSV')
       WHERE date_column >= '2024-01-01'
           AND amount > 0
           AND category IN ('A', 'B', 'C')
       GROUP BY toYYYYMM(date_column), category
       ORDER BY year_month DESC, total_amount DESC
       LIMIT 1000
   """)
   print(result)

**Memory-Efficient Streaming**

.. code-block:: python

   conn = chdb.connect()
   cur = conn.cursor()
   
   cur.execute("""
       SELECT user_id, action, timestamp, details
       FROM file('large_log_file.csv', 'CSV')
       WHERE timestamp >= '2024-01-01'
       ORDER BY timestamp
   """)
   
   batch_size = 1000
   batch_count = 0
   
   while True:
       rows = cur.fetchmany(batch_size)
       if not rows:
           break
       
       batch_count += 1
       print(f"Processing batch {batch_count}: {len(rows)} rows")
       
       for row in rows:
           pass
   
   conn.close()
   print(f"Processed {batch_count} batches total")

Error Handling and Debugging
-----------------------------

**Query Debugging and Validation**

.. code-block:: python

   import chdb
   
   def safe_query(sql, format="CSV", description=""):
       """Execute query with proper error handling"""
       try:
           print(f"Executing: {description}")
           print(f"SQL: {sql}")
           
           result = chdb.query(sql, format)
           print("Query executed successfully")
           return result
           
       except chdb.ChdbError as e:
           print(f"chDB Error: {e}")
           return None
       except Exception as e:
           print(f"Unexpected error: {e}")
           return None
   
   result = safe_query("""
       SELECT 
           count(*) as total_rows,
           count(DISTINCT column_name) as unique_values,
           min(date_column) as earliest_date,
           max(date_column) as latest_date
       FROM file('data.csv', 'CSV')
   """, "DataFrame", "Data validation query")
   
   if result is not None:
       print("Query Results:")
       print(result)
   else:
       print("Query failed - check your data and SQL syntax")

Next Steps
----------

These examples demonstrate chDB's versatility and power. To continue learning:

- Explore the :doc:`udf` guide for custom functions
- Check :doc:`session` for stateful operations  
- Review :doc:`dbapi` for DB-API 2.0 compatibility
- See :doc:`api` for complete function reference

For more advanced use cases, visit the `chDB GitHub repository <https://github.com/chdb-io/chdb>`_ and community discussions.
