User Defined Functions (UDF)
=============================

chDB supports Python User Defined Functions (UDFs) that can be called from SQL queries.

UDF Module
----------

.. automodule:: chdb.udf.udf
   :show-inheritance:

UDF Interface
-------------

.. automodule:: chdb.udf
   :show-inheritance:

Creating UDFs
--------------

Define Python functions that can be called from SQL:

.. code-block:: python

   import chdb
   from chdb import udf
   
   # Define a simple UDF
   @udf.register("my_add", return_type="Int32")
   def my_add(a: int, b: int) -> int:
       return a + b
   
   # Use the UDF in a query
   result = chdb.query("SELECT my_add(1, 2) as result")
   print(result)  # 3

Advanced UDFs
-------------

UDFs can work with various data types:

.. code-block:: python

   import chdb
   from chdb import udf
   
   # String processing UDF
   @udf.register("reverse_string", return_type="String")
   def reverse_string(s: str) -> str:
       return s[::-1]
   
   # Array processing UDF
   @udf.register("sum_array", return_type="Int64")
   def sum_array(arr: list) -> int:
       return sum(arr)
   
   # Use UDFs in queries
   result = chdb.query("""
       SELECT 
           reverse_string('hello') as reversed,
           sum_array([1, 2, 3, 4, 5]) as array_sum
   """)
   print(result)

UDF with External Libraries
----------------------------

UDFs can use external Python libraries:

.. code-block:: python

   import chdb
   from chdb import udf
   import json
   import re
   
   # JSON processing UDF
   @udf.register("parse_json", return_type="String")
   def parse_json(json_str: str, key: str) -> str:
       try:
           data = json.loads(json_str)
           return str(data.get(key, ""))
       except:
           return ""
   
   # Regex UDF
   @udf.register("extract_email", return_type="String")
   def extract_email(text: str) -> str:
       pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
       match = re.search(pattern, text)
       return match.group(0) if match else ""

UDF Configuration
-----------------

Configure UDF behavior and paths:

.. code-block:: python

   import chdb
   
   # Specify UDF path
   result = chdb.query(
       "SELECT my_custom_function(column) FROM table",
       udf_path="./my_udfs/"
   )