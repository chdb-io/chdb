User Defined Functions (UDF)
=============================

chDB supports Python User Defined Functions (UDFs) that allow you to extend SQL capabilities with custom Python logic. UDFs can be called from SQL queries and integrate seamlessly with chDB's query engine.

Overview
--------

Python UDFs in chDB provide:

- **Custom Business Logic**: Implement domain-specific calculations and transformations
- **External Library Integration**: Use any Python library within your SQL queries
- **Data Processing**: Advanced text processing, JSON manipulation, and mathematical functions
- **Stateless Operations**: Functions are called for each row of input data

API Reference
-------------

.. automodule:: chdb.udf
   :members:
   :show-inheritance:

Creating UDFs
--------------

**Basic UDF Registration**

.. code-block:: python

   from chdb.udf import chdb_udf
   from chdb import query
   
   # Simple UDF with default String return type
   @chdb_udf()
   def sum_udf(lhs, rhs):
       return int(lhs) + int(rhs)
   
   # Use the UDF in a query
   result = query("SELECT sum_udf(12, 22)")
   print(result)  # Returns: 34

**Specifying Return Types**

.. code-block:: python

   # UDF with specific return type
   @chdb_udf(return_type="UInt64")
   def calculate_total(price, quantity):
       return int(price) * int(quantity)
   
   # UDF returning Float64
   @chdb_udf(return_type="Float64")
   def calculate_average(total, count):
       return float(total) / float(count) if int(count) > 0 else 0.0
   
   # Use in complex queries
   result = query("""
       SELECT 
           product_name,
           calculate_total(price, quantity) as total_value,
           calculate_average(total_sales, days_active) as avg_daily_sales
       FROM sales_data
   """)

Text Processing UDFs
--------------------

**String Manipulation**

.. code-block:: python

   from chdb.udf import chdb_udf
   from chdb import query
   
   @chdb_udf()
   def clean_text(text):
       # Import modules inside the function
       import re
       import string
       
       # Remove punctuation and normalize
       text = text.translate(str.maketrans('', '', string.punctuation))
       text = re.sub(r'\s+', ' ', text.strip().lower())
       return text
   
   @chdb_udf()
   def extract_domain(email):
       import re
       pattern = r'@([A-Za-z0-9.-]+\.[A-Za-z]{2,})'
       match = re.search(pattern, email)
       return match.group(1) if match else ''
   
   # Usage example
   result = query("""
       SELECT 
           email,
           extract_domain(email) as domain,
           clean_text(description) as clean_desc
       FROM user_data
   """)

**Advanced Text Analysis**

.. code-block:: python

   @chdb_udf()
   def sentiment_score(text):
       # Simple sentiment scoring (you could use NLTK, TextBlob, etc.)
       import re
       
       positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful']
       negative_words = ['bad', 'terrible', 'awful', 'horrible', 'poor']
       
       words = re.findall(r'\b\w+\b', text.lower())
       
       positive_count = sum(1 for word in words if word in positive_words)
       negative_count = sum(1 for word in words if word in negative_words)
       
       return str(positive_count - negative_count)

JSON and Data Processing UDFs
------------------------------

**JSON Manipulation**

.. code-block:: python

   @chdb_udf()
   def parse_json_field(json_str, field_path):
       import json
       
       try:
           data = json.loads(json_str)
           
           # Support nested field paths like "user.profile.name"
           fields = field_path.split('.')
           current = data
           
           for field in fields:
               if isinstance(current, dict) and field in current:
                   current = current[field]
               else:
                   return ''
                   
           return str(current)
       except:
           return ''
   
   @chdb_udf()
   def json_array_length(json_str):
       import json
       
       try:
           data = json.loads(json_str)
           if isinstance(data, list):
               return str(len(data))
           return '0'
       except:
           return '0'
   
   # Usage in analytics queries
   result = query("""
       SELECT 
           user_id,
           parse_json_field(metadata, 'profile.age') as age,
           parse_json_field(metadata, 'preferences.theme') as theme,
           json_array_length(tags) as tag_count
       FROM user_events
   """)

Mathematical and Statistical UDFs
----------------------------------

**Custom Calculations**

.. code-block:: python

   @chdb_udf(return_type="Float64")
   def compound_interest(principal, rate, time):
       # Calculate compound interest
       import math
       p = float(principal)
       r = float(rate) / 100  # Convert percentage to decimal
       t = float(time)
       
       amount = p * math.pow(1 + r, t)
       return amount - p  # Return interest earned
   
   @chdb_udf(return_type="Float64")
   def haversine_distance(lat1, lon1, lat2, lon2):
       # Calculate distance between two coordinates
       import math
       
       # Convert latitude and longitude from degrees to radians
       lat1, lon1, lat2, lon2 = map(math.radians, 
                                   [float(lat1), float(lon1), float(lat2), float(lon2)])
       
       # Haversine formula
       dlat = lat2 - lat1
       dlon = lon2 - lon1
       a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
       c = 2 * math.asin(math.sqrt(a))
       r = 6371  # Radius of earth in kilometers
       
       return c * r

Data Validation UDFs
--------------------

**Input Validation and Cleaning**

.. code-block:: python

   @chdb_udf()
   def validate_email(email):
       import re
       
       pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'
       return 'true' if re.match(pattern, email) else 'false'
   
   @chdb_udf()
   def normalize_phone(phone):
       import re
       
       # Remove all non-digit characters
       digits = re.sub(r'[^\d]', '', phone)
       
       # Format as US phone number
       if len(digits) == 10:
           return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
       elif len(digits) == 11 and digits[0] == '1':
           return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
       else:
           return digits
   
   @chdb_udf()
   def clean_currency(amount):
       import re
       
       # Remove currency symbols and formatting
       cleaned = re.sub(r'[^\d.-]', '', amount)
       
       try:
           return str(float(cleaned))
       except:
           return '0.0'

UDF with External Libraries
----------------------------

**Using Third-party Libraries**

.. code-block:: python

   @chdb_udf()
   def hash_password(password):
       # Example using hashlib (built-in)
       import hashlib
       
       return hashlib.sha256(password.encode()).hexdigest()
   
   @chdb_udf()
   def parse_user_agent(ua_string):
       # This would require installing user-agents library
       # pip install user-agents
       try:
           from user_agents import parse
           
           user_agent = parse(ua_string)
           return f"{user_agent.browser.family}|{user_agent.os.family}"
       except ImportError:
           # Fallback if library not available
           return "unknown|unknown"
   
   @chdb_udf()
   def geocode_ip(ip_address):
       # This would require a geolocation service
       # For demo purposes, returning mock data
       import re
       
       # Simple IP validation
       if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', ip_address):
           return f"Country:Unknown,City:Unknown"
       return "Invalid IP"

UDF Best Practices and Guidelines
-----------------------------------

**Important Notes from README**

Based on the official documentation, here are the key guidelines for chDB UDFs:

1. **Stateless Functions**: UDFs should be stateless. Only User Defined Functions are supported, not User Defined Aggregation Functions (UDAFs).

2. **Default Return Type**: Default return type is String. Specify return type explicitly for other types.

3. **Input Arguments**: All function arguments are received as strings (TabSeparated format), so convert them as needed.

4. **Row Processing**: The function is called for each line of input, similar to:

.. code-block:: python

   # Conceptual execution model
   def sum_udf(lhs, rhs):
       return int(lhs) + int(rhs)

   for line in sys.stdin:
       args = line.strip().split('\t')
       lhs = args[0]
       rhs = args[1]
       print(sum_udf(lhs, rhs))
       sys.stdout.flush()

5. **Module Imports**: Import all required Python modules **inside the function**:

.. code-block:: python

   @chdb_udf()
   def func_use_json(arg):
       import json  # Import inside the function
       
       try:
           data = json.loads(arg)
           return str(data.get('key', ''))
       except:
           return ''

6. **Python Interpreter**: The Python interpreter used is the same as the one running the script (from ``sys.executable``).

UDF Configuration and Advanced Usage
------------------------------------

**UDF Path Configuration**

.. code-block:: python

   import chdb
   
   # Specify custom UDF path when querying
   result = chdb.query(
       "SELECT my_custom_function(column) FROM table",
       udf_path="./my_udfs/"
   )

**Supported Data Types**

UDFs support all ClickHouse data types. Specify return type using ClickHouse type names:

.. code-block:: python

   @chdb_udf(return_type="UInt32")
   def count_words(text):
       import re
       words = re.findall(r'\b\w+\b', text)
       return len(words)
   
   @chdb_udf(return_type="Array(String)")
   def split_text(text, delimiter):
       return text.split(delimiter)
   
   @chdb_udf(return_type="Decimal(10,2)")
   def calculate_tax(amount, rate):
       return float(amount) * float(rate) / 100

**Error Handling in UDFs**

Always implement proper error handling:

.. code-block:: python

   @chdb_udf()
   def safe_divide(a, b):
       try:
           numerator = float(a)
           denominator = float(b)
           
           if denominator == 0:
               return 'NULL'
           
           return str(numerator / denominator)
       except (ValueError, TypeError):
           return 'ERROR'

Performance Considerations
--------------------------

**Optimization Tips**

1. **Minimize Imports**: Only import what you need inside the function
2. **Avoid Heavy Computations**: UDFs are called for each row
3. **Use Built-in Functions When Possible**: ClickHouse built-ins are usually faster
4. **Cache Expensive Operations**: For complex calculations, consider pre-processing data

.. code-block:: python

   @chdb_udf()
   def optimized_function(input_data):
       # Import only what's needed
       import json
       
       # Handle common cases quickly
       if not input_data or input_data == 'null':
           return ''
       
       try:
           # Main processing
           data = json.loads(input_data)
           return str(data.get('result', ''))
       except:
           return ''

Debugging UDFs
--------------

**Testing UDFs Locally**

.. code-block:: python

   # Test your UDF independently before using in queries
   @chdb_udf()
   def test_function(input_val):
       import json
       try:
           data = json.loads(input_val)
           return str(data.get('value', 0))
       except Exception as e:
           # Debug: print error (remove in production)
           print(f"Error: {e}")
           return '0'
   
   # Test locally first
   test_input = '{"value": 42}'
   result = test_function(test_input)
   print(f"Test result: {result}")
   
   # Then use in query
   query_result = query("SELECT test_function('{}') as result".format(test_input))

**Common Issues and Solutions**

1. **Import Errors**: Always import inside the function
2. **Type Conversion**: All inputs are strings, convert explicitly
3. **Return Type Mismatch**: Ensure returned value matches declared type
4. **Exception Handling**: Wrap operations in try-catch blocks

Complete Example: Log Processing UDF
------------------------------------

Here's a comprehensive example that demonstrates multiple UDF concepts:

.. code-block:: python

   from chdb.udf import chdb_udf
   from chdb import query
   
   @chdb_udf()
   def parse_log_entry(log_line):
       """Parse Apache/Nginx log entries"""
       import re
       import json
       from datetime import datetime
       
       # Apache Common Log Format regex
       pattern = r'(\S+) \S+ \S+ \[(.*?)\] "(.*?)" (\d+) (\d+|-) "(.*?)" "(.*?)"'
       
       try:
           match = re.match(pattern, log_line)
           if not match:
               return json.dumps({"error": "Invalid log format"})
           
           ip, timestamp, request, status, size, referer, user_agent = match.groups()
           
           # Parse request
           request_parts = request.split(' ')
           method = request_parts[0] if len(request_parts) > 0 else ''
           path = request_parts[1] if len(request_parts) > 1 else ''
           
           result = {
               "ip": ip,
               "timestamp": timestamp,
               "method": method,
               "path": path,
               "status": int(status),
               "size": int(size) if size != '-' else 0,
               "referer": referer if referer != '-' else '',
               "user_agent": user_agent
           }
           
           return json.dumps(result)
           
       except Exception as e:
           return json.dumps({"error": str(e)})
   
   @chdb_udf()
   def extract_log_field(log_json, field_name):
       """Extract specific field from parsed log JSON"""
       import json
       
       try:
           data = json.loads(log_json)
           return str(data.get(field_name, ''))
       except:
           return ''
   
   # Example usage
   sample_log = '192.168.1.1 - - [10/Oct/2000:13:55:36 -0700] "GET /index.html HTTP/1.0" 200 2326 "-" "Mozilla/4.0"'
   
   result = query(f"""
       SELECT 
           parse_log_entry('{sample_log}') as parsed,
           extract_log_field(parse_log_entry('{sample_log}'), 'ip') as client_ip,
           extract_log_field(parse_log_entry('{sample_log}'), 'status') as status_code
   """)
   
   print(result)

See Also
--------

- :doc:`troubleshooting` - UDF troubleshooting and common issues
- :doc:`examples` - More UDF examples and use cases  
- :doc:`api` - Complete API reference
- `ClickHouse Data Types <https://clickhouse.com/docs/en/sql-reference/data-types>`_ - Supported return types