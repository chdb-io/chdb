## Background

chDB's core is the ClickHouse database engine. While the most commonly used table engine is the MergeTree table engine (and its variants), ClickHouse also supports numerous other table engines, bringing unique advantages to data science on the table. We hope that chDB can provide data scientists with a unified, efficient, and easy-to-use data processing interface that fully leverages the advantages of ClickHouse while maintaining seamless integration with the existing Python data science ecosystem.

## Core Advantages

### Local File Table Engines

- **Rich format support**: CSV, Parquet, ORC, JSON, MySQL dumps, and a lot of other data formats
- **Zero-copy access (no ETL)**: Directly query local files without data transformation or migration
- **Local development friendly**: Supports common local file workflows for data scientists

### Remote Server Table Engines

- **Protocol support**: HTTP, S3, FTP, HDFS, and many other remote access protocols
- **Format compatibility**: Supports the same data formats as the local file engines
- **Query pushdown**: Pushdown of database operations (e.g., filters) to the remote data based on internal database engine abstraction

### Database Table Engines

- **Multi-database support**: SQLite, MySQL, PostgreSQL, etc.
- **Data Lake integration**: Modern data lake formats like Iceberg and DeltaLake
- **Cross-data source operations**: Support for join and aggregation operations across different data sources

### Data Science Workflow Advantages

- **Enhanced feature engineering**: Efficient feature joining based on View and Materialized View
- **Local development optimization**: Local sampling data development, reducing remote database pressure. Avoiding complex ACL applications and rate-limiting logic
- **Query optimization**: Leveraging ClickHouse's query optimization capabilities, including query pushdown
- **High-performance JSON support**: ClickHouse's JSON support is very powerful, enabling complex JSON column query operations

## Target Scenarios and Users

- Data scientists and data analysts who need to process multi-source, large-scale, and multi-modal data, who are familiar with Pandas or SQL and require high performance
- Data-intensive application developers who need to process large-scale and multi-modal data, requiring high-performance computing capabilities, accustomed to Python's Function Chaining or SQL

## Core Differences from Competitors

- vs **Pandas**
    - **Big data processing capability**: chDB is based on ClickHouse columnar storage, breaking through memory limitations to process TB-level data, while Pandas is limited by single-machine memory capacity
    - **Unified multi-source querying**: Can directly query files, databases, cloud storage and other data sources without pre-loading into memory, while Pandas needs to read data into DataFrame first
    - **Native SQL support**: Provides complete SQL query capabilities and ClickHouse function library, supporting complex analytical queries, while Pandas mainly relies on DataFrame API for data operations
- vs **Polars**
    - **Multi-source integration**: chDB supports cross-database and cross-filesystem queries, can directly query MySQL, S3, HDFS, etc., while Polars mainly focuses on high-performance processing of single data sources
    - **Native SQL support**: chDB is based on ClickHouse engine with native support for complex SQL queries and functions, while Polars supports SQL but is still primarily DataFrame API driven
    - **ClickHouse ecosystem advantages**: Inherits ClickHouse's columnar storage optimization, query optimizer and rich function library, with stronger performance in OLAP scenarios
- vs **DuckDB**
    - **Rich table engines**: chDB supports more diverse table engines (remote databases, Data Lake, streaming data, etc.), while DuckDB mainly focuses on local OLAP analysis
    - **Distributed capabilities**: Based on ClickHouse engine, natively supports distributed queries and computing, while DuckDB is a single-machine engine
    - **Data science API**: chDB specifically designed Pythonic API for data science workflows, supporting Pandas function chaining, while DuckDB mainly provides SQL interface
- vs **PySpark**
    - **Local development friendly**: chDB supports direct querying of local files without cluster deployment, making development and debugging more convenient, while PySpark requires Spark cluster environment
    - **Zero-copy Pandas integration**: Seamless integration with Pandas DataFrame, can be directly used in existing data science workflows, while PySpark requires data conversion overhead
    - **Lightweight deployment**: Can run in a single process with low resource consumption, while PySpark requires JVM and distributed framework overhead

## Core Concepts

### DataStore

`DataStore` is the core concept of the new API, similar to tables in databases, as an abstraction of data:

- **Schema support**: Typically each DataStore has a clear Schema definition, if data is schema-less. user could `setSchema` or `inferSchema` to make it specified.
- **Type compatibility**: Conceptually consistent with Pandas DataFrame
- **Engine encapsulation**: Unified encapsulation of various ClickHouse table engines

### Function Chaining

API design supports Pandas-like chaining calls:

- **Fluent interface**: Supports `.function().function()` chaining operations
- **Lazy computation**: Supports Lazy Evaluation, only actually executes when `execute()` is called
- **Query optimization**: Chaining operations generate unified query plans for optimization

### Zero-copy Compatible Pandas DataFrame

Direct support for zero-copy read/write of Pandas DataFrame

- **Type consistency**: Generated DataFrame is fully compatible with Pandas (not Polars style)
- **Seamless conversion**: DataStore and DataFrame can be seamlessly converted
- **Pandas functions**: Supports chaining calls of Pandas functions

## API Design Decisions

### Pythonic Style API Design

- Rather than `ds.filter("a > 10")`, we prefer `ds.filter(col("a") > 10)`, which is more semantic and conforms to Python programming habits. The latter is also more convenient for debugging and easier to generate code for by an AI-assisted IDE.
- As syntactic sugar, `ds.filter("a > 10")` can also be written as `ds.filter(ds.a > 10)`, which is more concise.
- Due to Python implementation limitations, `ds.filter(a > 10)` cannot be implemented due to variable resolution issues, but `ds.filter(ds.a > 10)` can be implemented.
- Considering usability and SQL compatibility, we still provide string expressions (SQL fragments) like `ds.filter("a > 10 AND b = 'x'")` for user convenience.

### Using `DataStore` as Data Source Abstraction

- chDB supports zero-copy read/write of Pandas DataFrame (currently only supports reading), `DataStore` can be considered a thin wrapper of Pandas DataFrame. In query engine implementation, chDB automatically maps `DataStore` to ClickHouse tables or ClickHouse views.

### Supporting Pandas Functions

- In some data operations, Pandas or Python functions are the most convenient expression. chDB supports chaining calls of Pandas functions. However, we don't guarantee complete compatibility with Pandas functions, we try to maintain semantic consistency with Pandas functions.
    - One of the example is **Chinese Word Segmentation.** You can use a lot of Python third party libs do that, but most SQL engine do not has such kind of thing
    - Another example is while pre-training LLM, we need to map every word to a token value. It’s easy and straightforward to just use a Python map to handle it.

### Supporting SQL Syntax

- In some data operations, SQL is still the most convenient expression. chDB supports SQL fragments to build DataStore query conditions. chDB automatically converts SQL fragments into final SQL-based query statements.

## Architecture Design

- For function chain expressions, we default to Lazy Evaluation, only actually building ClickHouse SQL-based query statements when `execute()` is called. We will record all operations and their dependencies in the class instance. When `execute()` is called, we generate the final SQL statement based on the info and submit it to the ClickHouse engine for query execution.
- Users may call Pandas functions in the function chain, these functions will be executed directly in Pandas and will not be converted to ClickHouse SQL. Based on the current implementation plan, the entire execution plan may be split into multiple segments, like Data -> ClickHouse -> Pandas -> ClickHouse. But throughout the process, Pandas DataFrame will serve as intermediate results and will not be converted to ClickHouse tables.

## Common APIs

> The following text is not the final API design, only for reference of related API styles.
> 

### Creating DataStore

We can create a DataStore object through `DataStore()`. At this time, the DataStore object is empty with no Schema information. But this object can already be used for function chain calls.

Let’s say a user is creating a PostgreSQL based DataSource, but he does not really want to connect to a remote server while still coding. Then he could set a schema and test data to make the demo work.

```python
pg = DataStore("pg", connection="someServer")
#pg.connect()
pg.mockData('{"a":[1,2,3], "b":["auxten","tom","jerry"]}')
...
```

Through `DataStore.connect()` you can try to connect to the data source and get Schema information (when connection is successful and Schema can be obtained in O(1) time). Connection failure will throw exceptions, making it easy to discover filesystem, network or authentication issues early. 

```python
from chdb import DataStore

# Create local Parquet file data source
pq = DataStore("file", path="data/sales.parquet")
pq.connect()

# Create ClickHouse table data source, supports ClickHouse connection strings, reference: <https://clickhouse.com/docs/sql-reference/table-functions/remote>
ch_table = DataStore("clickhouse", host="localhost", table="customer_info")
```

### Select/Drop Column

```python
ds.select("a", "b")
ds.drop("a", "b")
# or
ds.select(col("a"), col("b"))
ds.drop(col("a"), col("b"))
# or
ds.select(col("a") + 1, col("b") * 2)
# or
ds.select(ds.a + 1, ds.b * 2)
```

### Filter (Select)

```python
ds.filter(col("a") > 5 & col("b") == "x")
# or
ds.filter(ds.a > 5 & ds.b == "x")
```

Also supports SQL fragments and string expressions

```python
ds.filter("a > 5 AND b = 'x'")
```

### Assign(Mutate)

Similar to Pandas' `assign` method, supports updating or adding columns

```python
ds.assign(col("a") = col("a") + 1)
# or
ds.assign(a=col("a") + 1)
# or
ds.assign(a=ds.a + 1)
# or
ds.assign(a="a + 1")
```

Update multiple columns

```python
ds.assign(a="a + 1", new_b="b * 2")
```

Based on string expressions

```python
ds.assign(a="a + 1", b="b * 2")
```

### Aggregate

```python
ds.groupby("a").agg(   # col("a") will be auto added to aggregation results
  total=col("b").sum(),
  avg=col("c").mean(),
  count=col("d").count()
)
```

### Join

```python
ds.join(other_ds, on="a")
# or
ds.join(other_ds, left_on="a", right_on="b", how="left")
```

### Sort

```python
ds.sort("a", "b", ascending=True)
# or
ds.sort(col("a"), col("b"), ascending=False))
```

### NaN,Null,None Handling

```python
ds.fillna(0) # np.NaN
ds.fillnull(0) # or ds.fillnone(0)
ds.dropna()
ds.dropnull() # or ds.dropnone
# or
ds.fillna(0, subset=["a", "b"])  # Fill specified columns
ds.fillnull(strategy="mean", subset=["a", "b"])  # Fill column a, b using mean
```

### Conditional Column Creation (CASE WHEN)

Create columns with conditional logic, equivalent to SQL `CASE WHEN` or `np.where()`/`np.select()`:

```python
# Simple binary condition (equivalent to np.where)
ds['is_high'] = ds.when(ds['value'] >= 100, 'high').otherwise('low')

# Multiple conditions (equivalent to nested np.where or np.select)
ds['grade'] = (
    ds.when(ds['score'] >= 90, 'A')
      .when(ds['score'] >= 80, 'B')
      .when(ds['score'] >= 70, 'C')
      .when(ds['score'] >= 60, 'D')
      .otherwise('F')
)

# Using column expressions as values
ds['result'] = ds.when(ds['a'] > ds['b'], ds['a']).otherwise(ds['b'])

# With arithmetic expressions
ds['adjusted'] = ds.when(ds['value'] < 0, 0).otherwise(ds['value'] * 2)

# Compound conditions
ds['segment'] = (
    ds.when((ds['age'] >= 60) | (ds['income'] >= 80000), 'Premium')
      .when((ds['age'] >= 40) & (ds['income'] >= 50000), 'Standard')
      .otherwise('Basic')
)
```

This is semantically equivalent to:
```python
# np.where (binary)
df['is_high'] = np.where(df['value'] >= 100, 'high', 'low')

# np.select (multiple conditions)
conditions = [df['score'] >= 90, df['score'] >= 80, df['score'] >= 70, df['score'] >= 60]
choices = ['A', 'B', 'C', 'D']
df['grade'] = np.select(conditions, choices, default='F')
```

**Execution Engine Configuration:**

By default, `when().otherwise()` uses the chDB SQL engine for better performance. You can switch to pandas (`np.select`) via configuration:

```python
from datastore import function_config

# Default: use chDB SQL engine
function_config.use_chdb('when')  # This is the default

# Switch to pandas (np.select) execution
function_config.use_pandas('when')

# Check which engine an expression will use
expr = ds.when(ds['score'] >= 90, 'A').otherwise('B')
print(expr.execution_engine())  # 'chDB' or 'Pandas'

# Reset to defaults
function_config.reset()
```

Use `ds.explain()` to see which engine will be used in the execution plan:
```python
ds['grade'] = ds.when(ds['score'] >= 90, 'A').otherwise('B')
ds.explain()
# Shows: [chDB] Assign column 'grade' = CASE WHEN ...
```

### SQL Fragments

```python
ds.filter("a > 5 AND b = 'x'")
```

Can also directly execute SQL statements

```python
ds.query("SELECT a, b FROM ds WHERE a > 5 AND b = 'x'")
```

### Explain & Profile

```python
ds.explain() # Print final SQL statement or execution plan
ds.profile() # Print final SQL statement execution plan and execution time
```

### Config

chDB supports setting some global configurations through `config`, such as default engine, memory limits, timeout, etc.

```python
from chdb import config

config.set_memory_limit("8GB")
config.set_timeout(300)
config.set_engine("remote") # or hybrid?
```

### Exception

```python
# Recommended error handling mechanism
try:
    result = ds.filter(...).join(...).execute()
except chdb.DataStoreNotFoundError:
    # Data source not found
except chdb.SchemaCompatibilityError:
    # Schema incompatible
except chdb.QueryTimeoutError:
    # Query timeout
```

## API Design Examples

### Basic Data Source Operations

```python
from chdb import DataStore, col

# Create local Parquet file data source
pq = DataStore("file", path="data/sales.parquet")

# Get Schema information
schema = pq.schema
print(schema)

# Basic filtering operation (lazy execution)
filtered_data = pq.filter(col("revenue") > 1000)

# Execute query and convert to DataFrame
result_ds = filtered_data.execute()

result_ds.to_df() # no cost
```

### Multi-Source Integration

```python
# ClickHouse table data source
ch_table = DataStore("clickhouse",
                            host="localhost",
                            table="customer_info")

# Local CSV file data source
csv_data = DataStore("file", path="local_sales.csv")

print(ch_table.schema)
print(csv_data.schema)

# Cross-source JOIN operation
joined_data = ch_table.join(csv_data, on="customer_id") \
                     .filter(col("purchase_date") >= "2024-01-01") \
                     .select("customer_name", "product", "revenue")

# Execute and get results
result = joined_data.execute()
```

Generated SQL from chDB SDK

```sql
SELECT 
    customer_name,
    product, 
    revenue
FROM (
    SELECT *
    FROM remote('localhost:9000', 'default', 'customer_info')
) AS ch_table
JOIN (
    SELECT *
    FROM file('local_sales.csv', 'CSVWithNames')
) AS csv_data
ON ch_table.customer_id = csv_data.customer_id
WHERE purchase_date >= '2024-01-01';
```

The above multi-source JOIN can be further implemented through `DataStore.from_multi_sources()` to facilitate unified data JOIN and feature engineering.

```python
from chdb import DataStore

ds = DataStore.from_multi_sources(
    DataStore("file", path="local_sales.csv"),
    DataStore("clickhouse", host="localhost", table="customer_info"),
    on="customer_id"
)
```

The underlying SQL:

```sql
SELECT *
FROM (
    SELECT *
    FROM file('local_sales.csv', 'CSVWithNames')
) AS source1
INNER JOIN (
    SELECT *
    FROM remote('localhost:9000', 'default', 'customer_info')  
) AS source2
ON source1.customer_id = source2.customer_id;
```

### Aggregation and Grouping Operations

```python
# Create data source
sales_data = DataStore("file", path="sales_data.parquet")

# Aggregation operations
aggregated = sales_data \
    .filter(col("category") == "electronics") \
    .groupby("region") \
    .agg(
        total_revenue=col("revenue").sum(),
        avg_quantity=col("quantity").mean(),
        order_count=col("order_id").count()
    )

# Further processing
final_result = aggregated \
    .filter(col("total_revenue") > 5000) \
    .sort("total_revenue", ascending=False) \
    .execute()
```

The underlying SQL:

```sql
SELECT 
    region,
    total_revenue,
    avg_quantity,
    order_count
FROM (
    SELECT 
        region,
        sum(revenue) AS total_revenue,
        avg(quantity) AS avg_quantity,
        count(order_id) AS order_count
    FROM file('sales_data.parquet', 'Parquet')
    WHERE category = 'electronics'
    GROUP BY region
) AS aggregated_data
WHERE total_revenue > 5000
ORDER BY total_revenue DESC;
```

### Data Transformation Operations

```python
# Read data from ClickHouse table
source_table = DataStore("clickhouse",
                                host="localhost",
                                table="raw_data").connect()

# Data processing and transformation
processed = source_table \
    .filter(col("status") == "active") \
    .assign(
        processed_score=col("score") + 1,
        grade=col("score") * 0.1
    ) \
    .select("id", "name", "processed_score", "grade").limit(100)

# Write back to original table (overwrite)
processed.save("new_table")
# view? or all in clickhouse server
processed.save_view("new_view")
processed.save_mview("new_executed_view")

# Save to file
processed.save("output.csv")
```

### Data Cleaning Operations

```python
# Create data source
raw_data = DataStore("file", path="messy_data.parquet")

# Data cleaning
cleaned_data = raw_data \
    .dropna() \
    .fillnull(0, subset=["revenue", "quantity"]) \
    .filter(raw_data.age > 0) \
    .assign(
        name_clean=raw_data.name.str().upper(),
        age_group=raw_data.age // 10 * 10
    ) \
    .drop("temp_column", "unused_field")

# Get cleaned data
result = cleaned_data.execute()
```

### **Remote API as a Table**

with the [Python defined table engine](https://github.com/chdb-io/chdb/blob/main/tests/test_query_py.py#L84) in chDB, we could even use HTTP API as a data source

```python
import chdb

class myReader(chdb.PyReader):
    def __init__(self, data):
        self.data = data
        self.cursor = 0
        super().__init__(data)

    def read(self, col_names, count):
        print("Python func read", col_names, count, self.cursor)
        if self.cursor >= len(self.data["a"]):
            self.cursor = 0
            return []
        block = [self.data[col] for col in col_names]
        self.cursor += len(block[0])
        return block

    def get_schema(self):
        return [
            ("a", "int"),
            ("b", "str"),
            ("dict_col", "json")
        ]

reader = myReader(
    {
        "a": [1, 2, 3, 4, 5, 6],
        "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
        "dict": [
            {'id': 1, 'tags': ['urgent', 't1'], 'meta': {'created': '2024-01-01'}},
            {'id': 2, 'tags': ['normal'], 'meta': {'created': '2024-02-01'}},
            {'id': 3, 'name': 'tom'},
            {'id': 4, 'value': '100'},
            {'id': 5, 'value': 101},
            {'id': 6, 'value': 102}
        ],
    }
)

chdb.query("SELECT b, sum(a) FROM Python(reader) GROUP BY b ORDER BY b").show()
chdb.query("SELECT dict.id FROM Python(reader) WHERE dict.value='100'").show()
```

Joining S3 data and API returned data

```python
# S3 data source
s3_data = DataStore("s3", path="s3://my-bucket/datasets/user_behavior.parquet")

# HTTP API data source
api_data = DataStore("http", url="https://api.example.com/users")

# Transparent remote data operations
analysis = s3_data \
    .join(api_data, on="user_id") \
    .groupby("user_segment") \
    .agg(
        avg_duration=col("session_duration").mean(),
        total_views=col("page_views").sum()
    ) \
    .filter(col("avg_duration") > 300)

# Lazy execution - only performs network transfer when needed
result = analysis.execute()

```

## Application Scenarios

### PyTorch DataLoader Integration

PyTorch's DataLoader can accept an iterable object. We can get an iterable object through `DataStore.iter()`, which can then be directly used for PyTorch's DataLoader.

Due to the specificity of machine learning training, we need to load data in batches and often need to shuffle the data. Pandas-based implementations are very memory intensive and perform poorly. PyTorch's internal implementation often relies on multiprocessing to circumvent GIL limitations, but multiprocessing implementations often require users to manually implement `torch.utils.data.Dataset` data shuffle and batch logic, with poor performance.

chDB's implementation can fully utilize ClickHouse's performance advantages to perform data shuffle output batches by epoch, while supporting Pandas interface for unified data JOIN and feature engineering.

Reference https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/dataloader.ipynb implementation showed below

```jsx

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    

for batch in dataloader:
    x, y = batch

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings
```

user can use the DataLoader from chDB to shuffle data in batch of every epoch:

```python
from chdb import DataStore

ds = DataStore("file", path="data/sales.parquet")

for batch in ds.iter(batch_size=100, shuffle=True):
    print(batch)
```

The underlying SQL of the code above could be something like:

```sql
SELECT *,
       cityHash64(toString(row_number() OVER ()), 'session_seed') as shuffle_key,
       intDiv(row_number() OVER (ORDER BY shuffle_key), 100) as batch_number
FROM file('data/sales.parquet', Parquet)
ORDER BY shuffle_key;
```

### Multi-Modal Data Processing

![*An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality.*](attachment:9e55f0ff-3a86-4989-a17f-97f058f8795c:multi-modal-llm.webp)

*An illustration of a multimodal LLM that can accept different input modalities (audio, text, images, and videos) and returns text as the output modality.*

Multi-modal data processing is an important scenario in machine learning, whether in traditional machine learning (recommendation, search) or Multi-Modal LLM, it is an important and complex data processing step.

Simply put, the main inference and training approach of Multi-modal LLM is to process multimedia data, generate embeddings or other normalized matrix data, and then concatenate this data with tokens produced from text as part of the inference and training process

![*Annotated figure adapted from the Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models paper: https://www.arxiv.org/abs/2409.17146.*](attachment:86a395e0-1715-4218-a2ff-2aac288a75bb:image.png)

*Annotated figure adapted from the Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models paper: https://www.arxiv.org/abs/2409.17146.*

For multi-modal data, data scientists often hope to simultaneously process text, images, audio and other multi-modal data during training. The final model input is often vectors from multi-modal data JOIN, rather than separate text, image, audio vectors.

![*Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191*](attachment:b29eacc8-b9cb-42dd-b84e-2675e4278307:image.png)

*Annotated figure from the Qwen2-VL paper: https://arxiv.org/abs/2409.12191*

chDB can support multi-modal data processing through subsequent support for multimedia data sources (e.g., image archives on S3) and basic image processing functions.

### Multi-Modal Data Processing Example

```python
from chdb import DataStore, col
from some_third_lib import clip_resize, embedding

# Create multiple modal data sources
text_data = DataStore("file", path="data/product_descriptions.parquet")
image_data = DataStore("s3", path="s3://ml-bucket/product_images.zip")
llm_qa_data = DataStore("file", path="data/llm_qa_dataset.jsonl")

# Multi-modal data JOIN and feature engineering
multimodal_features = text_data \
    .join(image_data, on="product_id", how="inner") \
    .join(llm_qa_data, on="product_id", how="left") \
    .assign(
        # Text feature processing
        text_length=col("description").length(),
        text_embedding=col("description").embedding(model_name="multilingual-e5-large", dim=1024),
        
        # Image feature processing (extracted from images in archive)
        image_height=col("image").height(),
        image_width=col("image").width(),
        image_channels=col("image").channels(),
        image_embedding=col("image").clip_resize(8, 224, 224).embedding(model_name="clip-vit-large-patch14", device="cuda", dim=1024),
        
        # LLM Q&A feature processing
        qa_question_length=col("question").length(),
        qa_answer_length=col("answer").length(),
        qa_response_quality=col("answer_score"),
        
        # use json function from clickhouse
        
        # Cross features
        text_image_ratio=col("text_length") / (col("image_height") * col("image_width")),
        qa_relevance=col("description").text_similarity(col("question")),
        multimodal_complexity=col("text_length") + col("image_channels") + col("qa_answer_length")
    ) \
    .filter(col("qa_relevance") > 0.6) \
    .select(
        "product_id", "description", "image",
        "question", "answer", "qa_response_quality",
        "text_image_ratio", "multimodal_complexity"
    )

# Execute multi-modal feature extraction
print(multimodal_features.head(10))

# For training data preparation
for batch in multimodal_features.iter(batch_size=512, shuffle=True):
    # Directly used for PyTorch DataLoader or other ML frameworks
    train_step(batch)
```

## Integration Architecture Design

For machine learning, the green parts are the basic data-related stack. The yellow parts are capabilities that ClickHouse can provide, and the gray parts are key capabilities we need to implement and complement.

![ml-stack.png](attachment:07fb3c3d-da47-448b-9689-d23900b15579:ml-stack.png)

## Competitive Comparison

| Feature Dimension | **chDB** | **Pandas** | **LanceDB** | **DuckDB** | **Polars** | **PySpark** |
| --- | --- | --- | --- | --- | --- | --- |
| **API Style** | Pythonic + SQL hybridSupports chainingLazy execution | Pandas styleChaining callsImmediate execution | Python APIVector database queriesLimited SQL support | SQL-firstPython API secondaryImmediate execution | Pandas-likeLazy loadingExpression API | Spark API + SQLRDD/DataFrameLazy computation |
| **SQL Support** | ✅ Full SQL supportBased on ClickHouse SQL | ❌ No native SQLRequires other tools | ⚠️ Limited SQL supportMainly vector queries | ✅ Full SQL supportOLAP optimized | ✅ SQL expressionsSQL string queries | ✅ Full SQL supportSpark SQL engine |
| **Distributed Support** | ✅ SupportedBased on ClickHouse clusterMulti-source JOIN | ❌ Single machineRequires Dask extension | ⚠️ Limited distributedMainly vector storage | ⚠️ PlannedCurrently mainly single machine | ⚠️ PlannedCurrently mainly single machine | 🌟 **Native distributed**Cluster computingElastic scaling |
| **Performance** | 🚀 Extremely highClickHouse engineColumnar storageVectorized execution | ⚠️ MediumMainly single-threadedMemory limitations | 🚀 Extremely fast vector queriesHNSW indexingGPU acceleration support | 🚀 High performanceColumnar storageVectorized execution | 🚀 High performanceRust implementationParallel processing | ✅ Distributed high performanceMemory+disk computingJVM overhead |
| **Multi-modal Support** | ✅ Planned supportImage/audio processingVector embeddings | ❌ Not supportedRequires additional libraries | ✅ Native supportVector embeddingsMulti-modal indexing | ❌ Not supportedFocus on structured data | ❌ Not supportedFocus on structured data | ✅ MLlib supportImage processingML pipelines |
| **Data Source Types** | 🌟 **Most comprehensive**• Local files (CSV, Parquet, JSON, etc.)• Remote services (S3, HTTP, FTP, etc.)• Databases (MySQL, PostgreSQL, etc.)• Data Lake (Iceberg, etc.) | ⚠️ Limited• Local files• Partial remote data• Requires additional drivers | ⚠️ Specialized• Vector data• Embedding data• Limited file formats | ✅ Rich• Multiple file formats• Remote data sources• Database connections | ✅ Rich• Multiple file formats• Streaming data• Cloud storage | 🌟 **Big data ecosystem**• HDFS/S3, etc.• Various databases• Streaming data• Enterprise data sources |
| **Memory Efficiency** | ✅ ExcellentZero-copy operationsColumnar compression | ❌ AverageHigh memory usage | ✅ ExcellentVector compressionIndex optimization | ✅ ExcellentColumnar storageCompression algorithms | ✅ ExcellentMemory mappingLazy computation | ⚠️ HeavyJVM overheadCluster resource management |
| **Ecosystem Compatibility** | ✅ Pandas compatiblePyTorch integrationJupyter support | 🌟 **Most mature**Huge ecosystemWidest support | ⚠️ EmergingAI/ML specializedVector ecosystem | ✅ GoodPython integrationR support | ✅ GoodMulti-language supportRapid growth | 🌟 **Big data standard**Hadoop ecosystemWide enterprise adoption |
| **Learning Curve** | ✅ Multiple API styles | ✅ Easy to learnWidely adoptedRich documentation | ⚠️ SpecializedRequires vector knowledge | ✅ SQL user friendlyStandard SQL syntax | ⚠️ MediumNew expression syntax | ❌ SteepDistributed conceptsComplex cluster configuration |
| **Deployment & Debugging** | 🌟 **Most convenient**pip install readyDirect local file queriesJupyter friendly | ✅ Extremely simplepip installReady to useRich debugging tools | ✅ SimplePython package installVector database configCloud deployment support | ✅ Zero configurationSingle file databaseConvenient local developmentSimple SQL debugging | ✅ SimpleRust binaryQuick installationClear error messages | ❌ ComplexCluster deploymentResource configurationDifficult distributed debugging |

## Roadmap

Phase 1: Core Functionality

1. Basic DataStore abstraction and local file support
2. Basic chaining operations (filter, select, join)
3. Pandas-compatible data type system (read/write)

Phase 2: High-Level Functionality

1. PyTorch DataLoader integration (marketing)
2. Multi-Modal data processing (marketing)

Phase 3: ClickHouse Server Integration

1. Support ClickHouse Server as execution engine
2. Hybrid Execution support