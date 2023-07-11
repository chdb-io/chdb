**Tested Business Intelligence(BI) Engine(ChDB) on TPCH_SF10 Queries**

*Description*   
- Tested SQL OLAP(Online Analytical Processing) engine namely chDB using the TPCH_SF10 decision support queries
-	Executed a series of 22 complex SQL queries against large datasets to assess the engine query execution times, resource utilization and overall performance
-	Wrote python script & visualized results using Matplotlib

*Findings*
- Currently, chDB(the embedded version of the clickhouse dbms) only supports Python 3.8+ on macOS(x86_64 and ARM64) and Linux
- ChDB don't need full server installation. Its easier to get started just by importing the required module from python library
- ChDB can access data from a wide variety of formats - both on-disk and in-memory(Parquet, CSV, JSON etc)
- Running SQL queries is very straight forward. ChDB use 'chdb.query' command to run SQL queries
- Currently ChDB don't support creating views
- ChDB fails to execute Query 21 of TPCH_SF10

TPC-H reference: http://tpc.org/tpc_documents_current_versions/pdf/tpc-h_v3.0.0.pdf