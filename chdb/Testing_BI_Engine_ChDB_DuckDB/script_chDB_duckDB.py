###ChDB QUERIES###
sql_chdb=("""

SELECT
    L_RETURNFLAG,
    L_LINESTATUS,
    SUM(L_QUANTITY) AS sum_qty,
    SUM(L_EXTENDEDPRICE) AS sum_base_price,
    SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS sum_disc_price,
    SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT) * (1 + L_TAX)) AS sum_charge,
    AVG(L_QUANTITY) AS avg_qty,
    AVG(L_EXTENDEDPRICE) AS avg_price,
    AVG(L_DISCOUNT) AS avg_disc,
    COUNT(*) AS count_order
FROM
    file('lineitem.parquet')
WHERE
    l_shipdate <= date '1998-09-02'
GROUP BY
    L_RETURNFLAG,
    L_LINESTATUS
ORDER BY
    L_RETURNFLAG,
    L_LINESTATUS;

SELECT
    supplier.S_ACCTBAL,
    supplier.S_NAME,
    nation.N_NAME,
    part.P_PARTKEY,
    part.P_MFGR,
    supplier.S_ADDRESS,
    supplier.S_PHONE,
    supplier.S_COMMENT
FROM
    file('part.parquet') as part
    JOIN file('partsupp.parquet') as partsupp ON part.P_PARTKEY = partsupp.PS_PARTKEY
    JOIN file('supplier.parquet') as supplier ON supplier.S_SUPPKEY = partsupp.PS_SUPPKEY
    JOIN file('nation.parquet') as nation ON nation.N_NATIONKEY = supplier.S_NATIONKEY
    JOIN file('region.parquet') as region ON region.R_REGIONKEY = nation.N_REGIONKEY,
(
        SELECT
            MIN(partsupp0.PS_SUPPLYCOST) as cp_lowest,
            part0.P_PARTKEY as cp_partkey
        FROM
            file('partsupp.parquet') as partsupp0
            JOIN file('part.parquet') as part0 ON part0.P_PARTKEY = partsupp0.PS_PARTKEY
            JOIN file('supplier.parquet') as supplier0 ON supplier0.S_SUPPKEY = partsupp0.PS_SUPPKEY
            JOIN file('nation.parquet') as nation0 ON nation0.N_NATIONKEY = supplier0.S_NATIONKEY
            JOIN file('region.parquet') as region0 ON region0.R_REGIONKEY = nation0.N_REGIONKEY
        WHERE
            region0.R_NAME = 'EUROPE'
        GROUP BY
            cp_partkey
    ) as cheapest_part
WHERE
    part.P_SIZE = 15
    AND part.P_TYPE LIKE '%BRASS'
    AND region.R_NAME = 'EUROPE'
    AND partsupp.PS_SUPPLYCOST = cp_lowest
    AND part.P_PARTKEY = cp_partkey
ORDER BY
    supplier.S_ACCTBAL DESC,
    nation.N_NAME,
    supplier.S_NAME,
    part.P_PARTKEY
LIMIT
    100;

SELECT
    line.L_ORDERKEY,
    SUM(line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)) AS revenue,
    ord.o_orderdate,
    ord.O_SHIPPRIORITY
FROM
    file('customer.parquet') as cust
    JOIN file('orders.parquet') as ord ON cust.C_CUSTKEY = ord.O_CUSTKEY
    JOIN file('lineitem.parquet') as line ON line.L_ORDERKEY = ord.O_ORDERKEY
WHERE
    cust.C_MKTSEGMENT = 'BUILDING'
    AND ord.o_orderdate < date '1995-03-15'
    AND line.l_shipdate > date '1995-03-15'
GROUP BY
    line.L_ORDERKEY,
    ord.o_orderdate,
    ord.O_SHIPPRIORITY
ORDER BY
    revenue DESC,
    ord.o_orderdate
LIMIT
    10;

SELECT
    O_ORDERPRIORITY,
    COUNT(*) AS order_count
FROM
    file('orders.parquet')
WHERE
    o_orderdate >= date '1993-07-01'
    AND o_orderdate < date '1993-10-01'
    AND O_ORDERKEY IN (
        SELECT
            line.L_ORDERKEY
        FROM
            file('lineitem.parquet') as line
        WHERE
            line.l_commitdate < line.l_receiptdate
    )
GROUP BY
    O_ORDERPRIORITY
ORDER BY
    O_ORDERPRIORITY;

SELECT
    nation.N_NAME,
    SUM(line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)) AS revenue
FROM
    file('customer.parquet') as cus
    JOIN file('orders.parquet') as ord ON cus.C_CUSTKEY = ord.O_CUSTKEY
    JOIN file('lineitem.parquet') as line ON line.L_ORDERKEY = ord.O_ORDERKEY
    JOIN file('supplier.parquet') as supp ON line.L_SUPPKEY = supp.S_SUPPKEY
    JOIN file('nation.parquet') as nation ON supp.S_NATIONKEY = nation.N_NATIONKEY
    JOIN file('region.parquet') as region ON nation.N_REGIONKEY = region.R_REGIONKEY
WHERE
    cus.C_NATIONKEY = supp.S_NATIONKEY
    AND region.R_NAME = 'ASIA'
    AND ord.o_orderdate >= date '1994-01-01'
    AND ord.o_orderdate < date '1995-01-01'
GROUP BY
    nation.N_NAME
ORDER BY
    revenue DESC;

SELECT
    SUM(L_EXTENDEDPRICE * L_DISCOUNT) AS revenue
FROM
    file('lineitem.parquet')
WHERE
    l_shipdate >= date '1994-01-01'
    AND l_shipdate < date '1995-01-01'
    AND L_DISCOUNT BETWEEN toDecimal64(0.05, 2)
    AND toDecimal64(0.07, 2)
    AND L_QUANTITY < 24;

SELECT
    supp_nation,
    cust_nation,
    l_year,
    SUM(volume) AS revenue
FROM
    (
        SELECT
            n1.N_NAME AS supp_nation,
            n2.N_NAME AS cust_nation,
            EXTRACT(
                year
                FROM
                    line.l_shipdate
            ) AS l_year,
            line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT) AS volume
        FROM
            file('supplier.parquet') as supp
            JOIN file('lineitem.parquet') as line ON supp.S_SUPPKEY = line.L_SUPPKEY
            JOIN file('orders.parquet') as ord ON ord.O_ORDERKEY = line.L_ORDERKEY
            JOIN file('customer.parquet') as cus ON cus.C_CUSTKEY = ord.O_CUSTKEY
            JOIN file('nation.parquet') as n2 ON cus.C_NATIONKEY = n2.N_NATIONKEY
            JOIN file('nation.parquet') as n1 ON supp.S_NATIONKEY = n1.N_NATIONKEY
        WHERE
            (
                (
                    n1.N_NAME = 'FRANCE'
                    AND n2.N_NAME = 'GERMANY'
                )
                OR (
                    n1.N_NAME = 'GERMANY'
                    AND n2.N_NAME = 'FRANCE'
                )
            )
            AND line.l_shipdate BETWEEN date '1995-01-01'
            AND date '1996-12-31'
    ) AS shipping
GROUP BY
    supp_nation,
    cust_nation,
    l_year
ORDER BY
    supp_nation,
    cust_nation,
    l_year;

SELECT
    o_year,
    SUM(
        CASE
            WHEN nation = 'BRAZIL' THEN volume
            ELSE 0
        END
    ) / SUM(volume) AS mkt_share
FROM
    (
        SELECT
            EXTRACT(
                year
                FROM
                    ord.o_orderdate
            ) AS o_year,
            line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT) AS volume,
            n2.N_NAME AS nation
        FROM
            file('part.parquet') as part
            JOIN file('lineitem.parquet') as line ON part.P_PARTKEY = line.L_PARTKEY
            JOIN file('orders.parquet') as ord ON line.L_ORDERKEY = ord.O_ORDERKEY
            JOIN file('customer.parquet') as cus ON ord.O_CUSTKEY = cus.C_CUSTKEY
            JOIN file('nation.parquet') as n1 ON cus.C_NATIONKEY = n1.N_NATIONKEY
            JOIN file('region.parquet') as reg ON n1.N_REGIONKEY = reg.R_REGIONKEY
            JOIN file('supplier.parquet') as supp ON supp.S_SUPPKEY = line.L_SUPPKEY
            JOIN file('nation.parquet') as n2 ON supp.S_NATIONKEY = n2.N_NATIONKEY
        WHERE
            R_NAME = 'AMERICA'
            AND o_orderdate BETWEEN date '1995-01-01'
            AND date '1996-12-31'
            AND P_TYPE = 'ECONOMY ANODIZED STEEL'
    ) AS all_nations
GROUP BY
    o_year
ORDER BY
    o_year;

SELECT
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM
(
        SELECT
            nat.N_NAME AS nation,
            EXTRACT(
                year
                FROM
                    ord.o_orderdate
            ) AS o_year,
            line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT) - partsupp.PS_SUPPLYCOST * line.L_QUANTITY AS amount
        FROM
            file('partsupp.parquet') as partsupp
            JOIN file('lineitem.parquet') as line ON partsupp.PS_PARTKEY = line.L_PARTKEY
            AND partsupp.PS_SUPPKEY = line.L_SUPPKEY
            JOIN file('part.parquet') as part ON part.P_PARTKEY = line.L_PARTKEY
            JOIN file('supplier.parquet') as supp ON supp.S_SUPPKEY = line.L_SUPPKEY
            JOIN file('orders.parquet') as ord ON ord.O_ORDERKEY = line.L_ORDERKEY
            JOIN file('nation.parquet') as nat ON supp.S_NATIONKEY = nat.N_NATIONKEY
        WHERE
            part.P_NAME LIKE '%green%'
    ) AS profit
GROUP BY
    nation,
    o_year
ORDER BY
    nation,
    o_year DESC;

SELECT
    cus.C_CUSTKEY,
    cus.C_NAME,
    SUM(line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)) AS revenue,
    cus.C_ACCTBAL,
    nat.N_NAME,
    cus.C_ADDRESS,
    cus.C_PHONE,
    cus.C_COMMENT
FROM
    file('lineitem.parquet') as line
    JOIN file('orders.parquet') as ord ON line.L_ORDERKEY = ord.O_ORDERKEY
    JOIN file('customer.parquet') as cus ON cus.C_CUSTKEY = ord.O_CUSTKEY
    JOIN file('nation.parquet') as nat ON cus.C_NATIONKEY = nat.N_NATIONKEY
WHERE
    ord.o_orderdate >= date '1993-10-01'
    AND ord.o_orderdate < date '1994-01-01'
    AND line.L_RETURNFLAG = 'R'
GROUP BY
    cus.C_CUSTKEY,
    cus.C_NAME,
    cus.C_ACCTBAL,
    cus.C_PHONE,
    nat.N_NAME,
    cus.C_ADDRESS,
    cus.C_COMMENT
ORDER BY
    revenue DESC
LIMIT
    20;

SELECT
    partsupp.PS_PARTKEY,
    SUM(partsupp.PS_SUPPLYCOST * partsupp.PS_AVAILQTY) AS value
FROM
    file('partsupp.parquet') as partsupp
    JOIN file('supplier.parquet') as supp ON partsupp.PS_SUPPKEY = supp.S_SUPPKEY
    JOIN file('nation.parquet') as nat ON supp.S_NATIONKEY = nat.N_NATIONKEY
WHERE
    nat.N_NAME = 'GERMANY'
GROUP BY
    partsupp.PS_PARTKEY
HAVING
    SUM(partsupp.PS_SUPPLYCOST * partsupp.PS_AVAILQTY) > (
        SELECT
            SUM(partsupp0.PS_SUPPLYCOST * partsupp0.PS_AVAILQTY) * (0.0001 / 10)
        FROM
            file('partsupp.parquet') as partsupp0
            JOIN file('supplier.parquet') as supp0 ON partsupp0.PS_SUPPKEY = supp0.S_SUPPKEY
            JOIN file('nation.parquet') as nat0 ON supp0.S_NATIONKEY = nat0.N_NATIONKEY
        WHERE
            nat0.N_NAME = 'GERMANY'
    )
ORDER BY
    value DESC;

SELECT
    line.L_SHIPMODE,
    SUM(
        CASE
            WHEN ord.O_ORDERPRIORITY = '1-URGENT'
            OR ord.O_ORDERPRIORITY = '2-HIGH' THEN 1
            ELSE 0
        END
    ) AS high_line_count,
    SUM(
        CASE
            WHEN ord.O_ORDERPRIORITY <> '1-URGENT'
            AND ord.O_ORDERPRIORITY <> '2-HIGH' THEN 1
            ELSE 0
        END
    ) AS low_line_count
FROM
    file('orders.parquet') as ord
    JOIN file('lineitem.parquet') as line ON ord.O_ORDERKEY = line.L_ORDERKEY
WHERE
    line.L_SHIPMODE IN ('MAIL', 'SHIP')
    AND line.l_commitdate < line.l_receiptdate
    AND line.l_shipdate < line.l_commitdate
    AND line.l_receiptdate >= date '1994-01-01'
    AND line.l_receiptdate < date '1995-01-01'
GROUP BY
    line.L_SHIPMODE
ORDER BY
    line.L_SHIPMODE;

SELECT
    c_count,
    COUNT(*) AS custdist
FROM
(
        SELECT
            cus.C_CUSTKEY,
            COUNT(ord.O_ORDERKEY) AS c_count
        FROM
            file('customer.parquet') as cus
            LEFT OUTER JOIN file('orders.parquet') as ord ON cus.C_CUSTKEY = ord.O_CUSTKEY
            AND ord.O_COMMENT NOT LIKE '%special%requests%'
        GROUP BY
            cus.C_CUSTKEY
    ) AS c_orders
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;

SELECT
    toDecimal64(100.00, 2) * SUM(
        CASE
            WHEN part.P_TYPE LIKE 'PROMO%' THEN line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)
            ELSE 0
        END
    ) / SUM(line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)) AS promo_revenue
FROM
    file('lineitem.parquet') as line
    JOIN file('part.parquet') as part ON line.L_PARTKEY = part.P_PARTKEY
WHERE
    line.l_shipdate >= date '1995-09-01'
    AND line.l_shipdate < date '1995-10-01';

SELECT
    supp.S_SUPPKEY,
    supp.S_NAME,
    supp.S_ADDRESS,
    supp.S_PHONE,
    revenue0.total_revenue
FROM
    file('supplier.parquet') as supp,
(
        SELECT
            line1.L_SUPPKEY AS supplier_no,
            SUM(line1.L_EXTENDEDPRICE * (1 - line1.L_DISCOUNT)) AS total_revenue
        FROM
            file('lineitem.parquet') as line1
        WHERE
            line1.l_shipdate >= date '1996-01-01'
            AND line1.l_shipdate < date '1996-04-01'
        GROUP BY
            supplier_no
    ) as revenue0
WHERE
    supp.S_SUPPKEY = revenue0.supplier_no
    AND revenue0.total_revenue = (
        SELECT
            MAX(revenue1.total_revenue)
        FROM
            (
                SELECT
                    line0.L_SUPPKEY AS supplier_no,
                    SUM(line0.L_EXTENDEDPRICE * (1 - line0.L_DISCOUNT)) AS total_revenue
                FROM
                    file('lineitem.parquet') as line0
                WHERE
                    line0.l_shipdate >= date '1996-01-01'
                    AND line0.l_shipdate < date '1996-04-01'
                GROUP BY
                    supplier_no
            ) as revenue1
    )
ORDER BY
    supp.S_SUPPKEY;

SELECT
    part.P_BRAND,
    part.P_TYPE,
    part.P_SIZE,
    COUNT(DISTINCT partsupp.PS_SUPPKEY) AS supplier_cnt
FROM
    file('partsupp.parquet') as partsupp
    JOIN file('part.parquet') as part ON part.P_PARTKEY = partsupp.PS_PARTKEY
WHERE
    part.P_BRAND <> 'Brand#45'
    AND part.P_TYPE NOT LIKE 'MEDIUM POLISHED%'
    AND part.P_SIZE IN (49, 14, 23, 45, 19, 3, 36, 9)
    AND partsupp.PS_SUPPKEY NOT IN (
        SELECT
            supp.S_SUPPKEY
        FROM
            file('supplier.parquet') as supp
        WHERE
            supp.S_COMMENT LIKE '%Customer%Complaints%'
    )
GROUP BY
    part.P_BRAND,
    part.P_TYPE,
    part.P_SIZE
ORDER BY
    supplier_cnt DESC,
    part.P_BRAND,
    part.P_TYPE,
    part.P_SIZE;

SELECT
    SUM(line.L_EXTENDEDPRICE) / toDecimal64(7.0, 2) AS avg_yearly
FROM
    file('lineitem.parquet') as line
    JOIN file('part.parquet') as part ON part.P_PARTKEY = line.L_PARTKEY,
(
        SELECT
            toDecimal64(0.2 * AVG(line0.L_QUANTITY), 12) as limit_qty,
            line0.L_PARTKEY as lpk
        FROM
            file('lineitem.parquet') as line0
        GROUP BY
            lpk
    ) as part_avg
WHERE
    part.P_BRAND = 'Brand#23'
    AND part.P_CONTAINER = 'MED BOX'
    AND part.P_PARTKEY = lpk
    AND line.L_QUANTITY < limit_qty;

SELECT
    cus.C_NAME,
    cus.C_CUSTKEY,
    ord.O_ORDERKEY,
    ord.o_orderdate,
    ord.O_TOTALPRICE,
    SUM(line.L_QUANTITY)
FROM
    file('customer.parquet') as cus
    JOIN file('orders.parquet') as ord ON cus.C_CUSTKEY = ord.O_CUSTKEY
    JOIN file('lineitem.parquet') as line ON ord.O_ORDERKEY = line.L_ORDERKEY
WHERE
    ord.O_ORDERKEY IN (
        SELECT
            line0.L_ORDERKEY
        FROM
            file('lineitem.parquet') as line0
        GROUP BY
            line0.L_ORDERKEY
        HAVING
            SUM(line0.L_QUANTITY) > 300
    )
GROUP BY
    cus.C_NAME,
    cus.C_CUSTKEY,
    ord.O_ORDERKEY,
    ord.o_orderdate,
    ord.O_TOTALPRICE
ORDER BY
    ord.O_TOTALPRICE DESC,
    ord.o_orderdate
LIMIT
    100;

SELECT
    SUM(line.L_EXTENDEDPRICE * (1 - line.L_DISCOUNT)) AS revenue
FROM
    file('lineitem.parquet') as line
    JOIN file('part.parquet') as part ON part.P_PARTKEY = line.L_PARTKEY
WHERE
    (
        part.P_BRAND = 'Brand#12'
        AND part.P_CONTAINER IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
        AND line.L_QUANTITY >= 1
        AND line.L_QUANTITY <= 1 + 10
        AND (
            part.P_SIZE BETWEEN 1
            AND 5
        )
        AND line.L_SHIPMODE IN ('AIR', 'AIR REG')
        AND line.L_SHIPINSTRUCT = 'DELIVER IN PERSON'
    )
    OR (
        part.P_PARTKEY = line.L_PARTKEY
        AND part.P_BRAND = 'Brand#23'
        AND part.P_CONTAINER IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
        AND line.L_QUANTITY >= 10
        AND line.L_QUANTITY <= 10 + 10
        AND (
            part.P_SIZE BETWEEN 1
            AND 10
        )
        AND line.L_SHIPMODE IN ('AIR', 'AIR REG')
        AND line.L_SHIPINSTRUCT = 'DELIVER IN PERSON'
    )
    OR (
        part.P_PARTKEY = line.L_PARTKEY
        AND part.P_BRAND = 'Brand#34'
        AND part.P_CONTAINER IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
        AND line.L_QUANTITY >= 20
        AND line.L_QUANTITY <= 20 + 10
        AND (
            part.P_SIZE BETWEEN 1
            AND 15
        )
        AND line.L_SHIPMODE IN ('AIR', 'AIR REG')
        AND line.L_SHIPINSTRUCT = 'DELIVER IN PERSON'
    );

SELECT
    supp.S_NAME,
    supp.S_ADDRESS
FROM
    file('supplier.parquet') as supp
    JOIN file('nation.parquet') as nat ON supp.S_NATIONKEY = nat.N_NATIONKEY
WHERE
    supp.S_SUPPKEY IN (
        SELECT
            partsupp1.PS_SUPPKEY
        FROM
            file('partsupp.parquet') as partsupp1,
(
                SELECT
                    0.5 * SUM(line0.L_QUANTITY) as ps_halfqty,
                    line0.L_PARTKEY as pkey,
                    line0.L_SUPPKEY as skey
                FROM
                    file('lineitem.parquet') as line0
                WHERE
                    line0.l_shipdate >= date '1994-01-01'
                    AND line0.l_shipdate < date '1995-01-01'
                group by
                    pkey,
                    skey
            ) as availability_part_supp
        WHERE
            partsupp1.PS_PARTKEY IN (
                SELECT
                    part0.P_PARTKEY
                FROM
                    file('part.parquet') as part0
                WHERE
                    part0.P_NAME LIKE 'forest%'
            )
            AND partsupp1.PS_AVAILQTY > availability_part_supp.ps_halfqty
            AND partsupp1.PS_SUPPKEY = availability_part_supp.skey
            AND partsupp1.PS_PARTKEY = availability_part_supp.pkey
    )
    AND nat.N_NAME = 'CANADA'
ORDER BY
    supp.S_NAME;

SELECT
    cntrycode,
    COUNT(*) AS numcust,
    SUM(c_acctbal) AS totacctbal
FROM
    (
        SELECT
            SUBSTRING(cus2.C_PHONE, 1, 2) AS cntrycode,
            cus2.C_ACCTBAL as c_acctbal
        FROM
            file('customer.parquet') as cus2
        WHERE
            SUBSTRING(cus2.C_PHONE, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')
            AND cus2.C_ACCTBAL > (
                SELECT
                    AVG(cus1.C_ACCTBAL)
                FROM
                    file('customer.parquet') as cus1
                WHERE
                    cus1.C_ACCTBAL > 0.00
                    AND SUBSTRING(cus1.C_PHONE, 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')
            )
            AND cus2.C_CUSTKEY NOT IN (
                SELECT
                    ord.O_CUSTKEY
                FROM
                    file('orders.parquet') as ord
            )
    ) AS custsale
GROUP BY
    cntrycode
ORDER BY
    cntrycode;
    
""")

##python script##
#pip install chdb

import time
import chdb
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 20)

def execute_query(sql_script):
    import chdb
    #df = pd.DataFrame(columns=['dur'])
    sql_arr = sql_script.split(";")
    chdb_dict={}
    for index, value in enumerate(sql_arr,start=1):
        val=value.strip()
        if len(val) > 0:
          start = time.time()
          qnum='Q'+str(index)
          print(qnum)       
          try:
              qnum=chdb.query(val,'DataFrame')
              print(qnum)
              stop = time.time()
              duration = stop-start            
          except  Exception as er:
              print(er)
              duration =0
          print(duration)
          #row = {'dur': duration}
          #df = pd.concat([df,pd.DataFrame(row, index=[index])], axis=0, ignore_index=True)
          chdb_dict[index]=duration
    #print(chdb_dict)
    return chdb_dict

mydict_chdb=execute_query(sql_chdb)
#print(mydict_chdb)


##appending values of mydict into mychdblist##
mychdblist=[]
for key, value in mydict_chdb.items():
    mychdblist.append(value)
#print(mychdblist)



###duckdb QUERIES###

#pip install duckdb
import duckdb 
lineitem= duckdb.read_parquet('lineitem.parquet')
orders=   duckdb.read_parquet('orders.parquet')
partsupp= duckdb.read_parquet('partsupp.parquet')
supplier= duckdb.read_parquet('supplier.parquet')
nation=   duckdb.read_parquet('nation.parquet')
region=   duckdb.read_parquet('region.parquet')
customer= duckdb.read_parquet('customer.parquet')
part=     duckdb.read_parquet('part.parquet')
duckdb.sql('PRAGMA disable_progress_bar')

sql=(f'''
SELECT
    --Query01
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;

    
SELECT
    --Query02
    s_acctbal,
    s_name,
    n_name,
    p_partkey,
    p_mfgr,
    s_address,
    s_phone,
    s_comment
FROM
    part,
    supplier,
    partsupp,
    nation,
    region
WHERE
    p_partkey = ps_partkey
    AND s_suppkey = ps_suppkey
    AND p_size = 15
    AND p_type LIKE '%BRASS'
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'EUROPE'
    AND ps_supplycost = (
        SELECT
            MIN(ps_supplycost)
        FROM
            partsupp,
            supplier,
            nation,
            region
        WHERE
            p_partkey = ps_partkey
            AND s_suppkey = ps_suppkey
            AND s_nationkey = n_nationkey
            AND n_regionkey = r_regionkey
            AND r_name = 'EUROPE'
    )
ORDER BY
    s_acctbal DESC,
    n_name,
    s_name,
    p_partkey
LIMIT
    100;



SELECT
    --Query03
    l_orderkey,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    o_orderdate,
    o_shippriority
FROM
    customer,
    orders,
    lineitem
WHERE
    c_mktsegment = 'BUILDING'
    AND c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate < CAST('1995-03-15' AS date)
    AND l_shipdate > CAST('1995-03-15' AS date)
GROUP BY
    l_orderkey,
    o_orderdate,
    o_shippriority
ORDER BY
    revenue DESC,
    o_orderdate
LIMIT
    10;



SELECT
    --Query04
    o_orderpriority,
    COUNT(*) AS order_count
FROM
    orders
WHERE
    o_orderdate >= CAST('1993-07-01' AS date)
    AND o_orderdate < CAST('1993-10-01' AS date)
    AND EXISTS (
        SELECT
            *
        FROM
            lineitem
        WHERE
            l_orderkey = o_orderkey
            AND l_commitdate < l_receiptdate
    )
GROUP BY
    o_orderpriority
ORDER BY
    o_orderpriority;



SELECT
    --Query05
    n_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    customer,
    orders,
    lineitem,
    supplier,
    nation,
    region
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND l_suppkey = s_suppkey
    AND c_nationkey = s_nationkey
    AND s_nationkey = n_nationkey
    AND n_regionkey = r_regionkey
    AND r_name = 'ASIA'
    AND o_orderdate >= CAST('1994-01-01' AS date)
    AND o_orderdate < CAST('1995-01-01' AS date)
GROUP BY
    n_name
ORDER BY
    revenue DESC;



SELECT
    --Query06
    SUM(l_extendedprice * l_discount) AS revenue
FROM
    lineitem
WHERE
    l_shipdate >= CAST('1994-01-01' AS date)
    AND l_shipdate < CAST('1995-01-01' AS date)
    AND l_discount BETWEEN 0.05
    AND 0.07
    AND l_quantity < 24;


SELECT
    --Query07
    supp_nation,
    cust_nation,
    l_year,
    SUM(volume) AS revenue
FROM
    (
        SELECT
            n1.n_name AS supp_nation,
            n2.n_name AS cust_nation,
            EXTRACT(
                year
                FROM
                    l_shipdate
            ) AS l_year,
            l_extendedprice * (1 - l_discount) AS volume
        FROM
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2
        WHERE
            s_suppkey = l_suppkey
            AND o_orderkey = l_orderkey
            AND c_custkey = o_custkey
            AND s_nationkey = n1.n_nationkey
            AND c_nationkey = n2.n_nationkey
            AND (
                (
                    n1.n_name = 'FRANCE'
                    AND n2.n_name = 'GERMANY'
                )
                OR (
                    n1.n_name = 'GERMANY'
                    AND n2.n_name = 'FRANCE'
                )
            )
            AND l_shipdate BETWEEN CAST('1995-01-01' AS date)
            AND CAST('1996-12-31' AS date)
    ) AS shipping
GROUP BY
    supp_nation,
    cust_nation,
    l_year
ORDER BY
    supp_nation,
    cust_nation,
    l_year;



SELECT
    --Query08
    o_year,
    SUM(
        CASE
            WHEN nation = 'BRAZIL' THEN volume
            ELSE 0
        END
    ) / SUM(volume) AS mkt_share
FROM
    (
        SELECT
            EXTRACT(
                year
                FROM
                    o_orderdate
            ) AS o_year,
            l_extendedprice * (1 - l_discount) AS volume,
            n2.n_name AS nation
        FROM
            part,
            supplier,
            lineitem,
            orders,
            customer,
            nation n1,
            nation n2,
            region
        WHERE
            p_partkey = l_partkey
            AND s_suppkey = l_suppkey
            AND l_orderkey = o_orderkey
            AND o_custkey = c_custkey
            AND c_nationkey = n1.n_nationkey
            AND n1.n_regionkey = r_regionkey
            AND r_name = 'AMERICA'
            AND s_nationkey = n2.n_nationkey
            AND o_orderdate BETWEEN CAST('1995-01-01' AS date)
            AND CAST('1996-12-31' AS date)
            AND p_type = 'ECONOMY ANODIZED STEEL'
    ) AS all_nations
GROUP BY
    o_year
ORDER BY
    o_year;



SELECT
    --Query09
    nation,
    o_year,
    SUM(amount) AS sum_profit
FROM
    (
        SELECT
            n_name AS nation,
            EXTRACT(
                year
                FROM
                    o_orderdate
            ) AS o_year,
            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity AS amount
        FROM
            part,
            supplier,
            lineitem,
            partsupp,
            orders,
            nation
        WHERE
            s_suppkey = l_suppkey
            AND ps_suppkey = l_suppkey
            AND ps_partkey = l_partkey
            AND p_partkey = l_partkey
            AND o_orderkey = l_orderkey
            AND s_nationkey = n_nationkey
            AND p_name LIKE '%green%'
    ) AS profit
GROUP BY
    nation,
    o_year
ORDER BY
    nation,
    o_year DESC;



SELECT
    --Query10
    c_custkey,
    c_name,
    SUM(l_extendedprice * (1 - l_discount)) AS revenue,
    c_acctbal,
    n_name,
    c_address,
    c_phone,
    c_comment
FROM
    customer,
    orders,
    lineitem,
    nation
WHERE
    c_custkey = o_custkey
    AND l_orderkey = o_orderkey
    AND o_orderdate >= CAST('1993-10-01' AS date)
    AND o_orderdate < CAST('1994-01-01' AS date)
    AND l_returnflag = 'R'
    AND c_nationkey = n_nationkey
GROUP BY
    c_custkey,
    c_name,
    c_acctbal,
    c_phone,
    n_name,
    c_address,
    c_comment
ORDER BY
    revenue DESC
LIMIT
    20;


SELECT
    --Query11
    ps_partkey,
    SUM(ps_supplycost * ps_availqty) AS value
FROM
    partsupp,
    supplier,
    nation
WHERE
    ps_suppkey = s_suppkey
    AND s_nationkey = n_nationkey
    AND n_name = 'GERMANY'
GROUP BY
    ps_partkey
HAVING
    SUM(ps_supplycost * ps_availqty) > (
        SELECT
            SUM(ps_supplycost * ps_availqty) * (0.0001/10)
            -- SUM(ps_supplycost * ps_availqty) * 1
        FROM
            partsupp,
            supplier,
            nation
        WHERE
            ps_suppkey = s_suppkey
            AND s_nationkey = n_nationkey
            AND n_name = 'GERMANY'
    )
ORDER BY
    value DESC;



SELECT
    --Query12
    l_shipmode,
    SUM(
        CASE
            WHEN o_orderpriority = '1-URGENT'
            OR o_orderpriority = '2-HIGH' THEN 1
            ELSE 0
        END
    ) AS high_line_count,
    SUM(
        CASE
            WHEN o_orderpriority <> '1-URGENT'
            AND o_orderpriority <> '2-HIGH' THEN 1
            ELSE 0
        END
    ) AS low_line_count
FROM
    orders,
    lineitem
WHERE
    o_orderkey = l_orderkey
    AND l_shipmode IN ('MAIL', 'SHIP')
    AND l_commitdate < l_receiptdate
    AND l_shipdate < l_commitdate
    AND l_receiptdate >= CAST('1994-01-01' AS date)
    AND l_receiptdate < CAST('1995-01-01' AS date)
GROUP BY
    l_shipmode
ORDER BY
    l_shipmode;



SELECT
    --Query13
    c_count,
    COUNT(*) AS custdist
FROM
    (
        SELECT
            c_custkey,
            COUNT(o_orderkey) AS c_count
        FROM
            customer
            LEFT OUTER JOIN orders ON c_custkey = o_custkey
            AND o_comment NOT LIKE '%special%requests%'
        GROUP BY
            c_custkey
    ) AS c_orders
GROUP BY
    c_count
ORDER BY
    custdist DESC,
    c_count DESC;




SELECT
    --Query14
    100.00 * SUM(
        CASE
            WHEN p_type LIKE 'PROMO%' THEN l_extendedprice * (1 - l_discount)
            ELSE 0
        END
    ) / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
FROM
    lineitem,
    part
WHERE
    l_partkey = p_partkey
    AND l_shipdate >= date '1995-09-01'
    AND l_shipdate < CAST('1995-10-01' AS date);



SELECT
    --Query15
    s_suppkey,
    s_name,
    s_address,
    s_phone,
    total_revenue
FROM
    supplier,
    (
        SELECT
            l_suppkey AS supplier_no,
            SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
        FROM
            lineitem
        WHERE
            l_shipdate >= CAST('1996-01-01' AS date)
            AND l_shipdate < CAST('1996-04-01' AS date)
        GROUP BY
            supplier_no
    ) revenue0
WHERE
    s_suppkey = supplier_no
    AND total_revenue = (
        SELECT
            MAX(total_revenue)
        FROM
            (
                SELECT
                    l_suppkey AS supplier_no,
                    SUM(l_extendedprice * (1 - l_discount)) AS total_revenue
                FROM
                    lineitem
                WHERE
                    l_shipdate >= CAST('1996-01-01' AS date)
                    AND l_shipdate < CAST('1996-04-01' AS date)
                GROUP BY
                    supplier_no
            ) revenue1
    )
ORDER BY
    s_suppkey;


SELECT
    --Query16
    p_brand,
    p_type,
    p_size,
    COUNT(DISTINCT ps_suppkey) AS supplier_cnt
FROM
    partsupp,
    part
WHERE
    p_partkey = ps_partkey
    AND p_brand <> 'Brand#45'
    AND p_type NOT LIKE 'MEDIUM POLISHED%'
    AND p_size IN (
        49,
        14,
        23,
        45,
        19,
        3,
        36,
        9
    )
    AND ps_suppkey NOT IN (
        SELECT
            s_suppkey
        FROM
            supplier
        WHERE
            s_comment LIKE '%Customer%Complaints%'
    )
GROUP BY
    p_brand,
    p_type,
    p_size
ORDER BY
    supplier_cnt DESC,
    p_brand,
    p_type,
    p_size;



SELECT
    --Query17
    SUM(l_extendedprice) / 7.0 AS avg_yearly
FROM
    lineitem,
    part
WHERE
    p_partkey = l_partkey
    AND p_brand = 'Brand#23'
    AND p_container = 'MED BOX'
    AND l_quantity < (
        SELECT
            0.2 * AVG(l_quantity)
        FROM
            lineitem
        WHERE
            l_partkey = p_partkey
    );
    


SELECT
    --Query18
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice,
    SUM(l_quantity)
FROM
    customer,
    orders,
    lineitem
WHERE
    o_orderkey IN (
        SELECT
            l_orderkey
        FROM
            lineitem
        GROUP BY
            l_orderkey
        HAVING
            SUM(l_quantity) > 300
    )
    AND c_custkey = o_custkey
    AND o_orderkey = l_orderkey
GROUP BY
    c_name,
    c_custkey,
    o_orderkey,
    o_orderdate,
    o_totalprice
ORDER BY
    o_totalprice DESC,
    o_orderdate
LIMIT
    100;
    




SELECT
    --Query19
    SUM(l_extendedprice * (1 - l_discount)) AS revenue
FROM
    lineitem,
    part
WHERE
    (
        p_partkey = l_partkey
        AND p_brand = 'Brand#12'
        AND p_container IN (
            'SM CASE',
            'SM BOX',
            'SM PACK',
            'SM PKG'
        )
        AND l_quantity >= 1
        AND l_quantity <= 1 + 10
        AND p_size BETWEEN 1
        AND 5
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR (
        p_partkey = l_partkey
        AND p_brand = 'Brand#23'
        AND p_container IN (
            'MED BAG',
            'MED BOX',
            'MED PKG',
            'MED PACK'
        )
        AND l_quantity >= 10
        AND l_quantity <= 10 + 10
        AND p_size BETWEEN 1
        AND 10
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    )
    OR (
        p_partkey = l_partkey
        AND p_brand = 'Brand#34'
        AND p_container IN (
            'LG CASE',
            'LG BOX',
            'LG PACK',
            'LG PKG'
        )
        AND l_quantity >= 20
        AND l_quantity <= 20 + 10
        AND p_size BETWEEN 1
        AND 15
        AND l_shipmode IN ('AIR', 'AIR REG')
        AND l_shipinstruct = 'DELIVER IN PERSON'
    );
    


SELECT
    --Query20
    s_name,
    s_address
FROM
    supplier,
    nation
WHERE
    s_suppkey IN (
        SELECT
            ps_suppkey
        FROM
            partsupp
        WHERE
            ps_partkey IN (
                SELECT
                    p_partkey
                FROM
                    part
                WHERE
                    p_name LIKE 'forest%'
            )
            AND ps_availqty > (
                SELECT
                    0.5 * SUM(l_quantity)
                FROM
                    lineitem
                WHERE
                    l_partkey = ps_partkey
                    AND l_suppkey = ps_suppkey
                    AND l_shipdate >= CAST('1994-01-01' AS date)
                    AND l_shipdate < CAST('1995-01-01' AS date)
            )
    )
    AND s_nationkey = n_nationkey
    AND n_name = 'CANADA'
ORDER BY
    s_name;
    


SELECT
    --Query22
    cntrycode,
    COUNT(*) AS numcust,
    SUM(c_acctbal) AS totacctbal
FROM
    (
        SELECT
            SUBSTRING(c_phone, 1, 2) AS cntrycode,
            c_acctbal
        FROM
            customer
        WHERE
            SUBSTRING(c_phone, 1, 2) IN (
                '13',
                '31',
                '23',
                '29',
                '30',
                '18',
                '17'
            )
            AND c_acctbal > (
                SELECT
                    AVG(c_acctbal)
                FROM
                    customer
                WHERE
                    c_acctbal > 0.00
                    AND SUBSTRING(c_phone, 1, 2) IN (
                        '13',
                        '31',
                        '23',
                        '29',
                        '30',
                        '18',
                        '17'
                    )
            )
            AND NOT EXISTS (
                SELECT
                    *
                FROM
                    orders
                WHERE
                    o_custkey = c_custkey
            )
    ) AS custsale
GROUP BY
    cntrycode
ORDER BY
    cntrycode;
    
''')


##python script##
pd.set_option('display.max_columns', 20)
def execute_query(engine, sql_script):
    #df = pd.DataFrame(columns=['dur'])
    sql_arr = sql_script.split(";")
    duckdb_dict={}
    for index, value in enumerate(sql_arr,start=1):
        if len(value.strip()) > 0:
            start = time.time()
            print('Query' + str(index))
            try : 
             engine.sql(value).show()
             stop = time.time()
             duration = stop-start
            except  Exception as er:
              print(er)
              duration =0
            print(duration)
            #row = {'dur': duration}
            #df = pd.concat([df,pd.DataFrame(row, index=[index])], axis=0, ignore_index=True)
            duckdb_dict[index]=duration
    return duckdb_dict

mydict_duck=execute_query(duckdb, sql)

##appending values of mydict_duck into myducklist##
myducklist=[]
for key, value in mydict_duck.items():
    val=round(value,6)
    myducklist.append(val)
#print(myducklist)



####creating dataframe & performing comparison b/w chdb and duckdb
df=pd.DataFrame(columns=['chDB'],data=mychdblist)
#adding new col 'duckdb' with myducklist
df['duckdb'] = myducklist
#changing index to start from 1
df.index += 1 
print(df)

#plotting graph
#fig=plt.figure(figsize=(30,20))
df.plot.bar(rot=0)
plt.xlabel("Queries")
plt.ylabel("Duration in Second. Lower is Better")
plt.title("TPCH-SF10")
plt.legend(loc=0)
#fig.autofmt_xdate()
plt.show()

