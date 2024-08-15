#!python3

import sys
import time
import timeit
import chdb
import argparse
import pandas as pd

ch_local = "/auxten/chdb/tests/ch24.5/usr/bin/clickhouse"
data_path = "/auxten/bench/hits_0.parquet"

queries = [
    """SELECT COUNT(*) FROM hits;""",
    """SELECT COUNT(*) FROM hits WHERE AdvEngineID <> 0;""",
    """SELECT SUM(AdvEngineID), COUNT(*), AVG(ResolutionWidth) FROM hits;""",
    """SELECT AVG(UserID) FROM hits;""",
    """SELECT COUNT(DISTINCT UserID) FROM hits;""",
    """SELECT COUNT(DISTINCT SearchPhrase) FROM hits;""",
    """SELECT MIN(EventDate), MAX(EventDate) FROM hits;""",
    """SELECT AdvEngineID, COUNT(*) FROM hits WHERE AdvEngineID <> 0 GROUP BY AdvEngineID ORDER BY COUNT(*) DESC;""",
    """SELECT RegionID, COUNT(DISTINCT UserID) AS u FROM hits GROUP BY RegionID ORDER BY u DESC LIMIT 10;""",
    """SELECT RegionID, SUM(AdvEngineID), COUNT(*) AS c, AVG(ResolutionWidth), COUNT(DISTINCT UserID) FROM hits GROUP BY RegionID ORDER BY c DESC LIMIT 10;""",
    """SELECT MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhoneModel ORDER BY u DESC LIMIT 10;""",
    """SELECT MobilePhone, MobilePhoneModel, COUNT(DISTINCT UserID) AS u FROM hits WHERE MobilePhoneModel <> '' GROUP BY MobilePhone, MobilePhoneModel ORDER BY u DESC LIMIT 10;""",
    """SELECT SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;""",
    """SELECT SearchPhrase, COUNT(DISTINCT UserID) AS u FROM hits WHERE SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY u DESC LIMIT 10;""",
    """SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10;""",
    """SELECT UserID, COUNT(*) FROM hits GROUP BY UserID ORDER BY COUNT(*) DESC LIMIT 10;""",
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;""",
    """SELECT UserID, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, SearchPhrase LIMIT 10;""",
    """SELECT UserID, extract(minute FROM toDateTime(EventTime)) AS m, SearchPhrase, COUNT(*) FROM hits GROUP BY UserID, m, SearchPhrase ORDER BY COUNT(*) DESC LIMIT 10;""",
    """SELECT UserID FROM hits WHERE UserID = 435090932899640449;""",
    """SELECT COUNT(*) FROM hits WHERE URL LIKE '%google%';""",
    """SELECT SearchPhrase, MIN(URL), COUNT(*) AS c FROM hits WHERE URL LIKE '%google%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;""",
    """SELECT SearchPhrase, MIN(URL), MIN(Title), COUNT(*) AS c, COUNT(DISTINCT UserID) FROM hits WHERE Title LIKE '%Google%' AND URL NOT LIKE '%.google.%' AND SearchPhrase <> '' GROUP BY SearchPhrase ORDER BY c DESC LIMIT 10;""",
    """SELECT * FROM hits WHERE URL LIKE '%google%' ORDER BY EventTime LIMIT 10;""",
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime LIMIT 10;""",
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY SearchPhrase LIMIT 10;""",
    """SELECT SearchPhrase FROM hits WHERE SearchPhrase <> '' ORDER BY EventTime, SearchPhrase LIMIT 10;""",
    """SELECT CounterID, AVG(length(URL)) AS l, COUNT(*) AS c FROM hits WHERE URL <> '' GROUP BY CounterID HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;""",
    """SELECT REGEXP_REPLACE(Referer, '^https?://(?:www\.)?([^/]+)/.*$', '\1') AS k, AVG(length(Referer)) AS l, COUNT(*) AS c, MIN(Referer) FROM hits WHERE Referer <> '' GROUP BY k HAVING COUNT(*) > 100000 ORDER BY l DESC LIMIT 25;""",
    """SELECT SUM(ResolutionWidth), SUM(ResolutionWidth + 1), SUM(ResolutionWidth + 2), SUM(ResolutionWidth + 3), SUM(ResolutionWidth + 4), SUM(ResolutionWidth + 5), SUM(ResolutionWidth + 6), SUM(ResolutionWidth + 7), SUM(ResolutionWidth + 8), SUM(ResolutionWidth + 9), SUM(ResolutionWidth + 10), SUM(ResolutionWidth + 11), SUM(ResolutionWidth + 12), SUM(ResolutionWidth + 13), SUM(ResolutionWidth + 14), SUM(ResolutionWidth + 15), SUM(ResolutionWidth + 16), SUM(ResolutionWidth + 17), SUM(ResolutionWidth + 18), SUM(ResolutionWidth + 19), SUM(ResolutionWidth + 20), SUM(ResolutionWidth + 21), SUM(ResolutionWidth + 22), SUM(ResolutionWidth + 23), SUM(ResolutionWidth + 24), SUM(ResolutionWidth + 25), SUM(ResolutionWidth + 26), SUM(ResolutionWidth + 27), SUM(ResolutionWidth + 28), SUM(ResolutionWidth + 29), SUM(ResolutionWidth + 30), SUM(ResolutionWidth + 31), SUM(ResolutionWidth + 32), SUM(ResolutionWidth + 33), SUM(ResolutionWidth + 34), SUM(ResolutionWidth + 35), SUM(ResolutionWidth + 36), SUM(ResolutionWidth + 37), SUM(ResolutionWidth + 38), SUM(ResolutionWidth + 39), SUM(ResolutionWidth + 40), SUM(ResolutionWidth + 41), SUM(ResolutionWidth + 42), SUM(ResolutionWidth + 43), SUM(ResolutionWidth + 44), SUM(ResolutionWidth + 45), SUM(ResolutionWidth + 46), SUM(ResolutionWidth + 47), SUM(ResolutionWidth + 48), SUM(ResolutionWidth + 49), SUM(ResolutionWidth + 50), SUM(ResolutionWidth + 51), SUM(ResolutionWidth + 52), SUM(ResolutionWidth + 53), SUM(ResolutionWidth + 54), SUM(ResolutionWidth + 55), SUM(ResolutionWidth + 56), SUM(ResolutionWidth + 57), SUM(ResolutionWidth + 58), SUM(ResolutionWidth + 59), SUM(ResolutionWidth + 60), SUM(ResolutionWidth + 61), SUM(ResolutionWidth + 62), SUM(ResolutionWidth + 63), SUM(ResolutionWidth + 64), SUM(ResolutionWidth + 65), SUM(ResolutionWidth + 66), SUM(ResolutionWidth + 67), SUM(ResolutionWidth + 68), SUM(ResolutionWidth + 69), SUM(ResolutionWidth + 70), SUM(ResolutionWidth + 71), SUM(ResolutionWidth + 72), SUM(ResolutionWidth + 73), SUM(ResolutionWidth + 74), SUM(ResolutionWidth + 75), SUM(ResolutionWidth + 76), SUM(ResolutionWidth + 77), SUM(ResolutionWidth + 78), SUM(ResolutionWidth + 79), SUM(ResolutionWidth + 80), SUM(ResolutionWidth + 81), SUM(ResolutionWidth + 82), SUM(ResolutionWidth + 83), SUM(ResolutionWidth + 84), SUM(ResolutionWidth + 85), SUM(ResolutionWidth + 86), SUM(ResolutionWidth + 87), SUM(ResolutionWidth + 88), SUM(ResolutionWidth + 89) FROM hits;""",
    """SELECT SearchEngineID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY SearchEngineID, ClientIP ORDER BY c DESC LIMIT 10;""",
    """SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits WHERE SearchPhrase <> '' GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;""",
    """SELECT WatchID, ClientIP, COUNT(*) AS c, SUM(IsRefresh), AVG(ResolutionWidth) FROM hits GROUP BY WatchID, ClientIP ORDER BY c DESC LIMIT 10;""",
    """SELECT URL, COUNT(*) AS c FROM hits GROUP BY URL ORDER BY c DESC LIMIT 10;""",
    """SELECT 1, URL, COUNT(*) AS c FROM hits GROUP BY 1, URL ORDER BY c DESC LIMIT 10;""",
    """SELECT ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3, COUNT(*) AS c FROM hits GROUP BY ClientIP, ClientIP - 1, ClientIP - 2, ClientIP - 3 ORDER BY c DESC LIMIT 10;""",
    """SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND URL <> '' GROUP BY URL ORDER BY PageViews DESC LIMIT 10;""",
    """SELECT Title, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND DontCountHits = 0 AND IsRefresh = 0 AND Title <> '' GROUP BY Title ORDER BY PageViews DESC LIMIT 10;""",
    """SELECT URL, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND IsRefresh = 0 AND IsLink <> 0 AND IsDownload = 0 GROUP BY URL ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;""",
    """SELECT TraficSourceID, SearchEngineID, AdvEngineID, CASE WHEN (SearchEngineID = 0 AND AdvEngineID = 0) THEN Referer ELSE '' END AS Src, URL AS Dst, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND IsRefresh = 0 GROUP BY TraficSourceID, SearchEngineID, AdvEngineID, Src, Dst ORDER BY PageViews DESC LIMIT 10 OFFSET 1000;""",
    """SELECT URLHash, EventDate, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND IsRefresh = 0 AND TraficSourceID IN (-1, 6) AND RefererHash = 3594120000172545465 GROUP BY URLHash, EventDate ORDER BY PageViews DESC LIMIT 10 OFFSET 100;""",
    """SELECT WindowClientWidth, WindowClientHeight, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-01'  AND toDate(EventDate) <= '2013-07-31' AND IsRefresh = 0 AND DontCountHits = 0 AND URLHash = 2868770270353813622 GROUP BY WindowClientWidth, WindowClientHeight ORDER BY PageViews DESC LIMIT 10 OFFSET 10000;""",
    """SELECT DATE_TRUNC('minute', toDateTime(EventTime)) AS M, COUNT(*) AS PageViews FROM hits WHERE CounterID = 62  AND toDate(EventDate) >= '2013-07-14'  AND toDate(EventDate) <= '2013-07-15' AND IsRefresh = 0 AND DontCountHits = 0 GROUP BY DATE_TRUNC('minute', toDateTime(EventTime)) ORDER BY DATE_TRUNC('minute', toDateTime(EventTime)) LIMIT 10 OFFSET 1000;""",
]


def chdb_query(i, output, times=1):
    sql = queries[i]
    sql = sql.replace(
        "FROM hits",
        f"FROM file('{data_path}', Parquet)",
    )
    return execute_query(i, output, times, sql)


def execute_query(i, output, times, sql):
    print(f"Q{i}: {sql}")
    time_list = []
    elapsed_list = []
    for t in range(times):
        start = timeit.default_timer()
        ret = chdb.query(
            sql,
            output,
        )
        end = timeit.default_timer()
        time_list.append(round(end - start, 2))
        elapsed_list.append(round(ret.elapsed(), 2))
        print(f"Times: {t}")
    print("FuncTime: ", time_list)
    print("Elapsed : ", elapsed_list)
    return (time_list, elapsed_list)


hits = None


def chdb_query_pandas(i, output, times=1):
    global hits
    if hits is None:
        hits = pd.read_parquet(data_path)
        # fix some types
        hits["EventTime"] = pd.to_datetime(hits["EventTime"], unit="s")
        hits["EventDate"] = pd.to_datetime(hits["EventDate"], unit="D")
        # print(hits["EventDate"][0:10])
        # fix all object columns to string
        for col in hits.columns:
            if hits[col].dtype == "O":
                # hits[col] = hits[col].astype('string')
                hits[col] = hits[col].astype(str)
        # print(hits.dtypes)
    sql = queries[i]
    sql = sql.replace("FROM hits", f"FROM Python(hits)")
    return execute_query(i, output, times, sql)


def exec_ch_local(i, log_level="test", output="Null", times=1):
    f"""
    execute clickhouse local binary like
    /auxten/chdb/tests/ch24.5/usr/bin/clickhouse -q "SELECT COUNT(*) FROM  file("{data_path}") WHERE URL LIKE '%google%'" --log-level=trace
    """
    sql = queries[i]
    sql = sql.replace("FROM hits", f"FROM file('{data_path}', Parquet)")
    import subprocess

    cmd = [
        ch_local,
        "-q",
        sql,
        "--log-level=" + log_level,
        "--time",
        "--output-format=" + output,
    ]
    print(" ".join(cmd))
    time_list = []
    for t in range(times):
        start = timeit.default_timer()
        subprocess.run(cmd)
        end = timeit.default_timer()
        time_list.append(round(end - start, 2))
        print(f"Times: {t}")
    print("ExecTime: ", time_list)
    return time_list


chdb_time_list = None
chdb_elapsed_list = None
chdb_pandas_time_list = None
chdb_pandas_elapsed_list = None
exec_time_list = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=int, help="query index")
    parser.add_argument("output", type=str, help="output format")
    parser.add_argument("--all", action="store_true", help="run all queries")
    parser.add_argument("--times", type=int, default=1, help="run times for each query")
    parser.add_argument("--chdb", action="store_true", help="use chdb to run query")
    parser.add_argument("--pandas", action="store_true", help="use pandas to run query")
    parser.add_argument(
        "--local", action="store_true", help="use local clickhouse binary"
    )
    parser.add_argument(
        "--log_level", type=str, default="test", help="log level for local"
    )
    args = parser.parse_args()
    if args.output == "Null":
        args.log_level = "error"
    if args.all:
        all_time_list = []
        for i in range(len(queries)):
            args.output = "Null"
            args.log_level = "error"
            args.query = i
            tmp = []
            if args.chdb:
                chdb_time_list, chdb_elapsed_list = chdb_query(
                    args.query, args.output, args.times
                )
                tmp.append(chdb_time_list)
                tmp.append(chdb_elapsed_list)
            if args.pandas:
                chdb_pandas_time_list, chdb_pandas_elapsed_list = chdb_query_pandas(
                    args.query, args.output, args.times
                )
                tmp.append(chdb_pandas_time_list)
                tmp.append(chdb_pandas_elapsed_list)
            if args.local:
                exec_time_list = exec_ch_local(
                    args.query, args.log_level, args.output, args.times
                )
                tmp.append(exec_time_list)
            all_time_list.append(tmp)
        # convert to pandas with columns like chdb_time_list, chdb_elapsed_list
        df = pd.DataFrame(all_time_list)
        columns = []
        if args.chdb:
            columns += ["chdb_time", "chdb_elapsed"]
        if args.pandas:
            columns += ["chdb_pd_time", "chdb_pd_elapsed"]
        if args.local:
            columns += ["ch_local_time"]
        df.columns = columns
        print("All queries:")
        print(df)
        sys.exit(0)

    if args.chdb:
        chdb_time_list, chdb_elapsed_list = chdb_query(
            args.query, args.output, args.times
        )
    if args.pandas:
        chdb_pandas_time_list, chdb_pandas_elapsed_list = chdb_query_pandas(
            args.query, args.output, args.times
        )
    if args.local:
        exec_time_list = exec_ch_local(
            args.query, args.log_level, args.output, args.times
        )

    # print summary
    print(f"Q{args.query}: {queries[args.query]}")
    print("Summary:")
    print(f"chdb_time_list:       {chdb_time_list}, elapsed: {chdb_elapsed_list}")
    print(f"chdb_pd_time_list:    {chdb_pandas_time_list}, elapsed: {chdb_pandas_elapsed_list}")
    print(f"local_time_list:      {exec_time_list}")
