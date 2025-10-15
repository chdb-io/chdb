#!python3

import io
import json
import unittest
import pyarrow as pa
from pyarrow import csv
import pyarrow.json
import pyarrow.parquet
import chdb


EXPECTED = """"auxten",9
"jerry",7
"tom",5
"""

SCORES_CSV = """score,result,dateOfBirth
758270,lose,1983-07-24
355079,win,2000-11-27
451231,lose,1980-03-11
854953,lose,1996-08-10
294257,lose,1966-12-12
756327,lose,1997-08-29
379755,lose,1981-10-24
916108,lose,1950-08-30
467033,win,2007-09-15
639860,win,1989-06-30
"""

ARROW_DATA_JSONL = """{"match_id": 3943077, "match_date": "2024-07-15", "kick_off": "04:15:00.000", "competition": {"competition_id": 223, "country_name": "South America", "competition_name": "Copa America"}, "season": {"season_id": 282, "season_name": "2024"}, "home_team": {"home_team_id": 779, "home_team_name": "Argentina", "home_team_gender": "male", "home_team_group": null, "country": {"id": 11, "name": "Argentina"}, "managers": [{"id": 5677, "name": "Lionel Sebasti\u00e1n Scaloni", "nickname": null, "dob": "1978-05-16", "country": {"id": 11, "name": "Argentina"}}]}, "away_team": {"away_team_id": 769, "away_team_name": "Colombia", "away_team_gender": "male", "away_team_group": null, "country": {"id": 49, "name": "Colombia"}, "managers": [{"id": 5905, "name": "N\u00e9stor Gabriel Lorenzo", "nickname": null, "dob": "1966-02-28", "country": {"id": 11, "name": "Argentina"}}]}, "home_score": 1, "away_score": 0, "match_status": "available", "match_status_360": "unscheduled", "last_updated": "2024-07-15T15:50:08.671355", "last_updated_360": null, "metadata": {"data_version": "1.1.0", "shot_fidelity_version": "2", "xy_fidelity_version": "2"}, "match_week": 6, "competition_stage": {"id": 26, "name": "Final"}, "stadium": {"id": 5337, "name": "Hard Rock Stadium", "country": {"id": 241, "name": "United States of America"}}, "referee": {"id": 2638, "name": "Raphael Claus", "country": {"id": 31, "name": "Brazil"}}}
{"match_id": 3943076, "match_date": "2024-07-14", "kick_off": "03:00:00.000", "competition": {"competition_id": 223, "country_name": "South America", "competition_name": "Copa America"}, "season": {"season_id": 282, "season_name": "2024"}, "home_team": {"home_team_id": 1833, "home_team_name": "Canada", "home_team_gender": "male", "home_team_group": null, "country": {"id": 40, "name": "Canada"}, "managers": [{"id": 165, "name": "Jesse Marsch", "nickname": null, "dob": "1973-11-08", "country": {"id": 241, "name": "United States of America"}}]}, "away_team": {"away_team_id": 783, "away_team_name": "Uruguay", "away_team_gender": "male", "away_team_group": null, "country": {"id": 242, "name": "Uruguay"}, "managers": [{"id": 269, "name": "Marcelo Alberto Bielsa Caldera", "nickname": "Marcelo Bielsa", "dob": "1955-07-21", "country": {"id": 11, "name": "Argentina"}}]}, "home_score": 2, "away_score": 2, "match_status": "available", "match_status_360": "unscheduled", "last_updated": "2024-07-15T07:57:02.660641", "last_updated_360": null, "metadata": {"data_version": "1.1.0", "shot_fidelity_version": "2", "xy_fidelity_version": "2"}, "match_week": 6, "competition_stage": {"id": 25, "name": "3rd Place Final"}, "stadium": {"id": 52985, "name": "Bank of America Stadium", "country": {"id": 241, "name": "United States of America"}}, "referee": {"id": 1849, "name": "Alexis Herrera", "country": {"id": 246, "name": "Venezuela\u00a0(Bolivarian Republic)"}}}
"""


class TestQueryPyArrow(unittest.TestCase):
    def test_query_arrow1(self):
        table = pa.table(
            {
                "a": pa.array([1, 2, 3, 4, 5, 6]),
                "b": pa.array(["tom", "jerry", "auxten", "tom", "jerry", "auxten"]),
            }
        )

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(table) GROUP BY b ORDER BY b"
        )
        self.assertEqual(str(ret), EXPECTED)

    def test_query_arrow2(self):
        t2 = pa.table(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": ["tom", "jerry", "auxten", "tom", "jerry", "auxten"],
            }
        )

        ret = chdb.query(
            "SELECT b, sum(a) FROM Python(t2) GROUP BY b ORDER BY b"
        )
        self.assertEqual(str(ret), EXPECTED)

    def test_query_arrow3(self):
        table = csv.read_csv(io.BytesIO(SCORES_CSV.encode()))
        ret = chdb.query(
            """
        SELECT sum(score), avg(score), median(score),
               avgIf(score, dateOfBirth > '1980-01-01') as avgIf,
               countIf(result = 'win') AS wins,
               countIf(result = 'draw') AS draws,
               countIf(result = 'lose') AS losses,
               count()
        FROM Python(table)
        """,
        )
        self.assertEqual(
            str(ret),
            "5872873,587287.3,553446.5,470878.25,3,0,7,10\n",
        )

    def test_query_arrow4(self):
        arrow_table = pa.json.read_json(io.BytesIO(ARROW_DATA_JSONL.encode()))
        # print(arrow_table.schema)
        ret = chdb.query("SELECT * FROM Python(arrow_table) LIMIT 10", "JSONEachRow")
        # print(ret)
        self.assertEqual("", ret.error_message())

    def test_query_arrow5(self):
        arrow_table = pa.parquet.read_table(
            "data/sample_2021-04-01_performance_mobile_tiles.parquet"
        )
        # print("Arrow Schema:\n", arrow_table.schema)
        ret = chdb.query("SELECT * FROM Python(arrow_table) LIMIT 1", "JSONCompact")
        # print("JSON:\n", ret)
        schema = json.loads(str(ret)).get("meta")
        # shema is array like:
        # [{"name":"quadkey","type":"String"},{"name":"tile","type":"String"}]
        schema_dict = {x["name"]: x["type"] for x in schema}
        self.assertDictEqual(
            schema_dict,
            {
                "quadkey": "String",
                "tile": "String",
                "tile_x": "Float64",
                "tile_y": "Float64",
                "avg_d_kbps": "Int64",
                "avg_u_kbps": "Int64",
                "avg_lat_ms": "Int64",
                "avg_lat_down_ms": "Float64",
                "avg_lat_up_ms": "Float64",
                "tests": "Int64",
                "devices": "Int64",
            },
        )
        ret = chdb.query(
            """
            WITH numericColumns AS (
            SELECT * EXCEPT ('tile.*') EXCEPT(quadkey)
            FROM Python(arrow_table)
            )
            SELECT * APPLY(max), * APPLY(median) APPLY(x -> round(x, 2))
            FROM numericColumns
            """,
            "JSONCompact",
        )
        # print("JSONCompact:\n", ret)
        self.assertDictEqual(
            {x["name"]: x["type"] for x in json.loads(str(ret)).get("meta")},
            {
                "max(avg_d_kbps)": "Int64",
                "max(avg_lat_down_ms)": "Float64",
                "max(avg_lat_ms)": "Int64",
                "max(avg_lat_up_ms)": "Float64",
                "max(avg_u_kbps)": "Int64",
                "max(devices)": "Int64",
                "max(tests)": "Int64",
                "round(median(avg_d_kbps), 2)": "Float64",
                "round(median(avg_lat_down_ms), 2)": "Float64",
                "round(median(avg_lat_ms), 2)": "Float64",
                "round(median(avg_lat_up_ms), 2)": "Float64",
                "round(median(avg_u_kbps), 2)": "Float64",
                "round(median(devices), 2)": "Float64",
                "round(median(tests), 2)": "Float64",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
