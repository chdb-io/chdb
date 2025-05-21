#!python3

import io
import json
import random
import unittest
import numpy as np
import pandas as pd
import pyarrow as pa
import chdb

EXPECTED = '"apple1",3,\\N\n\\N,4,2\n'

class TestQueryJSON(unittest.TestCase):

    def test_query_df(self):
        data = {
            'dict_col1': [
                {'id1': 1, 'name1': 'apple1' },
                {'id2': 2, 'name2': 'apple2' }
            ],
            'dict_col2': [
                {'id': 3, 'name': 'apple3' },
                {'id': 4, 'name': 'apple4' }
            ],
        }

        df_object = pd.DataFrame(data)

        ret = chdb.query("SELECT dict_col1.name1, dict_col2.id, dict_col1.id2  FROM Python(df_object)")

        print(ret)
        self.assertEqual(str(ret), EXPECTED)

if __name__ == "__main__":
    unittest.main(verbosity=3)
