#!python3

import chdb.dataframe as cdf
import pandas as pd
tbl = cdf.Table(dataframe=pd.DataFrame({'a': [1, 2, 3], 'b': ['a', 'b', 'c']}))
ret_tbl = tbl.query('select * from __table__')
print(ret_tbl)
print(ret_tbl.query('select b, sum(a) from __table__ group by b'))
