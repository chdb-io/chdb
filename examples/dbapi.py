#!/usr/bin/env python
from chdb import dbapi
from chdb.dbapi.cursors import DictCursor

print("chdb driver version: {0}".format(dbapi.get_client_info()))

conn1 = dbapi.connect()
cur1 = conn1.cursor()
cur1.execute('select version()')
print("description: ", cur1.description)
print("data: ", cur1.fetchone())
cur1.close()
conn1.close()

conn2 = dbapi.connect(cursorclass=DictCursor)
cur2 = conn2.cursor()
cur2.execute('''
SELECT
    town,
    district,
    count() AS c,
    round(avg(price)) AS price
FROM url('https://datasets-documentation.s3.eu-west-3.amazonaws.com/house_parquet/house_0.parquet')
GROUP BY
    town,
    district
LIMIT 10
''')
print("description", cur2.description)
for row in cur2:
    print(row)

cur2.close()
conn2.close()
