import sys
import argparse
from .__init__ import query

def main():
    prog = 'python -m chdb'
    description = ('''A simple command line interface for chdb
                   to run SQL and output in specified format''')
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument('sql', nargs=1,
                        type=str,
                        help='sql, e.g: select 1112222222,555')
    parser.add_argument('format', nargs='?',
                        type=str,
                        help='''sql result output format,
                        e.g: CSV, Dataframe, JSON etc,
                        more format checkout on
                        https://clickhouse.com/docs/en/interfaces/formats''',
                        default="CSV")
    options = parser.parse_args()
    sql = options.sql[0]
    output_format = options.format
    res = query(sql, output_format)
    if output_format.lower() in ('dataframe', 'arrowtable'):
        temp = res
    else:
        temp = res.data()
    print(temp, end="")

if __name__ == '__main__':
    main()
