import argparse
from .__init__ import query


def main():
    prog = 'python -m chdb'
    custom_usage = "%(prog)s [-h] \"SELECT 1\" [format]"
    description = ('''A simple command line interface for chdb
                   to run SQL and output in specified format''')
    parser = argparse.ArgumentParser(prog=prog,
                                     usage=custom_usage,
                                     description=description)
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
    try:
        if output_format.lower() in ("dataframe", "arrowtable"):
            temp = res
        else:
            temp = res.data()
        print(temp, end="")
    except UnicodeDecodeError:
        print(repr(res.bytes()))


if __name__ == '__main__':
    main()
