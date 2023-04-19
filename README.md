
<div align="center">
  <img src="docs/_static/snake-chdb.png" height="100">
</div>

[![Build](https://github.com/auxten/chdb/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/auxten/chdb/actions/workflows/build_wheels.yml)
[![PyPI](https://img.shields.io/pypi/v/chdb.svg)](https://pypi.org/project/chdb/)
[![Monthly Downloads](https://pepy.tech/badge/chdb/month)](https://pepy.tech/project/chdb)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/auxten)

# chDB

> chDB is an in-process SQL OLAP Engine powered by ClickHouse


## Features
     
* In-process SQL OLAP Engine, powered by ClickHouse
* No need to install ClickHouse
* Minimized data copy from C++ to Python with [python memoryview](https://docs.python.org/3/c-api/memoryview.html)
* Input&Output support Parquet, CSV, JSON, Arrow, ORC and [more](https://clickhouse.com/docs/en/interfaces/formats)

## Arch
<div align="center">
  <img src="docs/_static/arch-chdb.png" width="700">
</div>

## Installation
Currently, chDB only supports Python 3.7+ on macOS(x86_64 and ARM64) and Linux.
```bash
pip install chdb
```

## Usage

Currently, chDB only supports `query` function, which is used to execute SQL and return desired format data.

```python
import chdb
res = chdb.query('select version()', 'CSV'); print(str(res.get_memview().tobytes()))
```

### work with Parquet or CSV
```python
chdb.query('select * from file("data.parquet", Parquet)', 'CSV')
chdb.query('select * from file("data.csv", CSV)', 'CSV')
```

### pandas dataframe output
```python
chdb.query('select * from file("data.parquet", Parquet)', 'Dataframe')
```

more examples, please refer to [examples](examples)

## Demos

- [Serverless Query Demo](https://chdb.fly.dev/?user=default#Ly8gaHR0cHM6Ly9naXRodWIuY29tL21ldHJpY28vY2hkYi1zZXJ2ZXIKU0VMRUNUCiAgICB0b3duLAogICAgZGlzdHJpY3QsCiAgICBjb3VudCgpIEFTIGMsCiAgICByb3VuZChhdmcocHJpY2UpKSBBUyBwcmljZQpGUk9NIHVybCgnaHR0cHM6Ly9kYXRhc2V0cy1kb2N1bWVudGF0aW9uLnMzLmV1LXdlc3QtMy5hbWF6b25hd3MuY29tL2hvdXNlX3BhcnF1ZXQvaG91c2VfMC5wYXJxdWV0JykKR1JPVVAgQlkKICAgIHRvd24sCiAgICBkaXN0cmljdApMSU1JVCAxMA==)


## Documentation
- For SQL syntax, please refer to [ClickHouse SQL Reference](https://clickhouse.com/docs/en/sql-reference/syntax)

## Contributing
Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
There are something you can help:
- Report bugs on [GitHub Issues](https://github.com/auxten/chdb/issues).
- [ ] Help me with Windows support, I don't know much about Windows toolchain.
- [x] The Python Wrapper just have a `query` function. I want to add more functions to make it more convenient to use. like `toPandas`, `toNumpy` and so on.

## License
AGPL-v3.0 or Commercial License, see [LICENSE](LICENSE.txt) for more information.

## Acknowledgments
chDB is mainly based on [ClickHouse](https://github.com/ClickHouse/ClickHouse)
for trade mark and other reasons, I named it chDB.

## Contact
- Discord: https://discord.gg/nqSkfJRR
- Email: auxtenwpc@gmail.com
- Twitter: [@auxten](https://twitter.com/auxten)