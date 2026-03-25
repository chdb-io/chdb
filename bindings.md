## Welcome to the Bindings Contributors

Welcome to the community of bindings contributors! chDB offers a stable C ABI, which facilitates the development of bindings in various languages. For a C language calling demo, please refer to the examples in the `/examples` directory, such as `chdbDlopen.c`, `chdbSimple.c`, and `chdbStub.c`.

### Core Features

chDB exposes four main capabilities through its C API ([`chdb.h`](programs/local/chdb.h)):

| Feature | C API | Description |
|---|---|---|
| **Stateless Query** | `query_stable()` | One-shot query execution; each call bootstraps a new engine context. Simple but incurs startup overhead per query. |
| **Session (Connection)** | `chdb_connect()` / `chdb_query()` | Persistent connection with reusable engine context. Supports multi-statement workflows. |
| **Streaming Query** | `chdb_stream_query()` / `chdb_stream_fetch_result()` | Chunked result iteration with constant memory usage. Ideal for large result sets that should not be fully materialized. |
| **Arrow Scan** | `chdb_arrow_scan()` / `chdb_arrow_array_scan()` | Register Arrow streams or arrays as queryable table functions. Enables zero-copy data exchange with Arrow-native ecosystems. |

### Feature Matrix

| Binding | Stateless Query | Session | Streaming | Arrow Scan | Repository |
|:---|:---:|:---:|:---:|:---:|:---|
| **Python** (chdb) | ✅ | ✅ | ✅ | | [chdb-io/chdb](https://github.com/chdb-io/chdb) |
| **Go** | ✅ | ✅ | ✅ | | [chdb-io/chdb-go](https://github.com/chdb-io/chdb-go) |
| **Rust** | ✅ | ✅ | | ✅ | [chdb-io/chdb-rust](https://github.com/chdb-io/chdb-rust) |
| **Node.js** | ✅ | ✅ | | | [chdb-io/chdb-node](https://github.com/chdb-io/chdb-node) |
| **Ruby** | ✅ | ✅ | ✅ | | [chdb-io/chdb-ruby](https://github.com/chdb-io/chdb-ruby) |
| **Zig** | ✅ | ✅ | ✅ | | [chdb-io/chdb-zig](https://github.com/chdb-io/chdb-zig) |
| **Bun** | ✅ | | | | [chdb-io/chdb-bun](https://github.com/chdb-io/chdb-bun) |
| **.NET** | ✅ | | | | [chdb-io/chdb-dotnet](https://github.com/chdb-io/chdb-dotnet) |

> **Legend:** ✅ Supported  |  Blank = not yet implemented

Bindings not yet available: **Java**, **PHP**, **R** — contributions welcome!

### Adding a Feature to Your Binding

All bindings wrap the same stable C API defined in [`chdb.h`](programs/local/chdb.h):

1. **Session** — Wrap `chdb_connect()`, `chdb_query()`, and `chdb_close_conn()`.
2. **Streaming** — Wrap `chdb_stream_query()`, `chdb_stream_fetch_result()`, and `chdb_stream_cancel_query()`.
3. **Arrow Scan** — Wrap `chdb_arrow_scan()` / `chdb_arrow_array_scan()` and `chdb_arrow_unregister_table()`.

### Need Help?

If you have already developed bindings for a language not listed above, or are interested in contributing, please contact us at:

- Discord: [bindings](https://discord.gg/uUk6AKf7yM)
- Email: auxten@clickhouse.com
- Twitter: [@chdb](https://twitter.com/chdb_io)
