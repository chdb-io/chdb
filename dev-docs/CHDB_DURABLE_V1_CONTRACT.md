# chDB Durable V1 Contract

> 版本：**V1**  
> 适用：`chdb-core`、`chdb`（Python）、`chdb-node`、`chdb-go`、`chdb-rust`，以及直接 `dlopen(libchdb)` 的下游 adapter。  
> 性质：**规范**。V1 只覆盖单数据库、单 writer、statement WAL 和全量 checkpoint。下游产品的业务状态与接入约束不属于本协议。

## 0. 规范标记

| 标记 | 含义 |
| --- | --- |
| **[CORE]** | 必须由 `chdb-core` 判断或执行；binding 不得用 SQL 字符串拼接、前缀或正则实现等价逻辑 |
| **[FROZEN]** | 跨 binding 的数据模型和行为契约；JSON 的语义冻结，不要求空白、键顺序等字节级一致 |
| **[BINDING]** | 各 binding 自行实现；API 形式可符合语言习惯，但外部可观察行为必须一致 |

`MUST` / `MUST NOT` 为强制要求；`SHOULD` 为强烈建议，偏离时必须在对应 binding 文档中说明。

一个实现只有在满足全部 V1 要求并通过 §7 conformance 后，才能宣称支持 chDB Durable V1。某种语言晚于 V1 发布实现 binding，不会因此成为协议 V2。

---

## 1. V1 冻结的最小能力

V1 的范围只有：

- 一个 durable object 对应一个 chDB database；
- 一个 writer，任意数量的只读 opener；
- 公共 `execute()` 一次只接受一条 statement；
- database 内 mutation 使用 statement WAL；
- checkpoint 使用完整的 `BACKUP DATABASE`；
- object storage 中使用 immutable base/WAL object 和一个 CAS 更新的 `head.json`；
- lease、heartbeat、fencing、超时与歧义提交 reconcile；
- base/WAL 的长度和 SHA-256 完整性校验；
- 所有 binding 调用同一个 core backup/restore/query-analysis ABI。

V1 明确不支持：

- preamble、UDF 或其他 database 之外的持久状态；
- 多语句公共执行；
- 多 writer 合并；
- 一个 object 内的多个 database；
- Parquet/data WAL；
- 增量 checkpoint；
- GC、destroy 和跨 object 事务。

这些能力的归属见 §8。

---

## 2. 分层与职责

```text
                    +-------------------------------------+
   [BINDING]        | 语言 API、错误类型、资源释放         |
                    +-------------------------------------+
   [FROZEN]         | 对象布局、head、WAL、checksum        |
                    | lease、CAS、fencing、状态机、错误类别 |
                    +-------------------------------------+
   [BINDING]        | provider、认证、重试、超时、reconcile |
                    | 本地 scratch、流式上传下载、编排       |
                    +-------------------------------------+
   [CORE]           | backup、restore、query analysis      |
                    | connection 与 query execution        |
                    +-------------------------------------+
```

判据：

- 只有 ClickHouse parser、AST、catalog 或 backup engine 才能可靠回答的问题，归 core；
- 云 provider、分布式一致性和生命周期策略，归 binding；
- 决定对象能否被另一种语言恢复的格式与行为，归 frozen contract。

对象存储 SDK、认证、lease、WAL、CAS、fencing 和远端重试 **MUST NOT** 进入 core。

---

## 3. chdb-core 必需调整 **[CORE]**

### 3.1 V1 ABI 总览

`programs/local/chdb.h` 必须导出：

1. database full backup；
2. database restore；
3. 带 statement 数量和目标 database 证明的 query analysis；
4. 已有的 `chdb_version()`，供 binding 写入和校验 engine identity。

所有新增符号必须进入 Linux 与 macOS export allowlist，并由 C ABI 测试直接覆盖；只暴露 Python wrapper 不算完成 core ABI。

### 3.2 backup / restore ABI

现有 RC 形状可以保留可选 `base_file_path` 作为低层 engine 能力：

```c
chdb_result * chdb_backup_database_n(
    chdb_connection conn,
    const char * database,       size_t database_len,
    const char * file_path,      size_t file_path_len,
    const char * base_file_path, size_t base_file_path_len);

chdb_result * chdb_restore_database_n(
    chdb_connection conn,
    const char * database,  size_t database_len,
    const char * file_path, size_t file_path_len);
```

但 Durable V1 binding 调用 backup 时 **MUST** 传 `NULL/0`，只生成 full backup。`base_file_path` 不属于 V1 durable conformance；端到端可移植的增量链属于 V2。

core 必须保证：

- database identifier 与文件路径分别传入，由 core 构造 AST 和安全引用；binding **MUST NOT** 拼接 `BACKUP` / `RESTORE` SQL；
- `file_path` 为绝对路径，父目录已存在，并受 `backups.allowed_path` 限制；
- backup 不覆盖已存在文件；
- restore 不隐式改变连接的 current database；
- restore 到已有同名表时不得被 binding 当作 replace。V1 binding 必须恢复到全新 scratch 中的空目标 database；
- 错误通过标准 `chdb_result` 暴露，并使用标准 result destruction ABI 释放；
- backup/restore 失败不得留下一个可被 binding 当作成功归档或成功恢复的结果。

### 3.3 当前 classify ABI 不足以支持 V1

只返回：

```text
(query_class, has_secrets)
```

是不够的，因为 binding 还无法可靠判断：

- 输入究竟包含一条还是多条可执行 statement；
- `PARALLEL WITH` 是否展开为多条执行分支；
- mutation 的所有写目标是否都属于该 durable database；
- 未限定 database 的对象名按当前 session 设置解析后指向哪里；
- `RENAME`、`EXCHANGE` 等包含多个对象的语句是否跨 database；
- 一个看似普通的 `INSERT` 是否实际写向 table function、outfile 或 engine 外部状态。
- 语句是否创建、删除或重命名 durable database 本身；这属于 object lifecycle，不能作为普通 WAL mutation。

这些都必须由 parser/AST 判断，不能下放给 Python、TypeScript、Go 或 Rust 正则。

### 3.4 V1 query-analysis ABI

在首个 RC 冻结前，应将 `chdb_classify_query_n` 扩充为带 target database 的 analysis ABI；如果现有符号已经对外发布，则新增版本化符号而不是原地破坏 ABI。

推荐的最小 C 形状如下，字段使用固定宽度类型并保留 `struct_size`：

```c
typedef enum chdb_query_class {
    CHDB_QUERY_READ_ONLY       = 0,
    CHDB_QUERY_MUTATING        = 1,
    CHDB_QUERY_MUTATING_GLOBAL = 2,
    CHDB_QUERY_CONTROL         = 3,
    CHDB_QUERY_UNKNOWN         = 4
} chdb_query_class;

typedef enum chdb_query_analysis_flag {
    CHDB_QUERY_HAS_SECRETS                 = 1u << 0,
    CHDB_QUERY_WRITES_ONLY_TARGET_DATABASE = 1u << 1,
    CHDB_QUERY_CHANGES_DATABASE_LIFECYCLE  = 1u << 2
} chdb_query_analysis_flag;

typedef struct chdb_query_analysis_v1 {
    uint32_t struct_size;       /* caller sets to sizeof(struct) */
    uint32_t statement_count;   /* executable statements; PARALLEL arms count */
    uint32_t flags;             /* chdb_query_analysis_flag */
    uint32_t query_class;       /* chdb_query_class; fixed-width ABI field */
} chdb_query_analysis_v1;

chdb_state chdb_classify_query_n(
    chdb_connection conn,
    const char * sql,             size_t sql_len,
    const char * target_database, size_t target_database_len,
    chdb_query_analysis_v1 * out_analysis);
```

名字可以在 core PR 中按现有 ABI 规范调整，但以下语义不能减少：

- caller 必须设置 `struct_size`；core 对过小结构返回 `CHDBError`，不得写出 caller 声明的边界，成功时必须初始化所有 V1 字段；
- `statement_count`：统计可执行 statement；空输入或解析失败为 `0`，多个顶层 statement 大于 `1`，`PARALLEL WITH` 的执行分支也按多 statement 处理；
- `CHDB_QUERY_HAS_SECRETS`：AST 含凭证时置位；解析失败不得声称已安全识别；
- `CHDB_QUERY_WRITES_ONLY_TARGET_DATABASE`：只有在 core 能证明全部持久写目标都位于 `target_database` 时置位；只要存在其他 database、`system`、table function、outfile 或其他 engine 外写目标，就不得置位；
- `CHDB_QUERY_CHANGES_DATABASE_LIFECYCLE`：`CREATE/DROP/RENAME DATABASE` 等改变 database 容器本身时置位；V1 的冷对象建库由 adapter 内部完成，公共 WAL 不得改变容器生命周期；
- 未限定 database 的目标必须按照该 connection 实际 parser/session 语义解析，再与 `target_database` 比较；
- `MUTATING_GLOBAL` 在 V1 中只是准确的诊断分类，公共入口一律拒绝，不进入任何 preamble；
- `CONTROL` 包括 session mutation、BACKUP/RESTORE/SYSTEM 等受管理操作和外部写；
- 显式启用 async insert、关闭 insert wait 或降低 mutation 同步保证的 statement setting 必须归为 `CONTROL`；公共 SQL 不得绕过 managed connection 的同步完成策略；
- parse failure 或尚未覆盖的新 AST 返回 `UNKNOWN`，binding fail closed；
- analysis 不执行 SQL，不改变 current database、setting 或 query log；
- 必须正确跨过 `INSERT ... VALUES`、`INSERT ... FORMAT` 和 `EXPLAIN INSERT` 的内联数据；数据中的分号或空行不能被误判为下一条 statement。

V1 `execute(sql)` 的 core gate 固定为：

```text
statement_count == 1
AND query_class == MUTATING
AND WRITES_ONLY_TARGET_DATABASE
AND NOT CHANGES_DATABASE_LIFECYCLE
AND NOT HAS_SECRETS
```

V1 `query(sql)` 的 core gate 固定为：

```text
statement_count == 1
AND query_class == READ_ONLY
```

`READ_ONLY` SQL 即使含 secret 也可以执行，因为它不进入 WAL；binding 必须保证 SQL、错误和 tracing 不泄漏原文 secret。

Python `_chdb.Connection` 必须同步暴露 analysis 的全部字段；继续只返回 `(query_class, has_secrets)` 的旧 tuple 不能供 Durable V1 使用。Node、Go、Rust 和直接 FFI adapter 都必须绑定同一个结构，不能从 Python wrapper 反推或补算缺失字段。

### 3.5 core 验收矩阵

首个 core RC 至少必须覆盖：

- quoted database/table identifier；
- 未限定 database 的目标解析；
- 写入 target database 与写入其他 database；
- 跨 database `RENAME` / `EXCHANGE` / 多目标 DDL；
- `CREATE/DROP/RENAME DATABASE` lifecycle flag；
- `INSERT INTO TABLE` 与 `INSERT INTO FUNCTION`；
- `INTO OUTFILE`；
- `system` database write；
- `CREATE FUNCTION`、access entity、named collection → `MUTATING_GLOBAL`；
- `SET` / `USE` / `SYSTEM` / `BACKUP` / `RESTORE` → `CONTROL`；
- 显式 async insert 或降低 mutation 同步 setting → `CONTROL`；
- 普通 SELECT 与 secret-bearing read-only query；
- 单 statement、多个 statement、尾随分号和注释；
- `PARALLEL WITH`；
- `VALUES` / 任意 `FORMAT` 内联数据后的边界；
- parse failure / 新未知 AST → `UNKNOWN`；
- analysis 前后 session state 和 query log 不变；
- backup 绝对路径、allowed path、已存在目标、错误 result 生命周期；
- restore 到全新目录成功，恢复失败后不能被误认为可用；
- Linux/macOS symbol export 和 Python surface 调用同一 C ABI。

### 3.6 core 的进程约束

V1 继承当前 engine 约束：

- 一个进程只有一个 active EmbeddedServer / data path；
- 同一物理路径可以有多个 connection；
- 仍有旧路径 connection 存活时，打开不同路径必须失败；
- 全部旧 connection 关闭后才可以绑定新路径。

因此同一进程同一时刻只能激活一个 durable object 的 scratch 路径。批量扫描必须顺序打开，或由上层使用多进程 worker。解除该限制属于 core 的后续架构工作，不由 binding 用进程级 registry 伪装解决。

---

## 4. 对象协议 **[FROZEN]**

### 4.1 对象布局

```text
<namespace>/<object-id>/
  head.json
  checkpoints/<generation>-<seq>-<uuid8>.tar.gz
  wal/<generation>-<seq>-<uuid8>.jsonl
```

- `generation` 与 `seq` 为无前导零的十进制整数；
- `uuid8` 为 UUID4 十六进制小写形式的前 8 位；
- checkpoint 与 WAL key 每次尝试都必须唯一，并用真正的 conditional create 发布；
- head 中的 object reference 是相对于 `<namespace>/<object-id>/` 的 key，必须使用 `/` 分隔，不得以 `/` 开头，也不得包含空段、`.` 或 `..`；
- V1 没有 `preamble/` 或 blob 目录。

### 4.2 `head.json`

UTF-8 JSON，无 BOM。键顺序和空白不冻结；字段类型、含义和状态转换冻结。

```json
{
  "protocol": {
    "version": 1,
    "reader_features": [],
    "writer_features": []
  },
  "engine": {
    "name": "chdb",
    "version": "26.7.2-rc.2",
    "backup_format": 1,
    "min_reader": "26.7.2-rc.2"
  },
  "lease": {
    "generation": 3,
    "owner": "worker-visible-name",
    "instance": "unique-live-instance-id",
    "expires_at": 1788230400.0
  },
  "manifest": {
    "db": "default",
    "base": {
      "key": "checkpoints/3-8-acde1234.tar.gz",
      "size": 1048576,
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    },
    "wal": [
      {
        "key": "wal/3-9-acde5678.jsonl",
        "size": 127,
        "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
      }
    ],
    "seq": 9
  }
}
```

已释放 lease 的表示固定为：

```json
{
  "generation": 3,
  "owner": null,
  "instance": null,
  "expires_at": null
}
```

规则：

- `lease.generation` 每次从无 owner 状态获取、过期接管或显式 force takeover 时递增；heartbeat 和普通 manifest commit 不递增；
- `manifest.seq` 每次发布新的 WAL 或 checkpoint 引用时递增；heartbeat 不递增；
- `engine.version` 为写入端 `chdb_version()` 返回的精确值，只用于诊断和审计，不作为 exact-match gate；
- `engine.backup_format` 为 chDB backup archive 格式代际；V1 baseline 为 `1`；
- `engine.min_reader` 为能够读取当前 object 的最低 chDB engine 版本；
- 从首个 Durable V1 archive 开始，后续 chdb-core release 必须能恢复同一 release 或更早 release 经 `chdb_backup_database_n` 创建的 V1 full backup；只有当 archive 格式代际真的不兼容时，才通过提高 `backup_format` 让旧 reader fail closed；
- `manifest.db` 为该 object 持有的唯一 database；冷建时取调用方给的名字，调用方不给时所有 binding 一律用 `default`；写入之后 head 即为权威，再次打开时传入的 database 参数必须被忽略；
- `manifest.base` 为 object reference 或 `null`；
- 每个 object reference 都必须包含 `key`、字节 `size` 和小写完整 SHA-256；
- `manifest.wal` 按重放顺序排列；
- 所有整数必须在跨语言安全整数范围内；
- binding 写回 head 时必须保留顶层、`protocol`、`engine`、`lease` 和 `manifest` 中不认识的字段。

### 4.3 版本和 feature 协商

- 高于本实现基线的 `protocol.version` → 拒绝打开；
- 未识别的 `reader_features` → 拒绝读取；
- 未识别的 `writer_features` → 可以在读检查通过后只读打开，但拒绝取得 writer lease；
- V1 基线不定义任何非空 feature 名；新增可选语义必须先进入 registry、scenario 和 fixture，不能只改某个 binding；
- JSON 加字段不自动递增 protocol version，但 writer 必须 round-trip 未知字段。

Engine 兼容性不比较 `engine.version` 是否和当前运行版本完全相等，而是使用显式兼容字段：

```text
head.engine.backup_format > reader backup-format baseline  -> engine_incompatible
running chdb_version() < head.engine.min_reader             -> engine_incompatible
otherwise                                                   -> open
```

这意味着一个由 `26.7.2-rc.2` 写出的 V1 object，可以由满足 `min_reader` 且支持同一 `backup_format`
的后续 chdb-core release 打开；`engine.version` 仍保留在 head 中，方便排查“这个 object 最初是谁写的”。

### 4.4 WAL

WAL 为 UTF-8 JSONL，每行一个 statement record：

```json
{"sql":"INSERT INTO t VALUES (1)"}
```

规则：

- 一行恰好一个 JSON object，必须以 `\n` 结束；
- `sql` 必须为字符串，恢复时按 manifest 和行顺序执行；
- 恢复执行走 adapter 的内部 replay 路径，不能再次进入公共 `execute()`；
- 整个 segment 在上传前计算 `size` 和 SHA-256；下载后、解析前必须校验二者；
- writer 必须使用唯一 key 和 conditional create，不能覆盖已发布 segment；
- V1 单条 `sql` UTF-8 最大 64 MiB，未压缩 WAL segment 最大 128 MiB；writer 超限必须在本地拒绝，reader 必须能处理到该上限；checkpoint 不设协议级大小上限，使用 streaming/provider 限制；
- SQL 中的 `now()`、`rand()`、UUID、外部可变输入等不会被 binding 自动物化。V1 保证按顺序重放原 statement，不保证非确定性 statement 得到相同结果；调用方对此负责；
- secret-bearing mutation 已由 §3.4 拒绝，不得出现在 WAL。

### 4.5 checksum 与损坏处理

- base 和 WAL 必须在使用前同时校验 `size` 与 SHA-256；
- immutable object 缺失、长度不符或 checksum 不符均为 `corrupt`；
- 不得跳过损坏 WAL、退回旧 base 或返回部分恢复的 session；
- `head.json` 必须经过严格 schema 校验；未知字段保留不等于已知字段可以类型宽松；
- V1 `head.json` 最大 1 MiB；writer 必须在超过限制前 checkpoint 或返回 `limit_exceeded`，reader 对超限 head fail closed；
- JSON 比较是语义比较，不要求不同 binding 产生相同键顺序或空白。

---

## 5. Binding 状态机 **[BINDING + FROZEN BEHAVIOR]**

### 5.1 Backend 最小语义

每个 backend 必须提供等价能力：

```text
get(key)                               -> bytes | missing
get_with_etag(key)                     -> (bytes | missing, etag | missing)
put_file_if_absent(key, local_path)    -> created | already-exists | ambiguous
put_bytes_if_absent(key, bytes)        -> created | already-exists | ambiguous
replace_if_match(key, bytes, etag)     -> new-etag | not-replaced | ambiguous
download_to_file(key, local_path)      -> found | missing
```

具体方法名和 streaming API 可以不同，但必须满足：

- checkpoint 上传/下载不得强制把整个 archive 一次性载入内存；必须支持文件或流式传输；
- conditional create 和 conditional replace 必须是 provider 的真实原子条件操作，不能用 HEAD + PUT 模拟；
- ETag 是 opaque CAS token，不得假设为 MD5；
- 本地目标文件先写唯一临时路径，校验完成后再原子发布到 scratch 中的最终路径；
- `delete_prefix` 不属于 V1，因为 V1 没有 destroy/GC；
- 每个宣称支持的 provider 必须独立通过真实后端 conformance，不能用一个内存 mock 推导其他 provider 兼容。

### 5.2 writer open

writer `open` 必须：

1. 读取并严格校验 head；head 不存在时，用一次 conditional create 原子创建冷 manifest 并取得 generation 1 lease；
2. 检查 protocol feature、`backup_format` 和 `min_reader`；不得因为 `engine.version` 与当前运行版本不同而拒绝；
3. 已有 head 用 CAS 获取 lease；正常获取只允许无 owner 或已超过过期与时钟偏差窗口的 lease；
4. 创建 binding 私有、唯一、空的 scratch 目录；
5. 下载并校验 base，恢复到空目标 database；base 为 `null` 时创建冷 database；
6. 下载、校验并顺序重放 WAL；
7. 在返回对象前再次通过 CAS 续租，确认恢复期间没有失去 lease；
8. 任一步失败都关闭 partial engine、释放或使 lease 尽快失效，并清理 binding 自己拥有的 scratch。

只读 open：

- head 不存在时返回 `not_found`，不得隐式创建；
- 不取得 lease；
- 恢复第一次读取到的 manifest 快照；immutable reference 保证该快照在 writer 后续提交时仍可读；
- 只允许 `READ_ONLY` query。

### 5.3 operation serialization

每个 durable object 必须有一个明确的 operation queue/mutex。至少以下操作与 heartbeat head CAS 必须以一个确定顺序协调：

```text
execute / query / flush / checkpoint / close / lease renewal
```

不得依赖“当前语言运行时单线程”或“native 调用现在是同步的”作为串行化保证。异步 provider I/O 期间也必须维持状态机不变量。

Durable object 必须拥有不向调用方暴露的 managed connection，并把 current database 固定为 manifest database。adapter 必须关闭 async insert 或等待其完成，并把 mutation 配置为同步完成；公共 `SET/USE` 和降低这些保证的 statement setting 已由 core analysis 拒绝。

### 5.4 `query` / `execute` / `flush`

- `query()`：只接受 §3.4 的 query gate；不记录 WAL；secret-bearing read-only SQL 可以执行但不得泄漏；
- `execute()`：只接受 §3.4 的 execute gate；先本地执行，成功后追加到内存 WAL buffer；失败语句不得记录；
- `execute()` 成功只表示本地执行与 buffer 追加成功，不表示远端 durable；
- `flush()`：发布 immutable WAL segment，再 CAS 更新 head；只有 CAS 已确认或按 §5.8 reconcile 为已提交时才返回成功；
- 需要“调用成功即跨进程持久”的产品必须在自己的成功响应前调用并等待 `flush()`；这属于产品/binding 接入策略，不改变通用 `execute()` 语义。

### 5.5 `checkpoint`

checkpoint 必须在 operation queue 中独占执行：

1. 对包含全部已执行 mutation 的当前本地 database 生成 full backup；
2. 以流或文件方式上传唯一 checkpoint key；
3. 记录并校验其 `size` / SHA-256；
4. CAS head，将 `base` 替换为新 reference、清空 `wal`、递增 `manifest.seq`；
5. CAS 未提交时，旧 base/WAL manifest 仍是权威；本地未提交 WAL buffer 不得丢失；
6. CAS 成功后，checkpoint 已经包含的本地 WAL buffer 才可清空。

V1 不使用 incremental base，不依赖 provider 原生 `BACKUP TO S3()`。

### 5.6 `close`

writer close 必须：

1. 停止接受新操作；
2. 等待 operation queue drain；
3. 尝试 flush；
4. flush 成功后 CAS 释放 lease；
5. 无论远端步骤成功或失败，都关闭 native connection 并清理本地 scratch；
6. 持久化或 lease release 失败必须对调用者可见。

close 后的任何公共操作返回 `closed`。析构器/`Drop` 可以 best effort 回收资源，但不能冒充一个已成功完成 durability barrier 的显式 close。

### 5.7 lease、heartbeat 与 fencing

- writer 必须周期续租；heartbeat interval 不得大于 lease TTL 的三分之一；
- 所有 head CAS，包括 heartbeat，都必须携带当前 ETag，并在 operation queue 中协调；
- writer 一旦无法在本地认为的有效期内确认续租，必须进入 self-fenced 状态，拒绝新的 execute/flush/checkpoint；
- 普通 takeover 只允许 lease 已过期并超过实现声明的最大 clock-skew allowance；
- 未过期 lease 只能通过显式管理员 `force` 操作接管；不得在普通重试中自动 force；
- 每次 takeover 递增 generation；旧 writer 的任何后续 head CAS 都必须失败并映射为 `lease_fenced`；
- force 操作必须返回或记录明确 warning：原 writer 未 flush 的本地写入可能丢失。

默认 TTL、heartbeat 和 clock-skew allowance 可以由 binding API 配置，但其默认值、单位和生效规则必须写入 binding 文档并进入同一 scenario 测试。

### 5.8 超时和歧义提交 reconcile

provider 请求超时不能自动等同于失败：服务端可能已经提交。

- immutable PUT 响应不确定：重新读取同一唯一 key，`size` 与 SHA-256 一致即视为已上传；内容不一致为 `corrupt`；仍无法判断为 `commit_ambiguous`；
- head CAS 响应不确定：重新读取 head。如果 intended immutable key 已被引用、manifest/seq 符合预期且 lease instance/generation 仍属于当前 writer，则视为成功；
- reread 仍为旧 ETag/state 时，可以在同一截止时间内重试 CAS；
- head 已变且所有权丢失 → `lease_fenced`；
- 截止时间内仍不能证明成功或失败 → `commit_ambiguous`，不得向调用方谎报成功；
- 所有重试必须有 deadline、退避和最大次数，且不能跨越 self-fence 状态继续写。

---

## 6. 错误类别 **[FROZEN]**

语言可以用异常、error 值或 `Result` 表示，但以下类别必须可编程区分：

| 类别 | 触发 |
| --- | --- |
| `not_found` | 只读打开不存在的 object，或明确要求 existing-only 时不存在 |
| `lease_held` | 另一个未过期 writer 持有 lease |
| `lease_fenced` | 当前实例已失去 generation/ETag 所有权 |
| `engine_incompatible` | object 使用了当前 engine 不能恢复的 backup format，或当前 `chdb_version()` 低于 object 声明的 `min_reader`，或 engine 名称不是 `chdb` |
| `protocol_unsupported` | protocol version 或 feature 不支持 |
| `corrupt` | head schema 错、immutable object 缺失、长度/checksum 不符或恢复内容不完整 |
| `classification_refused` | statement count、class 或目标 database 不满足入口 gate |
| `secret_refused` | mutation 含 secret，不能写入 WAL |
| `engine` | core query/backup/restore 错误 |
| `backend` | provider 网络、认证或非条件冲突错误 |
| `timeout` | 操作明确未提交并超过 deadline |
| `commit_ambiguous` | reconcile 后仍不能证明远端是否提交 |
| `limit_exceeded` | SQL、WAL segment、head 或 provider object 超过 V1/实现已声明限制 |
| `closed` | 对已完成 close 的对象发起操作；self-fenced writer 返回 `lease_fenced` |

provider 的 precondition failed / 412 必须先解释为 CAS 竞争，再结合 lease/state 映射；不得直接混同为普通 `backend`。

错误消息不得包含 secret-bearing SQL、provider credential 或未脱敏连接参数。

---

## 7. V1 conformance

### 7.1 ABI

- [ ] backup/restore/classify 使用同一 core C ABI，不拼 SQL、不维护 SQL 正则表
- [ ] query analysis 返回 statement count、secret flag 和 target-database-only proof
- [ ] 所有平台导出相同符号，Python surface 只是同一 ABI 的 wrapper
- [ ] 互通参与者使用相同 Durable V1 contract/fixtures，并覆盖 producer-version 不同但兼容、reader 低于 `min_reader`、`backup_format` 过新的三类 engine gate

### 7.2 格式 fixtures

- [ ] `empty-object`
- [ ] read-only open missing object → `not_found`
- [ ] `checkpoint-only`
- [ ] `checkpoint-plus-wal`
- [ ] `quoted-database-name`
- [ ] `missing-base` / `missing-wal` → `corrupt`
- [ ] `bad-base-size` / `bad-base-sha256` → `corrupt`
- [ ] `bad-wal-size` / `bad-wal-sha256` → `corrupt`
- [ ] `unknown-reader-feature` → 拒绝打开
- [ ] `unknown-writer-feature` → 允许只读、拒绝 writer lease
- [ ] `future-protocol-version` → 拒绝打开
- [ ] `producer-version-differs-but-compatible` → open
- [ ] `engine-reader-too-old` → `engine_incompatible`
- [ ] `backup-format-too-new` → `engine_incompatible`
- [ ] 未知字段经过 `open → execute → flush → checkpoint → close` 后仍保留
- [ ] 不同 JSON 键顺序和空白可以互读

### 7.3 分类 scenarios

- [ ] query/execute 方法名不能绕过 analysis
- [ ] 多语句与 `PARALLEL WITH` 被拒绝
- [ ] 写其他 database、system 或外部 sink 被拒绝
- [ ] database lifecycle statement 被拒绝
- [ ] 所有 `MUTATING_GLOBAL` 被拒绝
- [ ] secret mutation 被拒绝且错误不泄漏 SQL
- [ ] secret-bearing `READ_ONLY` 可执行但不写 WAL
- [ ] SQL/WAL/head 超过冻结上限时返回 `limit_exceeded`
- [ ] UNKNOWN fail closed

### 7.4 fault matrix

- [ ] object/head 条件创建竞争只有一个 writer 成功
- [ ] WAL PUT 成功、head CAS 失败时旧 manifest 有效且本地 buffer 保留
- [ ] head CAS 已提交但响应丢失时能 reconcile 为成功
- [ ] 无法证明 CAS 结果时返回 `commit_ambiguous`
- [ ] checkpoint PUT 成功、head CAS 失败时旧 base/WAL 仍可恢复
- [ ] restore/replay 失败时关闭 partial engine，不返回部分 session
- [ ] heartbeat 与 flush/checkpoint 并发不破坏 ETag/seq
- [ ] heartbeat 失败至过期时 writer self-fence
- [ ] force takeover 后旧 writer 下一次提交被 fence
- [ ] close flush 失败时错误可见且本地资源仍释放

### 7.5 跨 binding

- [ ] 每个 writer 产出的 fixture 至少由另外两个 binding 读取
- [ ] 至少一个真实 object backend 通过完整条件写和故障测试
- [ ] 每个额外声明支持的 provider 独立通过 provider conformance
- [ ] Python、Node、Go 使用相同 protocol/scenario/fixture 权威来源
- [ ] Rust 在 Session/path lifecycle 完成后加入同一组 V1 测试；它的晚实施不是协议 V2

---

## 8. V2 与后续工作

以下能力不进入 V1；开始实现前必须分别形成 proposal、feature 名、fixtures 和升级/降级策略。

### 8.1 chdb-durable 协议 V2 候选

1. **preamble / global state**：SQL UDF、WASM blob、压缩、删除语义、driver 缺失与重放失败策略。V1 对全部 `MUTATING_GLOBAL` fail closed。
2. **增量 checkpoint**：解决 base chain 的跨机器寻址、链完整性、压缩、合并、最大链长和 GC 后再进入协议；不能直接持久化本地绝对 `base_file_path`。
3. **Parquet/data WAL**：定义数据 schema、DDL 与数据顺序、mutation 表达、MV 重放和 statement/data WAL 混用规则后再引入。
4. **GC / destroy**：安全枚举引用、orphan grace period、并发 reader 保护、恢复窗口和显式危险操作确认。
5. **多 writer**：需要冲突模型、提交顺序、幂等键和 merge/rebase 语义，不能只放宽 lease。
6. **多 database / 跨 object 事务**：需要新的 manifest 和原子提交模型。
7. **engine compatibility matrix 与在线迁移**：V1 已定义后续 engine 读取早期 full backup 的兼容 gate（`backup_format` + `min_reader`）；更复杂的在线迁移、schema migration、非兼容 archive format 切换和回滚策略后续设计。

### 8.2 不属于协议版本的发布顺序

- 具体下游 adapter 的产品策略不进入本 contract；
- `chdb-rust` 先修复 Session/path registry 与资源生命周期，再实现同一个 Durable V1 binding；
- 同进程多 durable path 并行必须先改变 chdb-core EmbeddedServer/path 生命周期模型；这是 core 后续能力，不因实现时间自动变成协议 V2；
- 某个 provider 或语言 binding 晚发布，只表示实现进度，不自动成为 V2；
- 下游应用数据库以外的 archive、配置或业务事务，应由该应用自己的 binding/integration 规范定义，除非未来证明存在跨应用的通用协议需求。

---

## Appendix A. 相对旧草案的 V1 收敛项

本次冻结前调整如下：

1. 删除 V1 `preamble/`、blob、UDF replay、相关 warning 和 fixtures；所有 `MUTATING_GLOBAL` 改为 fail closed。
2. 公共 `query()` / `execute()` 都限制为一个可执行 statement；`PARALLEL WITH` 不作为单 statement 绕过。
3. core classify 从 `(class, has_secrets)` 扩展为 query analysis，新增 statement count 和“全部写目标只属于指定 database”的证明。
4. `execute()` 只接受单条、无 secret、仅写 durable database 的 `MUTATING`；跨 database、system 和 external sink 拒绝。
5. JSON 从“跨 binding 字节完全相同”改为严格 schema 下的语义兼容；键顺序与空白不冻结。
6. base/WAL reference 新增长度和 SHA-256，恢复前强制校验。
7. checkpoint backend 从整对象 `bytes` 改为必须支持文件或 streaming，避免 full backup 全量驻留内存。
8. 明确每 object operation queue，heartbeat 也参与同一 CAS 串行化。
9. 明确 heartbeat、自我 fencing、过期 takeover 与显式未过期 force takeover。
10. 补充 `not_found`、`engine_incompatible`、`timeout`、`commit_ambiguous` 和 `closed` 错误类别。
11. 删除 V1 `delete_prefix` / destroy；GC 和销毁留到后续安全设计。
12. 增量 checkpoint、Parquet/data WAL、多 writer、多 database 和跨版本 migration 明确移入 V2 候选。
