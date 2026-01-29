# ClickHouse Format Settings - Code Reference

## Format-Specific Settings Map

### CSV Format Settings

#### Input Settings
```python
CSV_INPUT_SETTINGS = {
    # Delimiter and Quotes
    'format_csv_delimiter': ',',                              # String: Field delimiter
    'format_csv_allow_double_quotes': 1,                      # Bool: Allow double quotes
    'format_csv_allow_single_quotes': 1,                      # Bool: Allow single quotes
    
    # Header and Lines
    'input_format_csv_skip_first_lines': 0,                   # UInt64: Skip N header lines
    'input_format_csv_detect_header': 1,                      # Bool: Auto-detect header
    'input_format_csv_skip_trailing_empty_lines': 0,          # Bool: Skip trailing empty lines
    
    # Data Processing
    'input_format_csv_trim_whitespaces': 1,                   # Bool: Trim whitespace from fields
    'input_format_csv_empty_as_default': 0,                   # Bool: Treat empty as default
    'input_format_csv_allow_cr_end_of_line': 0,               # Bool: Allow \r line endings
    'input_format_csv_allow_whitespace_or_tab_as_delimiter': 0, # Bool: Allow space/tab delimiters
    
    # NULL Representation
    'format_csv_null_representation': '\\N',                  # String: NULL value representation
    
    # Arrays and Tuples
    'input_format_csv_arrays_as_nested_csv': 0,               # Bool: Parse arrays as nested CSV
    'input_format_csv_deserialize_separate_columns_into_tuple': 1, # Bool: Deserialize to tuple
    
    # Column Handling
    'input_format_csv_allow_variable_number_of_columns': 0,   # Bool: Allow variable column count
    
    # Type Handling
    'input_format_csv_enum_as_number': 0,                     # Bool: Treat enums as numbers
    'input_format_csv_use_default_on_bad_values': 0,          # Bool: Use default on bad values
    
    # Schema Inference
    'input_format_csv_use_best_effort_in_schema_inference': 1, # Bool: Use heuristics
    'input_format_csv_try_infer_numbers_from_strings': 0,     # Bool: Infer numbers from strings
    'input_format_csv_try_infer_strings_from_quoted_tuples': 0, # Bool: Infer strings from quoted tuples
}

CSV_OUTPUT_SETTINGS = {
    'output_format_csv_crlf_end_of_line': 0,                  # Bool: Use \r\n instead of \n
    'output_format_csv_serialize_tuple_into_separate_columns': 1, # Bool: Serialize tuples separately
}
```

### TSV (TabSeparated) Format Settings

#### Input Settings
```python
TSV_INPUT_SETTINGS = {
    # Lines and Header
    'input_format_tsv_skip_first_lines': 0,                   # UInt64: Skip N header lines
    'input_format_tsv_detect_header': 1,                      # Bool: Auto-detect header
    'input_format_tsv_skip_trailing_empty_lines': 0,          # Bool: Skip trailing empty lines
    'input_format_tsv_crlf_end_of_line': 0,                   # Bool: Support \r\n line endings
    
    # Data Processing
    'input_format_tsv_empty_as_default': 0,                   # Bool: Treat empty as default
    'input_format_tsv_enum_as_number': 0,                     # Bool: Treat enums as numbers
    
    # Column Handling
    'input_format_tsv_allow_variable_number_of_columns': 0,   # Bool: Allow variable column count
    
    # NULL Representation
    'format_tsv_null_representation': '\\N',                  # String: NULL value representation
    
    # Schema Inference
    'input_format_tsv_use_best_effort_in_schema_inference': 1, # Bool: Use heuristics
}

TSV_OUTPUT_SETTINGS = {
    'output_format_tsv_crlf_end_of_line': 0,                  # Bool: Use \r\n instead of \n
}
```

### JSON Format Settings

#### Input Settings
```python
JSON_INPUT_SETTINGS = {
    # Basic Settings
    'input_format_json_validate_types_from_metadata': 1,      # Bool: Validate types from metadata
    'input_format_import_nested_json': 0,                     # Bool: Import nested JSON objects
    'input_format_json_ignore_unnecessary_fields': 0,         # Bool: Ignore extra fields
    
    # Type Conversion
    'input_format_json_read_bools_as_numbers': 1,             # Bool: Parse bools as numbers
    'input_format_json_read_bools_as_strings': 1,             # Bool: Parse bools as strings
    'input_format_json_read_numbers_as_strings': 1,           # Bool: Parse numbers as strings
    'input_format_json_read_arrays_as_strings': 1,            # Bool: Parse arrays as strings
    'input_format_json_read_objects_as_strings': 1,           # Bool: Parse objects as strings
    
    # Named Tuples
    'input_format_json_named_tuples_as_objects': 1,           # Bool: Parse named tuples as objects
    'input_format_json_defaults_for_missing_elements_in_named_tuple': 1, # Bool: Use defaults
    'input_format_json_ignore_unknown_keys_in_named_tuple': 1, # Bool: Ignore unknown keys
    
    # Error Handling
    'input_format_json_throw_on_bad_escape_sequence': 1,      # Bool: Throw on bad escapes
    'input_format_json_empty_as_default': 0,                  # Bool: Empty as default
    
    # Maps
    'input_format_json_map_as_array_of_tuples': 0,            # Bool: Maps as tuple arrays
    
    # Schema Inference
    'input_format_json_try_infer_named_tuples_from_objects': 1, # Bool: Infer named tuples
    'input_format_json_try_infer_numbers_from_strings': 0,    # Bool: Infer numbers from strings
    'input_format_json_infer_incomplete_types_as_strings': 1, # Bool: Incomplete types as strings
    'input_format_json_infer_array_of_dynamic_from_array_of_different_types': 1, # Bool
    'input_format_json_use_string_type_for_ambiguous_paths_in_named_tuples_inference_from_objects': 1,
    
    # Limits
    'input_format_json_max_depth': 1000,                      # UInt64: Max nesting depth
    'input_format_json_compact_allow_variable_number_of_columns': 0, # Bool
}

JSON_OUTPUT_SETTINGS = {
    # Quoting
    'output_format_json_quote_64bit_integers': 1,             # Bool: Quote 64-bit integers
    'output_format_json_quote_64bit_floats': 0,               # Bool: Quote 64-bit floats
    'output_format_json_quote_decimals': 0,                   # Bool: Quote decimals
    'output_format_json_quote_denormals': 0,                  # Bool: Quote nan/inf
    
    # Formatting
    'output_format_json_escape_forward_slashes': 1,           # Bool: Escape forward slashes
    'output_format_json_named_tuples_as_objects': 1,          # Bool: Named tuples as objects
    'output_format_json_skip_null_value_in_named_tuples': 0,  # Bool: Skip null values
    'output_format_json_array_of_rows': 0,                    # Bool: Output as array of rows
    'output_format_json_validate_utf8': 0,                    # Bool: Validate UTF-8
    'output_format_json_pretty_print': 1,                     # Bool: Pretty print
    
    # Maps
    'output_format_json_map_as_array_of_tuples': 0,           # Bool: Maps as tuple arrays
}
```

### Parquet Format Settings

#### Input Settings
```python
PARQUET_INPUT_SETTINGS = {
    # Column Matching
    'input_format_parquet_allow_missing_columns': 1,          # Bool: Allow missing columns
    'input_format_parquet_case_insensitive_column_matching': 0, # Bool: Case-insensitive matching
    'input_format_parquet_skip_columns_with_unsupported_types_in_schema_inference': 0, # Bool
    
    # Import Settings
    'input_format_parquet_import_nested': 1,                  # Bool: Import nested structures
    
    # Reader Selection
    'input_format_parquet_use_native_reader': 0,              # Bool: Use native reader v1 (deprecated)
    'input_format_parquet_use_native_reader_v3': 0,           # Bool: Use native reader v3 (experimental)
    
    # Optimization
    'input_format_parquet_filter_push_down': 1,               # Bool: Filter pushdown
    'input_format_parquet_bloom_filter_push_down': 1,         # Bool: Bloom filter pushdown
    'input_format_parquet_page_filter_push_down': 1,          # Bool: Page filter pushdown
    'input_format_parquet_use_offset_index': 1,               # Bool: Use offset index
    
    # Performance
    'input_format_parquet_max_block_size': 65536,             # UInt64: Max block size
    'input_format_parquet_prefer_block_bytes': 16744704,      # UInt64: Preferred block bytes
    'input_format_parquet_local_file_min_bytes_for_seek': 8192, # UInt64: Min bytes for seek
    'input_format_parquet_enable_row_group_prefetch': 1,      # Bool: Enable prefetch
    'input_format_parquet_preserve_order': 0,                 # Bool: Preserve row order
    'input_format_parquet_row_batch_size': 100000,            # UInt64: Row batch size
    
    # Memory Management
    'input_format_parquet_memory_high_watermark': 0,          # UInt64: Memory limit
    'input_format_parquet_memory_low_watermark': 0,           # UInt64: Memory low threshold
    
    # Special Features
    'input_format_parquet_enable_json_parsing': 1,            # Bool: Parse JSON columns
    'input_format_parquet_allow_geoparquet_parser': 1,        # Bool: Allow geo parser
    'input_format_parquet_dictionary_as_low_cardinality': 1,  # Bool: Dictionary as LowCardinality
}

PARQUET_OUTPUT_SETTINGS = {
    # Row Groups
    'output_format_parquet_row_group_size': 1000000,          # UInt64: Row group size
    'output_format_parquet_row_group_size_bytes': 536870912,  # UInt64: Row group size in bytes
    
    # Pages
    'output_format_parquet_data_page_size': 1048576,          # UInt64: Page size in bytes
    'output_format_parquet_batch_size': 1024,                 # UInt64: Batch size
    
    # Compression
    'output_format_parquet_compression_method': 'zstd',       # String: snappy, lz4, brotli, zstd, gzip, none
    
    # Encoding
    'output_format_parquet_compliant_nested_types': 1,        # Bool: Use 'element' for lists
    'output_format_parquet_use_custom_encoder': 1,            # Bool: Use custom encoder
    'output_format_parquet_parallel_encoding': 1,             # Bool: Parallel encoding
    
    # Indexes
    'output_format_parquet_write_page_index': 1,              # Bool: Write page index
    'output_format_parquet_write_bloom_filter': 1,            # Bool: Write bloom filters
    'output_format_parquet_bloom_filter_bits_per_value': 10.0, # Float: Bloom filter bits
    'output_format_parquet_bloom_filter_flush_threshold_bytes': 0, # UInt64
    
    # Dictionary
    'output_format_parquet_max_dictionary_size': 1048576,     # UInt64: Max dictionary size
    
    # Version
    'output_format_parquet_version': '2.latest',              # String: 1.0, 2.4, 2.6, 2.latest
    
    # Type Mapping
    'output_format_parquet_string_as_string': 1,              # Bool: String as Parquet String
    'output_format_parquet_fixed_string_as_fixed_byte_array': 1, # Bool
    'output_format_parquet_date_as_uint16': 0,                # Bool: Date as UInt16
    'output_format_parquet_datetime_as_uint32': 0,            # Bool: DateTime as UInt32
    'output_format_parquet_enum_as_byte_array': 1,            # Bool: Enum as BYTE_ARRAY
    
    # Geo
    'output_format_parquet_geometadata': 1,                   # Bool: Write geo metadata
}
```

### Arrow Format Settings

#### Input Settings
```python
ARROW_INPUT_SETTINGS = {
    'input_format_arrow_allow_missing_columns': 1,            # Bool: Allow missing columns
    'input_format_arrow_case_insensitive_column_matching': 0, # Bool: Case-insensitive matching
    'input_format_arrow_skip_columns_with_unsupported_types_in_schema_inference': 0, # Bool
}

ARROW_OUTPUT_SETTINGS = {
    'output_format_arrow_low_cardinality_as_dictionary': 0,   # Bool: LowCardinality as dictionary
    'output_format_arrow_string_as_string': 1,                # Bool: String as Arrow String
    'output_format_arrow_fixed_string_as_fixed_byte_array': 1, # Bool
    'output_format_arrow_compression_method': 'lz4_frame',    # String: lz4_frame, zstd, none
    'output_format_arrow_use_signed_indexes_for_dictionary': 1, # Bool
    'output_format_arrow_use_64_bit_indexes_for_dictionary': 0, # Bool
}
```

### ORC Format Settings

#### Input Settings
```python
ORC_INPUT_SETTINGS = {
    'input_format_orc_allow_missing_columns': 1,              # Bool: Allow missing columns
    'input_format_orc_case_insensitive_column_matching': 0,   # Bool: Case-insensitive matching
    'input_format_orc_skip_columns_with_unsupported_types_in_schema_inference': 0, # Bool
    'input_format_orc_filter_push_down': 1,                   # Bool: Filter pushdown
    'input_format_orc_row_batch_size': 100000,                # UInt64: Batch size
    'input_format_orc_use_fast_decoder': 1,                   # Bool: Use fast decoder
    'input_format_orc_dictionary_as_low_cardinality': 1,      # Bool: Dictionary as LowCardinality
    'input_format_orc_reader_time_zone_name': 'GMT',          # String: Reader timezone
}

ORC_OUTPUT_SETTINGS = {
    'output_format_orc_string_as_string': 1,                  # Bool: String as ORC String
    'output_format_orc_compression_method': 'zstd',           # String: lz4, snappy, zlib, zstd, none
    'output_format_orc_compression_block_size': 262144,       # UInt64: Compression block size
    'output_format_orc_row_index_stride': 10000,              # UInt64: Row index stride
    'output_format_orc_dictionary_key_size_threshold': 0.0,   # Float: Dictionary threshold
    'output_format_orc_writer_time_zone_name': 'GMT',         # String: Writer timezone
}
```

### Avro Format Settings

#### Input Settings
```python
AVRO_INPUT_SETTINGS = {
    'input_format_avro_allow_missing_fields': 0,              # Bool: Allow missing fields
    'input_format_avro_null_as_default': 0,                   # Bool: NULL as default
    'format_avro_schema_registry_url': '',                    # String: Schema registry URL
}

AVRO_OUTPUT_SETTINGS = {
    'output_format_avro_codec': 'snappy',                     # String: null, deflate, snappy, zstd
    'output_format_avro_sync_interval': 16384,                # UInt64: Sync interval
    'output_format_avro_string_column_pattern': '',           # String: String column pattern
    'output_format_avro_rows_in_file': 1,                     # UInt64: Rows in file
}
```

### Protobuf Format Settings

```python
PROTOBUF_SETTINGS = {
    'format_schema': '',                                      # String: Schema file path
    'format_schema_message_name': '',                         # String: Message name
    'format_schema_source': 'file',                           # String: file, string, query
    'format_protobuf_use_autogenerated_schema': 0,            # Bool: Use autogenerated schema
    'input_format_protobuf_flatten_google_wrappers': 1,       # Bool: Flatten Google wrappers
    'input_format_protobuf_oneof_presence': 0,                # Bool: Oneof presence indicator
    'input_format_protobuf_skip_fields_with_unsupported_types_in_schema_inference': 0, # Bool
    'output_format_protobuf_nullables_with_google_wrappers': 0, # Bool
}
```

### CapnProto Format Settings

```python
CAPNPROTO_SETTINGS = {
    'format_schema': '',                                      # String: Schema file path
    'format_capn_proto_enum_comparising_mode': 'by_names',    # String: Enum comparison mode
    'format_capn_proto_use_autogenerated_schema': 0,          # Bool: Use autogenerated schema
    'input_format_capn_proto_skip_fields_with_unsupported_types_in_schema_inference': 0, # Bool
}
```

### Native Format Settings

```python
NATIVE_SETTINGS = {
    'input_format_native_allow_types_conversion': 1,          # Bool: Allow type conversion
    'input_format_native_decode_types_in_binary_format': 0,   # Bool: Decode types in binary
    'output_format_native_encode_types_in_binary_format': 0,  # Bool: Encode types in binary
    'output_format_native_write_json_as_string': 0,           # Bool: Write JSON as string
    'output_format_native_use_flattened_dynamic_and_json_serialization': 0, # Bool
}
```

### RowBinary Format Settings

```python
ROWBINARY_SETTINGS = {
    'format_binary_max_string_size': 1073741824,              # UInt64: Max string size
    'format_binary_max_array_size': 1073741824,               # UInt64: Max array size
    'input_format_binary_decode_types_in_binary_format': 0,   # Bool: Decode types in binary
    'input_format_binary_read_json_as_string': 0,             # Bool: Read JSON as string
    'output_format_binary_encode_types_in_binary_format': 0,  # Bool: Encode types in binary
    'output_format_binary_write_json_as_string': 0,           # Bool: Write JSON as string
}
```

### BSON Format Settings

```python
BSON_SETTINGS = {
    'input_format_bson_skip_fields_with_unsupported_types_in_schema_inference': 0, # Bool
    'output_format_bson_string_as_string': 0,                 # Bool: String as BSON String
}
```

### MsgPack Format Settings

```python
MSGPACK_SETTINGS = {
    'input_format_msgpack_number_of_columns': 0,              # UInt64: Number of columns
    'output_format_msgpack_uuid_representation': 'ext',       # String: UUID representation
}
```

### CustomSeparated Format Settings

```python
CUSTOMSEPARATED_SETTINGS = {
    'format_custom_escaping_rule': 'Escaped',                 # String: Escaping rule
    'format_custom_field_delimiter': '\t',                    # String: Field delimiter
    'format_custom_row_before_delimiter': '',                 # String: Row before delimiter
    'format_custom_row_after_delimiter': '\n',                # String: Row after delimiter
    'format_custom_row_between_delimiter': '',                # String: Row between delimiter
    'format_custom_result_before_delimiter': '',              # String: Result before delimiter
    'format_custom_result_after_delimiter': '',               # String: Result after delimiter
    'input_format_custom_allow_variable_number_of_columns': 0, # Bool
    'input_format_custom_detect_header': 1,                   # Bool: Auto-detect header
    'input_format_custom_skip_trailing_empty_lines': 0,       # Bool
}
```

### Template Format Settings

```python
TEMPLATE_SETTINGS = {
    'format_template_resultset': '',                          # String: Result set template
    'format_template_row': '',                                # String: Row template
    'format_template_rows_between_delimiter': '\n',           # String: Rows delimiter
    'format_template_resultset_format': '',                   # String: Result set format string
    'format_template_row_format': '',                         # String: Row format string
}
```

### Regexp Format Settings

```python
REGEXP_SETTINGS = {
    'format_regexp': '',                                      # String: Regular expression
    'format_regexp_escaping_rule': 'Raw',                     # String: Escaping rule
    'format_regexp_skip_unmatched': 0,                        # Bool: Skip unmatched lines
}
```

### MySQLDump Format Settings

```python
MYSQLDUMP_SETTINGS = {
    'input_format_mysql_dump_table_name': '',                 # String: Table name
    'input_format_mysql_dump_map_column_names': 1,            # Bool: Map column names
}
```

### SQLInsert Format Settings

```python
SQLINSERT_OUTPUT_SETTINGS = {
    'output_format_sql_insert_max_batch_size': 65536,         # UInt64: Max batch size
    'output_format_sql_insert_table_name': 'table',           # String: Table name
    'output_format_sql_insert_include_column_names': 1,       # Bool: Include column names
    'output_format_sql_insert_use_replace': 0,                # Bool: Use REPLACE instead of INSERT
    'output_format_sql_insert_quote_names': 1,                # Bool: Quote column names
}
```

### Pretty Format Settings

```python
PRETTY_OUTPUT_SETTINGS = {
    'output_format_pretty_max_rows': 10000,                   # UInt64: Max rows
    'output_format_pretty_max_column_pad_width': 250,         # UInt64: Max column padding
    'output_format_pretty_max_value_width': 10000,            # UInt64: Max value width
    'output_format_pretty_max_value_width_apply_for_single_value': 1, # Bool
    'output_format_pretty_color': 'auto',                     # String: 0, 1, auto
    'output_format_pretty_grid_charset': 'UTF-8',             # String: ASCII, UTF-8
    'output_format_pretty_row_numbers': 0,                    # Bool: Show row numbers
    'output_format_pretty_display_footer_column_names': 0,    # Bool
    'output_format_pretty_display_footer_column_names_min_rows': 50, # UInt64
    'output_format_pretty_multiline_fields': 1,               # Bool: Allow multi-line fields
    'output_format_pretty_single_large_number_tip_threshold': 1000000, # UInt64
    'output_format_pretty_squash_consecutive_ms': 0,          # UInt64: Squash delay
    'output_format_pretty_squash_max_wait_ms': 10000,         # UInt64: Max wait time
    'output_format_pretty_highlight_digit_groups': 1,         # Bool
    'output_format_pretty_highlight_trailing_spaces': 1,      # Bool
    'output_format_pretty_glue_chunks': 'auto',               # String: 0, 1, auto
    'output_format_pretty_fallback_to_vertical': 0,           # Bool
    'output_format_pretty_fallback_to_vertical_max_rows_per_chunk': 10, # UInt64
    'output_format_pretty_fallback_to_vertical_min_columns': 10, # UInt64
    'output_format_pretty_fallback_to_vertical_min_table_width': 120, # UInt64
    'output_format_pretty_max_column_name_width_cut_to': 60,  # UInt64
    'output_format_pretty_max_column_name_width_min_chars_to_cut': 8, # UInt64
}
```

---

## General Format Settings (Apply to Multiple Formats)

### Error Handling
```python
ERROR_HANDLING_SETTINGS = {
    'input_format_allow_errors_num': 0,                       # UInt64: Max errors allowed
    'input_format_allow_errors_ratio': 0,                     # Float: Max error ratio
    'input_format_allow_seeks': 1,                            # Bool: Allow seeks
    'input_format_record_errors_file_path': '',               # String: Error log path
    'errors_output_format': 'CSV',                            # String: Error output format
}
```

### Schema Inference
```python
SCHEMA_INFERENCE_SETTINGS = {
    'input_format_max_rows_to_read_for_schema_inference': 25000, # UInt64
    'input_format_max_bytes_to_read_for_schema_inference': 33554432, # UInt64
    'schema_inference_make_columns_nullable': 'auto',         # String: 0, 1, 2/auto, 3
    'schema_inference_make_json_columns_nullable': 1,         # Bool
    'schema_inference_mode': 'default',                       # String: default, union
    'schema_inference_hints': '',                             # String: Schema hints
    'column_names_for_schema_inference': '',                  # String: Column names
    'input_format_try_infer_integers': 1,                     # Bool
    'input_format_try_infer_dates': 1,                        # Bool
    'input_format_try_infer_datetimes': 1,                    # Bool
    'input_format_try_infer_datetimes_only_datetime64': 0,    # Bool
    'input_format_try_infer_exponent_floats': 0,              # Bool
    'input_format_try_infer_variants': 0,                     # Bool
}
```

### Default Values
```python
DEFAULT_VALUE_SETTINGS = {
    'input_format_defaults_for_omitted_fields': 1,            # Bool
    'input_format_null_as_default': 1,                        # Bool
    'input_format_force_null_for_omitted_fields': 0,          # Bool
    'input_format_ipv4_default_on_conversion_error': 0,       # Bool
    'input_format_ipv6_default_on_conversion_error': 0,       # Bool
}
```

### Column Matching
```python
COLUMN_MATCHING_SETTINGS = {
    'input_format_skip_unknown_fields': 0,                    # Bool
    'input_format_with_names_use_header': 1,                  # Bool
    'input_format_with_types_use_header': 1,                  # Bool
}
```

### DateTime Settings
```python
DATETIME_SETTINGS = {
    'date_time_input_format': 'best_effort',                  # String: best_effort, best_effort_us, basic
    'date_time_output_format': 'simple',                      # String: simple, iso, unix_timestamp
    'date_time_overflow_behavior': 'ignore',                  # String: ignore, throw, saturate
    'date_time_64_output_format_cut_trailing_zeros_align_to_groups_of_thousands': 0, # Bool
}
```

### Bool Settings
```python
BOOL_SETTINGS = {
    'bool_true_representation': 'true',                       # String
    'bool_false_representation': 'false',                     # String
    'allow_special_bool_values_inside_variant': 1,            # Bool
}
```

### Output Settings
```python
OUTPUT_SETTINGS = {
    'output_format_write_statistics': 1,                      # Bool
    'output_format_decimal_trailing_zeros': 0,                # Bool
    'output_format_schema': '',                               # String: Schema output path
}
```

### Misc Settings
```python
MISC_SETTINGS = {
    'format_display_secrets_in_show_and_select': 0,           # Bool
    'interval_output_format': 'numeric',                      # String: numeric, kusto
    'precise_float_parsing': 1,                               # Bool
    'input_format_max_block_size_bytes': 0,                   # UInt64
    'json_type_escape_dots_in_keys': 0,                       # Bool
    'type_json_skip_duplicated_paths': 0,                     # Bool
    'validate_experimental_and_suspicious_types_inside_nested_types': 1, # Bool
}
```

---

## Usage Examples

### Python Example
```python
# CSV with custom settings
settings = {
    'format_csv_delimiter': '|',
    'input_format_csv_skip_first_lines': 1,
    'input_format_csv_trim_whitespaces': 1,
    'input_format_csv_empty_as_default': 1,
}

query = "SELECT * FROM file('data.csv', 'CSV') SETTINGS " + \
        ", ".join([f"{k}={repr(v)}" for k, v in settings.items()])
```

### SQL Example
```sql
-- Parquet with optimization
SELECT * FROM file('data.parquet', 'Parquet')
SETTINGS
    input_format_parquet_filter_push_down = 1,
    input_format_parquet_bloom_filter_push_down = 1,
    input_format_parquet_max_block_size = 131072;
```

### Go Example
```go
settings := map[string]interface{}{
    "format_csv_delimiter": "|",
    "input_format_csv_skip_first_lines": 1,
    "input_format_csv_trim_whitespaces": 1,
}

// Build query with settings
var settingsStr []string
for k, v := range settings {
    settingsStr = append(settingsStr, fmt.Sprintf("%s=%v", k, v))
}
query := fmt.Sprintf("SELECT * FROM file('data.csv', 'CSV') SETTINGS %s", 
    strings.Join(settingsStr, ", "))
```

---

## Complete Format Names Table

### Text Formats

| Format Name | Input | Output | Description |
|------------|-------|--------|-------------|
| **TabSeparated** | ✓ | ✓ | Tab-separated values |
| **TabSeparatedRaw** | ✓ | ✓ | TSV without escaping |
| **TabSeparatedWithNames** | ✓ | ✓ | TSV with header row |
| **TabSeparatedWithNamesAndTypes** | ✓ | ✓ | TSV with header and types |
| **TabSeparatedRawWithNames** | ✓ | ✓ | Raw TSV with header |
| **TabSeparatedRawWithNamesAndTypes** | ✓ | ✓ | Raw TSV with header and types |
| **CSV** | ✓ | ✓ | Comma-separated values |
| **CSVWithNames** | ✓ | ✓ | CSV with header row |
| **CSVWithNamesAndTypes** | ✓ | ✓ | CSV with header and types |
| **CustomSeparated** | ✓ | ✓ | Custom delimiter format |
| **CustomSeparatedWithNames** | ✓ | ✓ | Custom format with header |
| **CustomSeparatedWithNamesAndTypes** | ✓ | ✓ | Custom format with header and types |
| **Values** | ✓ | ✓ | SQL VALUES format |
| **Vertical** | ✗ | ✓ | Vertical display format |
| **VerticalRaw** | ✗ | ✓ | Vertical without escaping |
| **JSON** | ✓ | ✓ | Standard JSON |
| **JSONStrings** | ✓ | ✓ | JSON with all strings |
| **JSONColumns** | ✓ | ✓ | Columnar JSON |
| **JSONColumnsWithMetadata** | ✓ | ✓ | Columnar JSON with metadata |
| **JSONAsString** | ✓ | ✗ | Parse entire JSON as string |
| **JSONAsObject** | ✓ | ✗ | Parse JSON as object |
| **JSONCompact** | ✓ | ✓ | Compact JSON format |
| **JSONCompactStrings** | ✗ | ✓ | Compact JSON all strings |
| **JSONCompactColumns** | ✓ | ✓ | Compact columnar JSON |
| **JSONEachRow** | ✓ | ✓ | One JSON object per row |
| **PrettyJSONEachRow** | ✗ | ✓ | Pretty JSONEachRow |
| **JSONStringsEachRow** | ✓ | ✓ | JSONEachRow all strings |
| **JSONCompactEachRow** | ✓ | ✓ | Compact JSONEachRow |
| **JSONCompactStringsEachRow** | ✓ | ✓ | Compact JSONEachRow strings |
| **JSONEachRowWithProgress** | ✗ | ✓ | JSONEachRow with progress |
| **JSONStringsEachRowWithProgress** | ✗ | ✓ | Strings with progress |
| **JSONCompactEachRowWithNames** | ✓ | ✓ | Compact with names |
| **JSONCompactEachRowWithNamesAndTypes** | ✓ | ✓ | Compact with names and types |
| **JSONCompactEachRowWithProgress** | ✗ | ✓ | Compact with progress |
| **JSONCompactStringsEachRowWithNames** | ✓ | ✓ | Compact strings with names |
| **JSONCompactStringsEachRowWithNamesAndTypes** | ✓ | ✓ | Compact strings with names/types |
| **JSONObjectEachRow** | ✓ | ✓ | Object-style JSON |
| **TSKV** | ✓ | ✓ | Tab-separated key-value |
| **Template** | ✓ | ✓ | Custom template format |
| **TemplateIgnoreSpaces** | ✓ | ✓ | Template ignoring spaces |
| **Regexp** | ✓ | ✗ | Regular expression parsing |
| **LineAsString** | ✓ | ✓ | Each line as string |
| **RawBLOB** | ✓ | ✓ | Raw binary data |
| **Markdown** | ✗ | ✓ | Markdown table format |
| **XML** | ✗ | ✓ | XML format |
| **MySQLDump** | ✓ | ✗ | MySQL dump format |
| **SQLInsert** | ✗ | ✓ | SQL INSERT statements |
| **Form** | ✓ | ✗ | HTML form format |

### Binary Formats

| Format Name | Input | Output | Description |
|------------|-------|--------|-------------|
| **Native** | ✓ | ✓ | ClickHouse native format (most efficient) |
| **RowBinary** | ✓ | ✓ | Row-based binary |
| **RowBinaryWithNames** | ✓ | ✓ | RowBinary with column names |
| **RowBinaryWithNamesAndTypes** | ✓ | ✓ | RowBinary with names and types |
| **RowBinaryWithDefaults** | ✓ | ✓ | RowBinary with default values |
| **Parquet** | ✓ | ✓ | Apache Parquet format |
| **ParquetMetadata** | ✓ | ✗ | Parquet metadata only |
| **Arrow** | ✓ | ✓ | Apache Arrow format |
| **ArrowStream** | ✓ | ✓ | Arrow streaming format |
| **ORC** | ✓ | ✓ | Apache ORC format |
| **Avro** | ✓ | ✓ | Apache Avro format |
| **AvroConfluent** | ✓ | ✓ | Confluent Avro format |
| **Protobuf** | ✓ | ✓ | Protocol Buffers |
| **ProtobufSingle** | ✓ | ✓ | Single Protobuf message |
| **ProtobufList** | ✓ | ✓ | Protobuf list |
| **CapnProto** | ✓ | ✓ | Cap'n Proto format |
| **MsgPack** | ✓ | ✓ | MessagePack format |
| **BSONEachRow** | ✓ | ✓ | BSON format (MongoDB) |
| **Npy** | ✓ | ✓ | NumPy array format |
| **DWARF** | ✓ | ✗ | DWARF debug information |

### Display Formats

| Format Name | Input | Output | Description |
|------------|-------|--------|-------------|
| **Pretty** | ✗ | ✓ | Pretty table output |
| **PrettyNoEscapes** | ✗ | ✓ | Pretty without ANSI escapes |
| **PrettyMonoBlock** | ✗ | ✓ | Pretty single block |
| **PrettyNoEscapesMonoBlock** | ✗ | ✓ | Pretty mono without escapes |
| **PrettyCompact** | ✗ | ✓ | Compact pretty format |
| **PrettyCompactNoEscapes** | ✗ | ✓ | Compact without escapes |
| **PrettyCompactMonoBlock** | ✗ | ✓ | Compact mono block |
| **PrettyCompactNoEscapesMonoBlock** | ✗ | ✓ | Compact mono no escapes |
| **PrettySpace** | ✗ | ✓ | Space-separated pretty |
| **PrettySpaceNoEscapes** | ✗ | ✓ | Space pretty no escapes |
| **PrettySpaceMonoBlock** | ✗ | ✓ | Space mono block |
| **PrettySpaceNoEscapesMonoBlock** | ✗ | ✓ | Space mono no escapes |

### Special Formats

| Format Name | Input | Output | Description |
|------------|-------|--------|-------------|
| **Null** | ✗ | ✓ | No output (performance testing) |
| **Hash** | ✗ | ✓ | Output data hash |
| **One** | ✓ | ✓ | Single value output |
| **Prometheus** | ✗ | ✓ | Prometheus monitoring format |

---

## Format Settings Mapping

### Quick Reference: Which Settings Apply to Which Format

#### CSV Family
```python
CSV_FAMILY = ['CSV', 'CSVWithNames', 'CSVWithNamesAndTypes']
APPLICABLE_SETTINGS = CSV_INPUT_SETTINGS | CSV_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### TSV Family
```python
TSV_FAMILY = [
    'TabSeparated', 'TabSeparatedRaw', 
    'TabSeparatedWithNames', 'TabSeparatedWithNamesAndTypes',
    'TabSeparatedRawWithNames', 'TabSeparatedRawWithNamesAndTypes'
]
APPLICABLE_SETTINGS = TSV_INPUT_SETTINGS | TSV_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### JSON Family
```python
JSON_FAMILY = [
    'JSON', 'JSONStrings', 'JSONColumns', 'JSONColumnsWithMetadata',
    'JSONAsString', 'JSONAsObject', 'JSONCompact', 'JSONCompactStrings',
    'JSONCompactColumns', 'JSONEachRow', 'PrettyJSONEachRow',
    'JSONStringsEachRow', 'JSONCompactEachRow', 'JSONCompactStringsEachRow',
    'JSONEachRowWithProgress', 'JSONStringsEachRowWithProgress',
    'JSONCompactEachRowWithNames', 'JSONCompactEachRowWithNamesAndTypes',
    'JSONCompactEachRowWithProgress', 'JSONCompactStringsEachRowWithNames',
    'JSONCompactStringsEachRowWithNamesAndTypes', 'JSONObjectEachRow'
]
APPLICABLE_SETTINGS = JSON_INPUT_SETTINGS | JSON_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Parquet
```python
PARQUET_FAMILY = ['Parquet', 'ParquetMetadata']
APPLICABLE_SETTINGS = PARQUET_INPUT_SETTINGS | PARQUET_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Arrow
```python
ARROW_FAMILY = ['Arrow', 'ArrowStream']
APPLICABLE_SETTINGS = ARROW_INPUT_SETTINGS | ARROW_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### ORC
```python
ORC_FAMILY = ['ORC']
APPLICABLE_SETTINGS = ORC_INPUT_SETTINGS | ORC_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Avro
```python
AVRO_FAMILY = ['Avro', 'AvroConfluent']
APPLICABLE_SETTINGS = AVRO_INPUT_SETTINGS | AVRO_OUTPUT_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Protobuf
```python
PROTOBUF_FAMILY = ['Protobuf', 'ProtobufSingle', 'ProtobufList']
APPLICABLE_SETTINGS = PROTOBUF_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### CapnProto
```python
CAPNPROTO_FAMILY = ['CapnProto']
APPLICABLE_SETTINGS = CAPNPROTO_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Native
```python
NATIVE_FAMILY = ['Native']
APPLICABLE_SETTINGS = NATIVE_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### RowBinary
```python
ROWBINARY_FAMILY = [
    'RowBinary', 'RowBinaryWithNames', 
    'RowBinaryWithNamesAndTypes', 'RowBinaryWithDefaults'
]
APPLICABLE_SETTINGS = ROWBINARY_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### CustomSeparated
```python
CUSTOMSEPARATED_FAMILY = [
    'CustomSeparated', 'CustomSeparatedWithNames', 
    'CustomSeparatedWithNamesAndTypes'
]
APPLICABLE_SETTINGS = CUSTOMSEPARATED_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Template
```python
TEMPLATE_FAMILY = ['Template', 'TemplateIgnoreSpaces']
APPLICABLE_SETTINGS = TEMPLATE_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Regexp
```python
REGEXP_FAMILY = ['Regexp']
APPLICABLE_SETTINGS = REGEXP_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### MySQLDump
```python
MYSQLDUMP_FAMILY = ['MySQLDump']
APPLICABLE_SETTINGS = MYSQLDUMP_SETTINGS | GENERAL_FORMAT_SETTINGS
```

#### Pretty
```python
PRETTY_FAMILY = [
    'Pretty', 'PrettyNoEscapes', 'PrettyMonoBlock', 'PrettyNoEscapesMonoBlock',
    'PrettyCompact', 'PrettyCompactNoEscapes', 'PrettyCompactMonoBlock',
    'PrettyCompactNoEscapesMonoBlock', 'PrettySpace', 'PrettySpaceNoEscapes',
    'PrettySpaceMonoBlock', 'PrettySpaceNoEscapesMonoBlock'
]
APPLICABLE_SETTINGS = PRETTY_OUTPUT_SETTINGS
```

---

## Complete Settings Reference by Category

### All CSV Settings
```python
ALL_CSV_SETTINGS = {
    # Delimiter
    'format_csv_delimiter',
    'format_csv_allow_double_quotes',
    'format_csv_allow_single_quotes',
    
    # Lines
    'input_format_csv_skip_first_lines',
    'input_format_csv_detect_header',
    'input_format_csv_skip_trailing_empty_lines',
    'output_format_csv_crlf_end_of_line',
    
    # Processing
    'input_format_csv_trim_whitespaces',
    'input_format_csv_empty_as_default',
    'input_format_csv_allow_cr_end_of_line',
    'input_format_csv_allow_whitespace_or_tab_as_delimiter',
    
    # NULL
    'format_csv_null_representation',
    
    # Complex Types
    'input_format_csv_arrays_as_nested_csv',
    'input_format_csv_deserialize_separate_columns_into_tuple',
    'output_format_csv_serialize_tuple_into_separate_columns',
    
    # Columns
    'input_format_csv_allow_variable_number_of_columns',
    
    # Types
    'input_format_csv_enum_as_number',
    'input_format_csv_use_default_on_bad_values',
    
    # Schema
    'input_format_csv_use_best_effort_in_schema_inference',
    'input_format_csv_try_infer_numbers_from_strings',
    'input_format_csv_try_infer_strings_from_quoted_tuples',
}
```

### All JSON Settings
```python
ALL_JSON_SETTINGS = {
    # Input
    'input_format_json_validate_types_from_metadata',
    'input_format_import_nested_json',
    'input_format_json_ignore_unnecessary_fields',
    'input_format_json_read_bools_as_numbers',
    'input_format_json_read_bools_as_strings',
    'input_format_json_read_numbers_as_strings',
    'input_format_json_read_arrays_as_strings',
    'input_format_json_read_objects_as_strings',
    'input_format_json_named_tuples_as_objects',
    'input_format_json_defaults_for_missing_elements_in_named_tuple',
    'input_format_json_ignore_unknown_keys_in_named_tuple',
    'input_format_json_throw_on_bad_escape_sequence',
    'input_format_json_empty_as_default',
    'input_format_json_map_as_array_of_tuples',
    'input_format_json_try_infer_named_tuples_from_objects',
    'input_format_json_try_infer_numbers_from_strings',
    'input_format_json_infer_incomplete_types_as_strings',
    'input_format_json_infer_array_of_dynamic_from_array_of_different_types',
    'input_format_json_use_string_type_for_ambiguous_paths_in_named_tuples_inference_from_objects',
    'input_format_json_max_depth',
    'input_format_json_compact_allow_variable_number_of_columns',
    
    # Output
    'output_format_json_quote_64bit_integers',
    'output_format_json_quote_64bit_floats',
    'output_format_json_quote_decimals',
    'output_format_json_quote_denormals',
    'output_format_json_escape_forward_slashes',
    'output_format_json_named_tuples_as_objects',
    'output_format_json_skip_null_value_in_named_tuples',
    'output_format_json_array_of_rows',
    'output_format_json_validate_utf8',
    'output_format_json_pretty_print',
    'output_format_json_map_as_array_of_tuples',
}
```

### All Parquet Settings
```python
ALL_PARQUET_SETTINGS = {
    # Input
    'input_format_parquet_allow_missing_columns',
    'input_format_parquet_case_insensitive_column_matching',
    'input_format_parquet_skip_columns_with_unsupported_types_in_schema_inference',
    'input_format_parquet_import_nested',
    'input_format_parquet_use_native_reader',
    'input_format_parquet_use_native_reader_v3',
    'input_format_parquet_filter_push_down',
    'input_format_parquet_bloom_filter_push_down',
    'input_format_parquet_page_filter_push_down',
    'input_format_parquet_use_offset_index',
    'input_format_parquet_max_block_size',
    'input_format_parquet_prefer_block_bytes',
    'input_format_parquet_local_file_min_bytes_for_seek',
    'input_format_parquet_enable_row_group_prefetch',
    'input_format_parquet_preserve_order',
    'input_format_parquet_row_batch_size',
    'input_format_parquet_memory_high_watermark',
    'input_format_parquet_memory_low_watermark',
    'input_format_parquet_enable_json_parsing',
    'input_format_parquet_allow_geoparquet_parser',
    'input_format_parquet_dictionary_as_low_cardinality',
    
    # Output
    'output_format_parquet_row_group_size',
    'output_format_parquet_row_group_size_bytes',
    'output_format_parquet_data_page_size',
    'output_format_parquet_batch_size',
    'output_format_parquet_compression_method',
    'output_format_parquet_compliant_nested_types',
    'output_format_parquet_use_custom_encoder',
    'output_format_parquet_parallel_encoding',
    'output_format_parquet_write_page_index',
    'output_format_parquet_write_bloom_filter',
    'output_format_parquet_bloom_filter_bits_per_value',
    'output_format_parquet_bloom_filter_flush_threshold_bytes',
    'output_format_parquet_max_dictionary_size',
    'output_format_parquet_version',
    'output_format_parquet_string_as_string',
    'output_format_parquet_fixed_string_as_fixed_byte_array',
    'output_format_parquet_date_as_uint16',
    'output_format_parquet_datetime_as_uint32',
    'output_format_parquet_enum_as_byte_array',
    'output_format_parquet_geometadata',
}
```

---

## Setting Value Types Reference

### String Types
- Delimiter settings: Single character string
- Path settings: File path string
- Enum settings: Predefined string values

### Bool Types
- 0 or 1
- true or false

### Numeric Types
- UInt64: Unsigned 64-bit integer
- Float: Floating point number

### Example Values
```python
SETTING_EXAMPLES = {
    # Compression methods
    'compression_method': ['lz4', 'snappy', 'zstd', 'gzip', 'brotli', 'none'],
    
    # Date/Time formats
    'date_time_input_format': ['best_effort', 'best_effort_us', 'basic'],
    'date_time_output_format': ['simple', 'iso', 'unix_timestamp'],
    
    # Schema sources
    'format_schema_source': ['file', 'string', 'query'],
    
    # Escaping rules
    'escaping_rule': ['Escaped', 'Quoted', 'CSV', 'JSON', 'XML', 'Raw'],
    
    # Parquet versions
    'parquet_version': ['1.0', '2.4', '2.6', '2.latest'],
    
    # Avro codecs
    'avro_codec': ['null', 'deflate', 'snappy', 'zstd'],
}
```

---

**Note**: This reference is based on ClickHouse source code and documentation. Settings may vary by version. Always test with your specific ClickHouse version.