# DataStore Function Reference

DataStore provides ClickHouse SQL functions through multiple interfaces:

1. **Accessor Pattern** (`.str`, `.dt`, `.arr`, `.json`, `.url`, `.ip`, `.geo`) - Pandas-like API for domain-specific functions
2. **Expression Methods** - Direct methods on column expressions
3. **Function Namespace (`F`)** - Explicit function calls

## Quick Reference

```python
from datastore import DataStore, F, Field

ds = DataStore.from_file('data.csv')

# Accessor pattern (recommended for chaining)
ds['name'].str.upper()           # String function
ds['date'].dt.year               # DateTime function
ds['tags'].arr.length            # Array function
ds['data'].json.get_string('name')  # JSON function
ds['link'].url.domain()          # URL function
ds['ip_addr'].ip.to_ipv4()       # IP function
ds['coords'].geo.l2_distance(other)  # Geo function

# Expression methods
ds['value'].abs()                # Math function
ds['price'].sum()                # Aggregate function
ds['value'].cast('Float64')      # Type conversion

# F namespace (explicit)
F.upper(Field('name'))
F.sum(Field('value'))
```

---

## Function Statistics

### ClickHouse Functions

| Category | CH Total | Implemented | Coverage |
|----------|----------|-------------|----------|
| **DATETIME** | 153 | 57 | 37.3% |
| **STRING** | 232 | 46 | 19.8% |
| **AGGREGATE** | 126 | 45 | 35.7% |
| **ARRAY** | 219 | 37 | 16.9% |
| **MATH** | 409 | 36 | 8.8% |
| **WINDOW** | 47 | 17 | 36.2% |
| **URL** | 26 | 15 | 57.7% |
| **GEO** | 73 | 14 | 19.2% |
| **JSON** | 11 | 13 | 100%+ |
| **CONDITIONAL** | 29 | 10 | 34.5% |
| **IP** | 23 | 9 | 39.1% |
| **TYPE_CONVERSION** | 40 | 7 | 17.5% |
| **ENCODING** | 4 | 7 | 100%+ |
| **HASH** | 17 | 4 | 23.5% |
| **UUID** | 12 | 4 | 33.3% |
| **Total** | **1,475** | **334** | **22.6%** |

> Note: Some categories exceed 100% because we implement additional Pandas-compatible functions not in ClickHouse.

### Pandas Compatibility

| Category | Pandas Total | Implemented | Notes |
|----------|--------------|-------------|-------|
| DataFrame methods | 209 | **209** | All pandas DataFrame methods |
| Series.str accessor | 56 | **56** | All pandas str methods |
| Series.dt accessor | 42 | **42+** | All pandas + ClickHouse extras |
| Series.arr accessor | - | 37 | ClickHouse-specific |
| Series.json accessor | - | 13 | ClickHouse-specific |
| Series.url accessor | - | 15 | ClickHouse-specific |
| Series.ip accessor | - | 9 | ClickHouse-specific |
| Series.geo accessor | - | 14 | ClickHouse-specific |

---

## String Functions (`.str` accessor)

Access via `ds['column'].str.<function>()`.

### Case Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `upper()` | `upper(s)` | Convert to uppercase | `ds['name'].str.upper()` |
| `lower()` | `lower(s)` | Convert to lowercase | `ds['name'].str.lower()` |
| `capitalize()` | `initcap(s)` | Capitalize first letter | `ds['name'].str.capitalize()` |
| `title()` | `initcap(s)` | Title case | `ds['name'].str.title()` |

### Length & Size

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `length()` / `len()` | `length(s)` | String length in bytes | `ds['name'].str.length()` |
| `char_length()` | `char_length(s)` | Length in Unicode code points | `ds['name'].str.char_length()` |

### Substring & Slicing

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `substring(offset, length)` | `substring(s, offset, length)` | Extract substring (1-indexed) | `ds['name'].str.substring(1, 5)` |
| `left(n)` | `left(s, n)` | Get leftmost N characters | `ds['name'].str.left(3)` |
| `right(n)` | `right(s, n)` | Get rightmost N characters | `ds['name'].str.right(3)` |
| `slice(start, stop)` | `substring(s, start, len)` | Python-style slicing | `ds['name'].str.slice(0, 5)` |

### Trimming

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `trim()` / `strip()` | `trim(s)` | Remove leading/trailing whitespace | `ds['name'].str.trim()` |
| `ltrim()` / `lstrip()` | `trimLeft(s)` | Remove leading whitespace | `ds['name'].str.ltrim()` |
| `rtrim()` / `rstrip()` | `trimRight(s)` | Remove trailing whitespace | `ds['name'].str.rtrim()` |

### Search & Match

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `contains(needle)` | `position(s, needle)` | Check if contains substring | `ds['name'].str.contains('test')` |
| `starts_with(prefix)` | `startsWith(s, prefix)` | Check if starts with | `ds['name'].str.starts_with('Mr.')` |
| `ends_with(suffix)` | `endsWith(s, suffix)` | Check if ends with | `ds['name'].str.ends_with('.txt')` |
| `find(sub)` | `position(s, sub)` | Find position (1-indexed) | `ds['email'].str.find('@')` |
| `match(pattern)` | `match(s, pattern)` | Regex match | `ds['email'].str.match(r'^[\w]+@')` |

### Replace & Transform

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `replace(old, new)` | `replace(s, old, new)` | Replace all occurrences | `ds['text'].str.replace('old', 'new')` |
| `replace(pat, repl, regex=True)` | `replaceRegexpAll(s, pat, repl)` | Regex replacement | `ds['text'].str.replace(r'\d+', 'X', regex=True)` |
| `reverse()` | `reverse(s)` | Reverse string | `ds['name'].str.reverse()` |
| `repeat(n)` | `repeat(s, n)` | Repeat string N times | `ds['char'].str.repeat(3)` |

### Padding

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `pad(len, fill)` | `leftPad(s, len, fill)` | Left-pad to length | `ds['id'].str.pad(5, '0')` |
| `rpad(len, fill)` | `rightPad(s, len, fill)` | Right-pad to length | `ds['name'].str.rpad(20)` |
| `zfill(width)` | `leftPad(s, width, '0')` | Zero-pad left | `ds['id'].str.zfill(5)` |
| `center(width)` | combination | Center string | `ds['name'].str.center(20)` |
| `ljust(width)` | `rightPad(s, width)` | Left justify | `ds['name'].str.ljust(20)` |
| `rjust(width)` | `leftPad(s, width)` | Right justify | `ds['name'].str.rjust(20)` |

### Splitting & Joining

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `split(sep)` | `splitByString(sep, s)` | Split into array | `ds['tags'].str.split(',')` |
| `split()` | `splitByWhitespace(s)` | Split by whitespace (default) | `ds['text'].str.split()` |
| `split(pat, regex=True)` | `splitByRegexp(pat, s)` | Split by regex pattern | `ds['text'].str.split(r'\s+', regex=True)` |
| `join_str(sep)` | `arrayStringConcat(arr, sep)` | Join array to string | `ds['arr'].str.join_str(',')` |

### Character Tests

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `isalpha()` | `match(s, '^[a-zA-Z]+$')` | All alphabetic | `ds['name'].str.isalpha()` |
| `isdigit()` | `match(s, '^[0-9]+$')` | All digits | `ds['code'].str.isdigit()` |
| `isalnum()` | `match(s, '^[a-zA-Z0-9]+$')` | Alphanumeric | `ds['id'].str.isalnum()` |
| `isspace()` | `match(s, '^\s+$')` | All whitespace | `ds['text'].str.isspace()` |
| `isupper()` | `equals(s, upper(s))` | All uppercase | `ds['code'].str.isupper()` |
| `islower()` | `equals(s, lower(s))` | All lowercase | `ds['text'].str.islower()` |

---

## DateTime Functions (`.dt` accessor)

Access via `ds['column'].dt.<property>` or `ds['column'].dt.<method>()`.

### Date Part Extraction (Properties)

| Property | ClickHouse | Description | Example |
|----------|------------|-------------|---------|
| `year` | `toYear(dt)` | Extract year | `ds['date'].dt.year` |
| `month` | `toMonth(dt)` | Extract month (1-12) | `ds['date'].dt.month` |
| `day` | `toDayOfMonth(dt)` | Extract day (1-31) | `ds['date'].dt.day` |
| `hour` | `toHour(dt)` | Extract hour (0-23) | `ds['ts'].dt.hour` |
| `minute` | `toMinute(dt)` | Extract minute (0-59) | `ds['ts'].dt.minute` |
| `second` | `toSecond(dt)` | Extract second (0-59) | `ds['ts'].dt.second` |
| `millisecond` | `toMillisecond(dt)` | Extract millisecond | `ds['ts'].dt.millisecond` |
| `microsecond` | `toMicrosecond(dt)` | Extract microsecond | `ds['ts'].dt.microsecond` |
| `quarter` | `toQuarter(dt)` | Extract quarter (1-4) | `ds['date'].dt.quarter` |
| `day_of_week` | `toDayOfWeek(dt)` | Day of week (1=Mon) | `ds['date'].dt.day_of_week` |
| `day_of_year` | `toDayOfYear(dt)` | Day of year (1-366) | `ds['date'].dt.day_of_year` |
| `week` | `toWeek(dt)` | Week number | `ds['date'].dt.week` |

### Date Truncation (Methods)

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `to_start_of_day()` | `toStartOfDay(dt)` | Truncate to day start | `ds['ts'].dt.to_start_of_day()` |
| `to_start_of_week()` | `toStartOfWeek(dt)` | Truncate to week start | `ds['date'].dt.to_start_of_week()` |
| `to_start_of_month()` | `toStartOfMonth(dt)` | Truncate to month start | `ds['date'].dt.to_start_of_month()` |
| `to_start_of_quarter()` | `toStartOfQuarter(dt)` | Truncate to quarter start | `ds['date'].dt.to_start_of_quarter()` |
| `to_start_of_year()` | `toStartOfYear(dt)` | Truncate to year start | `ds['date'].dt.to_start_of_year()` |
| `to_start_of_hour()` | `toStartOfHour(dt)` | Truncate to hour start | `ds['ts'].dt.to_start_of_hour()` |
| `to_start_of_minute()` | `toStartOfMinute(dt)` | Truncate to minute start | `ds['ts'].dt.to_start_of_minute()` |

### Date Arithmetic

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `add_years(n)` | `addYears(dt, n)` | Add years | `ds['date'].dt.add_years(1)` |
| `add_months(n)` | `addMonths(dt, n)` | Add months | `ds['date'].dt.add_months(3)` |
| `add_weeks(n)` | `addWeeks(dt, n)` | Add weeks | `ds['date'].dt.add_weeks(2)` |
| `add_days(n)` | `addDays(dt, n)` | Add days | `ds['date'].dt.add_days(7)` |
| `add_hours(n)` | `addHours(dt, n)` | Add hours | `ds['ts'].dt.add_hours(24)` |
| `add_minutes(n)` | `addMinutes(dt, n)` | Add minutes | `ds['ts'].dt.add_minutes(30)` |
| `add_seconds(n)` | `addSeconds(dt, n)` | Add seconds | `ds['ts'].dt.add_seconds(60)` |
| `subtract_years(n)` | `subtractYears(dt, n)` | Subtract years | `ds['date'].dt.subtract_years(1)` |
| `subtract_months(n)` | `subtractMonths(dt, n)` | Subtract months | `ds['date'].dt.subtract_months(1)` |
| `subtract_days(n)` | `subtractDays(dt, n)` | Subtract days | `ds['date'].dt.subtract_days(7)` |

### Date Checks

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `is_month_start()` | `equals(toDayOfMonth(dt), 1)` | First day of month | `ds['date'].dt.is_month_start()` |
| `is_month_end()` | computed | Last day of month | `ds['date'].dt.is_month_end()` |
| `is_quarter_start()` | computed | First day of quarter | `ds['date'].dt.is_quarter_start()` |
| `is_quarter_end()` | computed | Last day of quarter | `ds['date'].dt.is_quarter_end()` |
| `is_year_start()` | computed | First day of year | `ds['date'].dt.is_year_start()` |
| `is_year_end()` | computed | Last day of year | `ds['date'].dt.is_year_end()` |
| `is_leap_year()` | computed | Check leap year | `ds['date'].dt.is_leap_year()` |

### Formatting & Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `strftime(fmt)` | `formatDateTime(dt, fmt)` | Format as string | `ds['date'].dt.strftime('%Y-%m-%d')` |
| `tz_convert(tz)` | `toTimezone(dt, tz)` | Convert timezone | `ds['ts'].dt.tz_convert('UTC')` |
| `total_seconds()` | `toUnixTimestamp(dt)` | To Unix timestamp | `ds['ts'].dt.total_seconds()` |

---

## Array Functions (`.arr` accessor)

Access via `ds['column'].arr.<function>()`.

### Properties

| Property | ClickHouse | Description | Example |
|----------|------------|-------------|---------|
| `length` | `length(arr)` | Array length | `ds['tags'].arr.length` |
| `size` | `length(arr)` | Alias for length | `ds['tags'].arr.size` |
| `empty` | `empty(arr)` | Check if empty | `ds['tags'].arr.empty` |
| `not_empty` | `notEmpty(arr)` | Check if not empty | `ds['tags'].arr.not_empty` |

### Element Access

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_first()` | `arrayElement(arr, 1)` | Get first element | `ds['arr'].arr.array_first()` |
| `array_last()` | `arrayElement(arr, -1)` | Get last element | `ds['arr'].arr.array_last()` |
| `array_element(n)` | `arrayElement(arr, n)` | Get nth element | `ds['arr'].arr.array_element(3)` |
| `array_slice(offset, len)` | `arraySlice(arr, offset, len)` | Slice array | `ds['arr'].arr.array_slice(1, 3)` |

### Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_sum()` | `arraySum(arr)` | Sum of elements | `ds['nums'].arr.array_sum()` |
| `array_avg()` | `arrayAvg(arr)` | Average of elements | `ds['nums'].arr.array_avg()` |
| `array_min()` | `arrayMin(arr)` | Minimum element | `ds['nums'].arr.array_min()` |
| `array_max()` | `arrayMax(arr)` | Maximum element | `ds['nums'].arr.array_max()` |
| `array_product()` | `arrayProduct(arr)` | Product of elements | `ds['nums'].arr.array_product()` |

### Transformations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_sort()` | `arraySort(arr)` | Sort array | `ds['arr'].arr.array_sort()` |
| `array_reverse_sort()` | `arrayReverseSort(arr)` | Sort descending | `ds['arr'].arr.array_reverse_sort()` |
| `array_reverse()` | `arrayReverse(arr)` | Reverse array | `ds['arr'].arr.array_reverse()` |
| `array_distinct()` | `arrayDistinct(arr)` | Unique elements | `ds['arr'].arr.array_distinct()` |
| `array_compact()` | `arrayCompact(arr)` | Remove consecutive duplicates | `ds['arr'].arr.array_compact()` |
| `array_flatten()` | `arrayFlatten(arr)` | Flatten nested array | `ds['arr'].arr.array_flatten()` |

### Modification

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_push_back(elem)` | `arrayPushBack(arr, elem)` | Add to end | `ds['arr'].arr.array_push_back('x')` |
| `array_push_front(elem)` | `arrayPushFront(arr, elem)` | Add to front | `ds['arr'].arr.array_push_front('x')` |
| `array_pop_back()` | `arrayPopBack(arr)` | Remove last | `ds['arr'].arr.array_pop_back()` |
| `array_pop_front()` | `arrayPopFront(arr)` | Remove first | `ds['arr'].arr.array_pop_front()` |
| `array_concat(other)` | `arrayConcat(arr1, arr2)` | Concatenate | `ds['arr'].arr.array_concat(other)` |

### Search & Count

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `has(elem)` | `has(arr, elem)` | Check if contains | `ds['tags'].arr.has('python')` |
| `index_of(elem)` | `indexOf(arr, elem)` | Find index | `ds['arr'].arr.index_of('x')` |
| `count_equal(elem)` | `countEqual(arr, elem)` | Count occurrences | `ds['arr'].arr.count_equal('x')` |
| `array_uniq()` | `arrayUniq(arr)` | Count unique | `ds['arr'].arr.array_uniq()` |

### Higher-Order Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_map(lambda)` | `arrayMap(lambda, arr)` | Apply function | `ds['arr'].arr.array_map(lambda)` |
| `array_filter(lambda)` | `arrayFilter(lambda, arr)` | Filter elements | `ds['arr'].arr.array_filter(lambda)` |
| `array_exists(lambda)` | `arrayExists(lambda, arr)` | Any match | `ds['arr'].arr.array_exists(lambda)` |
| `array_all(lambda)` | `arrayAll(lambda, arr)` | All match | `ds['arr'].arr.array_all(lambda)` |

### Cumulative

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_cum_sum()` | `arrayCumSum(arr)` | Cumulative sum | `ds['nums'].arr.array_cum_sum()` |
| `array_difference()` | `arrayDifference(arr)` | Consecutive differences | `ds['nums'].arr.array_difference()` |

### String Operations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `array_string_concat(sep)` | `arrayStringConcat(arr, sep)` | Join to string | `ds['arr'].arr.array_string_concat(',')` |

---

## JSON Functions (`.json` accessor)

Access via `ds['column'].json.<function>()`.

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `json_extract_string(path)` | `JSONExtractString(j, path)` | Extract string | `ds['data'].json.json_extract_string('name')` |
| `json_extract_int(path)` | `JSONExtractInt(j, path)` | Extract integer | `ds['data'].json.json_extract_int('age')` |
| `json_extract_float(path)` | `JSONExtractFloat(j, path)` | Extract float | `ds['data'].json.json_extract_float('price')` |
| `json_extract_bool(path)` | `JSONExtractBool(j, path)` | Extract boolean | `ds['data'].json.json_extract_bool('active')` |
| `json_extract_raw(path)` | `JSONExtractRaw(j, path)` | Extract raw JSON | `ds['data'].json.json_extract_raw('nested')` |
| `json_extract_keys()` | `JSONExtractKeys(j)` | Get keys | `ds['data'].json.json_extract_keys()` |
| `json_type(path)` | `JSONType(j, path)` | Get type | `ds['data'].json.json_type('field')` |
| `json_length(path)` | `JSONLength(j, path)` | Get length | `ds['data'].json.json_length('items')` |
| `json_has(key)` | `JSONHas(j, key)` | Check key exists | `ds['data'].json.json_has('name')` |
| `is_valid_json()` | `isValidJSON(s)` | Validate JSON | `ds['text'].json.is_valid_json()` |
| `to_json_string()` | `toJSONString(x)` | Convert to JSON | `ds['obj'].json.to_json_string()` |

---

## URL Functions (`.url` accessor)

Access via `ds['column'].url.<function>()`.

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `domain()` | `domain(url)` | Extract domain | `ds['link'].url.domain()` |
| `domain_without_www()` | `domainWithoutWWW(url)` | Domain without www | `ds['link'].url.domain_without_www()` |
| `top_level_domain()` | `topLevelDomain(url)` | Extract TLD | `ds['link'].url.top_level_domain()` |
| `protocol()` | `protocol(url)` | Extract protocol | `ds['link'].url.protocol()` |
| `url_path()` | `path(url)` | Extract path | `ds['link'].url.url_path()` |
| `path_full()` | `pathFull(url)` | Full path with query | `ds['link'].url.path_full()` |
| `query_string()` | `queryString(url)` | Extract query string | `ds['link'].url.query_string()` |
| `fragment()` | `fragment(url)` | Extract fragment | `ds['link'].url.fragment()` |
| `url_port()` | `port(url)` | Extract port | `ds['link'].url.url_port()` |
| `extract_url_parameter(name)` | `extractURLParameter(url, n)` | Get query param | `ds['link'].url.extract_url_parameter('id')` |
| `extract_url_parameters()` | `extractURLParameters(url)` | All params as array | `ds['link'].url.extract_url_parameters()` |
| `cut_url_parameter(name)` | `cutURLParameter(url, n)` | Remove param | `ds['link'].url.cut_url_parameter('utm_source')` |
| `decode_url_component()` | `decodeURLComponent(s)` | URL decode | `ds['text'].url.decode_url_component()` |
| `encode_url_component()` | `encodeURLComponent(s)` | URL encode | `ds['text'].url.encode_url_component()` |

---

## IP Address Functions (`.ip` accessor)

Access via `ds['column'].ip.<function>()`.

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `to_ipv4()` | `toIPv4(s)` | Convert to IPv4 | `ds['ip'].ip.to_ipv4()` |
| `to_ipv6()` | `toIPv6(s)` | Convert to IPv6 | `ds['ip'].ip.to_ipv6()` |
| `ipv4_num_to_string()` | `IPv4NumToString(n)` | Number to string | `ds['ip_num'].ip.ipv4_num_to_string()` |
| `ipv4_string_to_num()` | `IPv4StringToNum(s)` | String to number | `ds['ip'].ip.ipv4_string_to_num()` |
| `ipv6_num_to_string()` | `IPv6NumToString(n)` | IPv6 num to string | `ds['ip_num'].ip.ipv6_num_to_string()` |
| `ipv4_to_ipv6()` | `IPv4ToIPv6(ip)` | IPv4 to IPv6 | `ds['ip4'].ip.ipv4_to_ipv6()` |
| `is_ipv4_string()` | `isIPv4String(s)` | Validate IPv4 | `ds['ip'].ip.is_ipv4_string()` |
| `is_ipv6_string()` | `isIPv6String(s)` | Validate IPv6 | `ds['ip'].ip.is_ipv6_string()` |
| `ipv4_cidr_to_range(cidr)` | `IPv4CIDRToRange(ip, cidr)` | CIDR to range | `ds['ip'].ip.ipv4_cidr_to_range(24)` |

---

## Geo/Distance Functions (`.geo` accessor)

Access via `ds['column'].geo.<function>()` or `F.<function>()`.

### Distance Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `great_circle_distance(...)` | `greatCircleDistance(...)` | Great circle distance | `F.great_circle_distance(lon1, lat1, lon2, lat2)` |
| `geo_distance(...)` | `geoDistance(...)` | WGS-84 distance | `F.geo_distance(lon1, lat1, lon2, lat2)` |
| `l1_distance(v1, v2)` | `L1Distance(v1, v2)` | Manhattan distance | `F.l1_distance(vec1, vec2)` |
| `l2_distance(v1, v2)` | `L2Distance(v1, v2)` | Euclidean distance | `F.l2_distance(vec1, vec2)` |
| `l2_squared_distance(v1, v2)` | `L2SquaredDistance(v1, v2)` | Squared Euclidean | `F.l2_squared_distance(vec1, vec2)` |
| `linf_distance(v1, v2)` | `LinfDistance(v1, v2)` | Chebyshev distance | `F.linf_distance(vec1, vec2)` |
| `cosine_distance(v1, v2)` | `cosineDistance(v1, v2)` | Cosine distance | `F.cosine_distance(vec1, vec2)` |

### Vector Operations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `dot_product(v1, v2)` | `dotProduct(v1, v2)` | Dot product | `F.dot_product(vec1, vec2)` |
| `l2_norm(vec)` | `L2Norm(vec)` | Vector norm | `F.l2_norm(vec)` |
| `l2_normalize(vec)` | `L2Normalize(vec)` | Normalize vector | `F.l2_normalize(vec)` |

### H3 Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `geo_to_h3(lon, lat, res)` | `geoToH3(lon, lat, res)` | Geo to H3 index | `F.geo_to_h3(lon, lat, 9)` |
| `h3_to_geo(h3)` | `h3ToGeo(h3)` | H3 to geo coords | `F.h3_to_geo(h3_index)` |

### Point Operations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `point_in_polygon(pt, poly)` | `pointInPolygon(pt, poly)` | Point in polygon | `F.point_in_polygon(point, polygon)` |
| `point_in_ellipses(...)` | `pointInEllipses(...)` | Point in ellipses | `F.point_in_ellipses(x, y, ...)` |

---

## Aggregate Functions

Access via `ds['column'].<method>()` or `F.<method>()`. Used with `groupby()`.

### Basic Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `sum()` | `sum(x)` | Sum | `ds['amount'].sum()` |
| `avg()` / `mean()` | `avg(x)` | Average | `ds['price'].avg()` |
| `count()` | `count(x)` | Count | `ds['id'].count()` |
| `min()` | `min(x)` | Minimum | `ds['price'].min()` |
| `max()` | `max(x)` | Maximum | `ds['price'].max()` |
| `count_distinct()` | `uniq(x)` | Approximate distinct | `ds['user_id'].count_distinct()` |
| `uniq_exact()` | `uniqExact(x)` | Exact distinct count | `F.uniq_exact(Field('id'))` |
| `uniq_combined()` | `uniqCombined(x)` | HyperLogLog++ | `F.uniq_combined(Field('id'))` |

### Statistical Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `stddev()` | `stddevPop(x)` | Std deviation (pop) | `ds['value'].stddev()` |
| `stddev_samp()` | `stddevSamp(x)` | Std deviation (sample) | `ds['value'].stddev_samp()` |
| `variance()` | `varPop(x)` | Variance (pop) | `ds['value'].variance()` |
| `var_samp()` | `varSamp(x)` | Variance (sample) | `ds['value'].var_samp()` |
| `median()` | `median(x)` | Median | `ds['value'].median()` |
| `quantile(level)` | `quantile(level)(x)` | Quantile | `F.quantile(Field('value'), 0.95)` |
| `quantiles(...)` | `quantiles(...)(x)` | Multiple quantiles | `F.quantiles(Field('value'), 0.25, 0.5, 0.75)` |
| `skew()` | `skewPop(x)` | Skewness | `ds['value'].skew()` |
| `kurt()` | `kurtPop(x)` | Kurtosis | `ds['value'].kurt()` |
| `corr(other)` | `corr(x, y)` | Correlation | `ds['x'].corr(ds['y'])` |
| `cov(other)` | `covarPop(x, y)` | Covariance | `ds['x'].cov(ds['y'])` |
| `entropy()` | `entropy(x)` | Entropy | `F.entropy(Field('value'))` |

### Conditional Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `count_if(cond)` | `countIf(cond)` | Count if condition | `F.count_if(Field('status') == 1)` |
| `sum_if(x, cond)` | `sumIf(x, cond)` | Sum if condition | `F.sum_if(Field('amount'), Field('valid'))` |
| `avg_if(x, cond)` | `avgIf(x, cond)` | Avg if condition | `F.avg_if(Field('price'), Field('in_stock'))` |
| `min_if(x, cond)` | `minIf(x, cond)` | Min if condition | `F.min_if(Field('value'), Field('active'))` |
| `max_if(x, cond)` | `maxIf(x, cond)` | Max if condition | `F.max_if(Field('value'), Field('active'))` |

### Collection Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `group_array()` | `groupArray(x)` | Collect to array | `ds['id'].group_array()` |
| `group_uniq_array()` | `groupUniqArray(x)` | Unique to array | `ds['category'].group_uniq_array()` |
| `group_concat(sep)` | `groupConcat(x, sep)` | Concat with separator | `F.group_concat(Field('name'), ',')` |
| `top_k(k)` | `topK(k)(x)` | Top K values | `F.top_k(Field('item'), 10)` |
| `histogram(bins)` | `histogram(bins)(x)` | Build histogram | `F.histogram(Field('value'), 10)` |

### Arg Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `argmin(val)` | `argMin(arg, val)` | Arg at min value | `ds['name'].argmin(ds['price'])` |
| `argmax(val)` | `argMax(arg, val)` | Arg at max value | `ds['name'].argmax(ds['score'])` |
| `any_value()` | `any(x)` | Any value | `ds['category'].any_value()` |
| `any_last()` | `anyLast(x)` | Last value | `ds['status'].any_last()` |

### Weighted Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `avg_weighted(weight)` | `avgWeighted(x, w)` | Weighted average | `F.avg_weighted(Field('price'), Field('qty'))` |
| `top_k_weighted(w, k)` | `topKWeighted(k)(x, w)` | Weighted top K | `F.top_k_weighted(Field('item'), Field('count'), 10)` |

### Bit Aggregations

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `group_bit_and()` | `groupBitAnd(x)` | Bitwise AND | `F.group_bit_and(Field('flags'))` |
| `group_bit_or()` | `groupBitOr(x)` | Bitwise OR | `F.group_bit_or(Field('flags'))` |
| `group_bit_xor()` | `groupBitXor(x)` | Bitwise XOR | `F.group_bit_xor(Field('flags'))` |

### Regression

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `simple_linear_regression(x, y)` | `simpleLinearRegression(x, y)` | Linear regression | `F.simple_linear_regression(Field('x'), Field('y'))` |

---

## Window Functions

Access via `ds['column'].<method>()` or `F.<method>()`. Requires OVER clause.

### Ranking Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `row_number()` | `row_number()` | Row number | `F.row_number()` |
| `rank()` | `rank()` | Rank with gaps | `F.rank()` |
| `dense_rank()` | `dense_rank()` | Rank without gaps | `F.dense_rank()` |
| `ntile(n)` | `ntile(n)` | N-tile bucket | `F.ntile(4)` |
| `percent_rank()` | `percent_rank()` | Relative rank (0-1) | `F.percent_rank()` |
| `cume_dist()` | `cume_dist()` | Cumulative distribution | `F.cume_dist()` |

### Value Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `lag(offset, default)` | `lagInFrame(x, n, d)` | Previous row value | `ds['value'].lag(1)` |
| `lead(offset, default)` | `leadInFrame(x, n, d)` | Next row value | `ds['value'].lead(1)` |
| `first_value()` | `first_value(x)` | First in window | `ds['value'].first_value()` |
| `last_value()` | `last_value(x)` | Last in window | `ds['value'].last_value()` |
| `nth_value(n)` | `nth_value(x, n)` | Nth value | `F.nth_value(Field('value'), 3)` |

### Cumulative Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `cumsum()` | window sum | Cumulative sum | `ds['value'].cumsum()` |
| `cummax()` | window max | Cumulative max | `ds['value'].cummax()` |
| `cummin()` | window min | Cumulative min | `ds['value'].cummin()` |

### Shift Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `shift(n)` | `lagInFrame`/`leadInFrame` | Shift values | `ds['value'].shift(1)` |
| `diff(n)` | `x - lagInFrame(x, n)` | Difference | `ds['value'].diff()` |
| `pct_change(n)` | `(x - lag) / lag` | Percent change | `ds['value'].pct_change()` |

---

## Math Functions

Access via `ds['column'].<method>()` or `F.<method>()`.

### Basic Math

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `abs()` | `abs(x)` | Absolute value | `ds['value'].abs()` |
| `round(n)` | `round(x, n)` | Round to N decimals | `ds['price'].round(2)` |
| `floor()` | `floor(x)` | Round down | `ds['value'].floor()` |
| `ceil()` | `ceiling(x)` | Round up | `ds['value'].ceil()` |
| `sign()` | `sign(x)` | Sign (-1, 0, 1) | `ds['value'].sign()` |
| `mod(b)` | `modulo(a, b)` | Modulo | `F.mod(Field('a'), Field('b'))` |

### Powers & Roots

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `sqrt()` | `sqrt(x)` | Square root | `ds['value'].sqrt()` |
| `cbrt()` | `cbrt(x)` | Cube root | `ds['value'].cbrt()` |
| `pow(n)` | `pow(x, n)` | Power | `ds['value'].pow(2)` |

### Logarithms & Exponentials

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `exp()` | `exp(x)` | e^x | `ds['value'].exp()` |
| `log()` / `ln()` | `log(x)` | Natural log | `ds['value'].log()` |
| `log10()` | `log10(x)` | Base-10 log | `ds['value'].log10()` |
| `log2()` | `log2(x)` | Base-2 log | `ds['value'].log2()` |

### Trigonometry

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `sin()` | `sin(x)` | Sine | `ds['angle'].sin()` |
| `cos()` | `cos(x)` | Cosine | `ds['angle'].cos()` |
| `tan()` | `tan(x)` | Tangent | `ds['angle'].tan()` |
| `asin()` | `asin(x)` | Arc sine | `ds['value'].asin()` |
| `acos()` | `acos(x)` | Arc cosine | `ds['value'].acos()` |
| `atan()` | `atan(x)` | Arc tangent | `ds['value'].atan()` |
| `atan2(y, x)` | `atan2(y, x)` | Arc tangent of y/x | `F.atan2(Field('y'), Field('x'))` |

### Hyperbolic

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `sinh()` | `sinh(x)` | Hyperbolic sine | `ds['value'].sinh()` |
| `cosh()` | `cosh(x)` | Hyperbolic cosine | `ds['value'].cosh()` |
| `tanh()` | `tanh(x)` | Hyperbolic tangent | `ds['value'].tanh()` |

### Angle Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `degrees()` | `degrees(x)` | Radians to degrees | `ds['rad'].degrees()` |
| `radians()` | `radians(x)` | Degrees to radians | `ds['deg'].radians()` |

### Special Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `erf()` | `erf(x)` | Error function | `ds['value'].erf()` |
| `erfc()` | `erfc(x)` | Complementary error | `ds['value'].erfc()` |
| `lgamma()` | `lgamma(x)` | Log-gamma | `ds['value'].lgamma()` |
| `tgamma()` | `tgamma(x)` | Gamma function | `ds['value'].tgamma()` |

### Comparison

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `greatest(...)` | `greatest(a, b, ...)` | Maximum value | `F.greatest(Field('a'), Field('b'), Field('c'))` |
| `least(...)` | `least(a, b, ...)` | Minimum value | `F.least(Field('a'), Field('b'), Field('c'))` |
| `clip(lower, upper)` | combination | Clip to range | `ds['value'].clip(0, 100)` |

### Random

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `rand()` | `rand()` | Random UInt32 | `F.rand()` |
| `rand64()` | `rand64()` | Random UInt64 | `F.rand64()` |
| `rand_uniform(min, max)` | `randUniform(min, max)` | Uniform random | `F.rand_uniform(0, 1)` |
| `rand_normal(mean, std)` | `randNormal(mean, std)` | Normal random | `F.rand_normal(0, 1)` |

---

## Hash Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `md5()` | `MD5(s)` | MD5 hash | `ds['data'].md5()` |
| `sha256()` | `SHA256(s)` | SHA256 hash | `ds['data'].sha256()` |
| `city_hash64()` | `cityHash64(s)` | CityHash64 (fast) | `ds['data'].city_hash64()` |
| `sip_hash64()` | `sipHash64(s)` | SipHash64 | `ds['data'].sip_hash64()` |

---

## Encoding Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `hex()` | `hex(x)` | To hexadecimal | `ds['data'].hex()` |
| `unhex()` | `unhex(s)` | From hexadecimal | `ds['hex'].unhex()` |
| `bin()` | `bin(x)` | To binary string | `ds['num'].bin()` |
| `unbin()` | `unbin(s)` | From binary string | `ds['binary'].unbin()` |
| `base64_encode()` | `base64Encode(s)` | Encode Base64 | `ds['data'].base64_encode()` |
| `base64_decode()` | `base64Decode(s)` | Decode Base64 | `ds['encoded'].base64_decode()` |
| `bit_count()` | `bitCount(x)` | Count set bits | `ds['flags'].bit_count()` |

---

## UUID Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `generate_uuid_v4()` | `generateUUIDv4()` | Random UUID v4 | `F.generate_uuid_v4()` |
| `generate_uuid_v7()` | `generateUUIDv7()` | Time-ordered UUID v7 | `F.generate_uuid_v7()` |
| `to_uuid()` | `toUUID(s)` | String to UUID | `ds['id'].to_uuid()` |
| `uuid_to_num()` | `UUIDToNum(uuid)` | UUID to FixedString | `ds['uuid'].uuid_to_num()` |

---

## Conditional Functions

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `if_(cond, then, else)` | `if(cond, then, else)` | Conditional | `F.if_(Field('x') > 0, 'pos', 'neg')` |
| `if_null(default)` | `ifNull(x, default)` | Default if NULL | `ds['value'].if_null(0)` |
| `null_if(value)` | `nullIf(x, value)` | NULL if equals | `ds['status'].null_if('')` |
| `coalesce(...)` | `coalesce(...)` | First non-NULL | `F.coalesce(Field('a'), Field('b'), 0)` |
| `multi_if(...)` | `multiIf(...)` | Multiple conditions | `F.multi_if(c1, v1, c2, v2, default)` |
| `fillna(value)` | `ifNull(x, value)` | Fill NULL values | `ds['value'].fillna(0)` |
| `isna()` | `isNull(x)` | Check if NULL | `ds['value'].isna()` |
| `notna()` | `isNotNull(x)` | Check if not NULL | `ds['value'].notna()` |
| `where_expr(cond, other)` | `if(cond, x, other)` | Where condition | `ds['x'].where_expr(ds['x'] > 0, 0)` |
| `mask(cond, other)` | `if(cond, other, x)` | Mask where true | `ds['x'].mask(ds['x'] < 0, 0)` |

---

## Type Conversion

| Method | ClickHouse | Description | Example |
|--------|------------|-------------|---------|
| `to_string()` | `toString(x)` | Convert to String | `ds['id'].to_string()` |
| `to_int8()` | `toInt8(x)` | Convert to Int8 | `ds['val'].to_int8()` |
| `to_int16()` | `toInt16(x)` | Convert to Int16 | `ds['val'].to_int16()` |
| `to_int32()` | `toInt32(x)` | Convert to Int32 | `ds['val'].to_int32()` |
| `to_int64()` | `toInt64(x)` | Convert to Int64 | `ds['val'].to_int64()` |
| `to_float32()` | `toFloat32(x)` | Convert to Float32 | `ds['val'].to_float32()` |
| `to_float64()` | `toFloat64(x)` | Convert to Float64 | `ds['val'].to_float64()` |
| `to_date()` | `toDate(x)` | Convert to Date | `ds['str_date'].to_date()` |
| `to_datetime(tz)` | `toDateTime(x, tz)` | Convert to DateTime | `ds['str'].to_datetime()` |

---

## F Namespace

For explicit function calls when you need more control:

```python
from datastore import F, Field

# String functions
F.upper(Field('name'))
F.concat(Field('first'), ' ', Field('last'))

# Aggregate functions
F.sum(Field('amount'))
F.count_distinct(Field('user_id'))
F.quantile(Field('value'), 0.95)
F.top_k(Field('item'), 10)

# Conditional functions
F.if_(Field('age') > 18, 'adult', 'minor')
F.coalesce(Field('a'), Field('b'), 0)

# Array functions
F.array_sum(Field('numbers'))
F.array_join(Field('tags'))

# Geo functions
F.geo_distance(lon1, lat1, lon2, lat2)
F.cosine_distance(vec1, vec2)

# Date/Time functions
F.now()
F.today()
F.date_diff('day', Field('start'), Field('end'))

# Random
F.rand()
F.rand_normal(0, 1)

# UUID
F.generate_uuid_v4()
```

---

## ClickHouse Function Reference

For the complete list of ClickHouse functions, see:
- [String Functions](https://clickhouse.com/docs/en/sql-reference/functions/string-functions)
- [Date/Time Functions](https://clickhouse.com/docs/en/sql-reference/functions/date-time-functions)
- [Array Functions](https://clickhouse.com/docs/en/sql-reference/functions/array-functions)
- [JSON Functions](https://clickhouse.com/docs/en/sql-reference/functions/json-functions)
- [URL Functions](https://clickhouse.com/docs/en/sql-reference/functions/url-functions)
- [IP Functions](https://clickhouse.com/docs/en/sql-reference/functions/ip-address-functions)
- [Geo Functions](https://clickhouse.com/docs/en/sql-reference/functions/geo/)
- [Math Functions](https://clickhouse.com/docs/en/sql-reference/functions/math-functions)
- [Aggregate Functions](https://clickhouse.com/docs/en/sql-reference/aggregate-functions)
- [Window Functions](https://clickhouse.com/docs/en/sql-reference/window-functions)
