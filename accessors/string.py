"""
StringAccessor - String functions via .str accessor.

Provides ClickHouse string functions in a Pandas-like API.
Maps to ClickHouse string functions: https://clickhouse.com/docs/en/sql-reference/functions/string-functions
"""

from typing import TYPE_CHECKING

from .base import BaseAccessor

if TYPE_CHECKING:
    from ..functions import Function


class StringAccessor(BaseAccessor):
    """
    Accessor for string functions via .str property.

    Maps to ClickHouse string functions with a Pandas-like interface.

    Example:
        >>> ds['name'].str.upper()           # upper(name)
        >>> ds['name'].str.length()          # length(name)
        >>> ds['name'].str.substring(1, 5)   # substring(name, 1, 5)
        >>> ds['name'].str.replace('a', 'b') # replace(name, 'a', 'b')
        >>> ds['name'].str.concat(' suffix') # concat(name, ' suffix')

    ClickHouse String Functions Reference:
        https://clickhouse.com/docs/en/sql-reference/functions/string-functions
    """

    # ========== Case Conversion ==========

    def upper(self, alias: str = None) -> 'Function':
        """
        Convert string to uppercase.

        Maps to ClickHouse: upper(s)

        Returns:
            Function expression for upper(expr)

        Example:
            >>> ds['name'].str.upper()
            >>> # SQL: upper("name")
        """
        return self._create_function('upper', alias=alias)

    def lower(self, alias: str = None) -> 'Function':
        """
        Convert string to lowercase.

        Maps to ClickHouse: lower(s)

        Returns:
            Function expression for lower(expr)

        Example:
            >>> ds['name'].str.lower()
            >>> # SQL: lower("name")
        """
        return self._create_function('lower', alias=alias)

    def capitalize(self, alias: str = None) -> 'Function':
        """
        Convert first character to uppercase, rest to lowercase.

        Maps to ClickHouse: concat(upper(substring(s, 1, 1)), lower(substring(s, 2)))

        Note: ClickHouse doesn't have a direct capitalize function,
        so we use a combination of functions.

        Returns:
            Function expression

        Example:
            >>> ds['name'].str.capitalize()
        """
        # Use initcap if available, otherwise compose
        return self._create_function('initcap', alias=alias)

    # ========== Length and Size ==========

    def length(self, alias: str = None) -> 'Function':
        """
        Get length of string in bytes.

        Maps to ClickHouse: length(s)

        Note: For UTF-8 character count, use char_length().

        Returns:
            Function expression for length(expr)

        Example:
            >>> ds['name'].str.length()
            >>> # SQL: length("name")
        """
        return self._create_function('length', alias=alias)

    def char_length(self, alias: str = None) -> 'Function':
        """
        Get length of string in Unicode code points.

        Maps to ClickHouse: char_length(s)

        Returns:
            Function expression for char_length(expr)

        Example:
            >>> ds['name'].str.char_length()
            >>> # SQL: char_length("name")
        """
        return self._create_function('char_length', alias=alias)

    def len(self, alias: str = None) -> 'Function':
        """Alias for length(). Get length of string in bytes."""
        return self.length(alias=alias)

    def empty(self, alias: str = None) -> 'Function':
        """
        Check if string is empty.

        Maps to ClickHouse: empty(s)

        Returns:
            Function expression for empty(expr) - returns 1 if empty, 0 otherwise

        Example:
            >>> ds['name'].str.empty()
            >>> # SQL: empty("name")
        """
        return self._create_function('empty', alias=alias)

    def not_empty(self, alias: str = None) -> 'Function':
        """
        Check if string is not empty.

        Maps to ClickHouse: notEmpty(s)

        Returns:
            Function expression for notEmpty(expr)

        Example:
            >>> ds['name'].str.not_empty()
            >>> # SQL: notEmpty("name")
        """
        return self._create_function('notEmpty', alias=alias)

    # ========== Substring and Slicing ==========

    def substring(self, offset: int, length: int = None, alias: str = None) -> 'Function':
        """
        Extract a substring.

        Maps to ClickHouse: substring(s, offset, length)

        Args:
            offset: Starting position (1-indexed in ClickHouse)
            length: Number of characters to extract (optional)

        Returns:
            Function expression for substring(expr, offset, length)

        Example:
            >>> ds['name'].str.substring(1, 3)  # First 3 characters
            >>> # SQL: substring("name", 1, 3)
        """
        if length is not None:
            return self._create_function('substring', offset, length, alias=alias)
        return self._create_function('substring', offset, alias=alias)

    def substr(self, offset: int, length: int = None, alias: str = None) -> 'Function':
        """Alias for substring()."""
        return self.substring(offset, length, alias=alias)

    def left(self, length: int, alias: str = None) -> 'Function':
        """
        Get leftmost N characters.

        Maps to ClickHouse: left(s, n)

        Args:
            length: Number of characters from left

        Returns:
            Function expression for left(expr, length)

        Example:
            >>> ds['name'].str.left(3)
            >>> # SQL: left("name", 3)
        """
        return self._create_function('left', length, alias=alias)

    def right(self, length: int, alias: str = None) -> 'Function':
        """
        Get rightmost N characters.

        Maps to ClickHouse: right(s, n)

        Args:
            length: Number of characters from right

        Returns:
            Function expression for right(expr, length)

        Example:
            >>> ds['name'].str.right(3)
            >>> # SQL: right("name", 3)
        """
        return self._create_function('right', length, alias=alias)

    # ========== Trimming ==========

    def trim(self, alias: str = None) -> 'Function':
        """
        Remove leading and trailing whitespace.

        Maps to ClickHouse: trim(s)

        Returns:
            Function expression for trim(expr)

        Example:
            >>> ds['name'].str.trim()
            >>> # SQL: trim("name")
        """
        return self._create_function('trim', alias=alias)

    def ltrim(self, alias: str = None) -> 'Function':
        """
        Remove leading whitespace.

        Maps to ClickHouse: trimLeft(s)

        Returns:
            Function expression for trimLeft(expr)

        Example:
            >>> ds['name'].str.ltrim()
            >>> # SQL: trimLeft("name")
        """
        return self._create_function('trimLeft', alias=alias)

    def rtrim(self, alias: str = None) -> 'Function':
        """
        Remove trailing whitespace.

        Maps to ClickHouse: trimRight(s)

        Returns:
            Function expression for trimRight(expr)

        Example:
            >>> ds['name'].str.rtrim()
            >>> # SQL: trimRight("name")
        """
        return self._create_function('trimRight', alias=alias)

    def strip(self, alias: str = None) -> 'Function':
        """Alias for trim(). Remove leading and trailing whitespace."""
        return self.trim(alias=alias)

    def lstrip(self, alias: str = None) -> 'Function':
        """Alias for ltrim(). Remove leading whitespace."""
        return self.ltrim(alias=alias)

    def rstrip(self, alias: str = None) -> 'Function':
        """Alias for rtrim(). Remove trailing whitespace."""
        return self.rtrim(alias=alias)

    # ========== Search and Match ==========

    def contains(self, pattern: str, alias: str = None) -> 'Function':
        """
        Check if string contains pattern.

        Maps to ClickHouse: position(s, pattern) > 0

        Args:
            pattern: Substring to search for

        Returns:
            Function expression that evaluates to 1 if found, 0 otherwise

        Example:
            >>> ds['name'].str.contains('test')
            >>> # SQL: position("name", 'test') > 0
        """
        from ..expressions import Literal

        pos_func = self._create_function('position', pattern)
        return pos_func > Literal(0)

    def startswith(self, prefix: str, alias: str = None) -> 'Function':
        """
        Check if string starts with prefix.

        Maps to ClickHouse: startsWith(s, prefix)

        Args:
            prefix: Prefix to check

        Returns:
            Function expression for startsWith(expr, prefix)

        Example:
            >>> ds['name'].str.startswith('Mr.')
            >>> # SQL: startsWith("name", 'Mr.')
        """
        return self._create_function('startsWith', prefix, alias=alias)

    def endswith(self, suffix: str, alias: str = None) -> 'Function':
        """
        Check if string ends with suffix.

        Maps to ClickHouse: endsWith(s, suffix)

        Args:
            suffix: Suffix to check

        Returns:
            Function expression for endsWith(expr, suffix)

        Example:
            >>> ds['name'].str.endswith('.txt')
            >>> # SQL: endsWith("name", '.txt')
        """
        return self._create_function('endsWith', suffix, alias=alias)

    def position(self, needle: str, alias: str = None) -> 'Function':
        """
        Find position of substring (1-indexed, 0 if not found).

        Maps to ClickHouse: position(s, needle)

        Args:
            needle: Substring to find

        Returns:
            Function expression for position(expr, needle)

        Example:
            >>> ds['name'].str.position('@')
            >>> # SQL: position("name", '@')
        """
        return self._create_function('position', needle, alias=alias)

    def find(self, needle: str, alias: str = None) -> 'Function':
        """Alias for position(). Find position of substring."""
        return self.position(needle, alias=alias)

    # ========== Replace and Transform ==========

    def replace(self, pattern: str, replacement: str, alias: str = None) -> 'Function':
        """
        Replace all occurrences of pattern with replacement.

        Maps to ClickHouse: replace(s, from, to)

        Args:
            pattern: Pattern to replace
            replacement: Replacement string

        Returns:
            Function expression for replace(expr, pattern, replacement)

        Example:
            >>> ds['name'].str.replace('old', 'new')
            >>> # SQL: replace("name", 'old', 'new')
        """
        return self._create_function('replace', pattern, replacement, alias=alias)

    def replace_one(self, pattern: str, replacement: str, alias: str = None) -> 'Function':
        """
        Replace first occurrence of pattern with replacement.

        Maps to ClickHouse: replaceOne(s, from, to)

        Args:
            pattern: Pattern to replace
            replacement: Replacement string

        Returns:
            Function expression for replaceOne(expr, pattern, replacement)

        Example:
            >>> ds['name'].str.replace_one('old', 'new')
            >>> # SQL: replaceOne("name", 'old', 'new')
        """
        return self._create_function('replaceOne', pattern, replacement, alias=alias)

    def replace_regex(self, pattern: str, replacement: str, alias: str = None) -> 'Function':
        """
        Replace using regular expression.

        Maps to ClickHouse: replaceRegexpAll(s, pattern, replacement)

        Args:
            pattern: Regex pattern
            replacement: Replacement string (can use \\1, \\2 for groups)

        Returns:
            Function expression for replaceRegexpAll(expr, pattern, replacement)

        Example:
            >>> ds['text'].str.replace_regex(r'\\d+', 'NUM')
            >>> # SQL: replaceRegexpAll("text", '\\d+', 'NUM')
        """
        return self._create_function('replaceRegexpAll', pattern, replacement, alias=alias)

    def reverse(self, alias: str = None) -> 'Function':
        """
        Reverse the string.

        Maps to ClickHouse: reverse(s)

        Returns:
            Function expression for reverse(expr)

        Example:
            >>> ds['name'].str.reverse()
            >>> # SQL: reverse("name")
        """
        return self._create_function('reverse', alias=alias)

    # ========== Concatenation ==========

    def concat(self, *others, alias: str = None) -> 'Function':
        """
        Concatenate with other strings.

        Maps to ClickHouse: concat(s1, s2, ...)

        Args:
            *others: Strings or expressions to concatenate

        Returns:
            Function expression for concat(expr, others...)

        Example:
            >>> ds['first'].str.concat(' ', ds['last'])
            >>> # SQL: concat("first", ' ', "last")
        """
        from ..functions import Function
        from ..expressions import Literal, Expression

        all_args = [self._expr]
        for other in others:
            if isinstance(other, Expression):
                all_args.append(other)
            else:
                all_args.append(Literal(other))

        return Function('concat', *all_args, alias=alias)

    def pad_left(self, length: int, pad_str: str = ' ', alias: str = None) -> 'Function':
        """
        Left-pad string to specified length.

        Maps to ClickHouse: leftPad(s, length, pad_string)

        Args:
            length: Target length
            pad_str: Padding string (default: space)

        Returns:
            Function expression for leftPad(expr, length, pad_str)

        Example:
            >>> ds['id'].str.pad_left(5, '0')
            >>> # SQL: leftPad("id", 5, '0')
        """
        return self._create_function('leftPad', length, pad_str, alias=alias)

    def pad_right(self, length: int, pad_str: str = ' ', alias: str = None) -> 'Function':
        """
        Right-pad string to specified length.

        Maps to ClickHouse: rightPad(s, length, pad_string)

        Args:
            length: Target length
            pad_str: Padding string (default: space)

        Returns:
            Function expression for rightPad(expr, length, pad_str)

        Example:
            >>> ds['name'].str.pad_right(20)
            >>> # SQL: rightPad("name", 20, ' ')
        """
        return self._create_function('rightPad', length, pad_str, alias=alias)

    def zfill(self, width: int, alias: str = None) -> 'Function':
        """
        Pad string with zeros on the left to specified width.

        Args:
            width: Target width

        Returns:
            Function expression for leftPad(expr, width, '0')

        Example:
            >>> ds['id'].str.zfill(5)
            >>> # SQL: leftPad("id", 5, '0')
        """
        return self.pad_left(width, '0', alias=alias)

    # ========== Splitting ==========

    def split(self, separator: str, alias: str = None) -> 'Function':
        """
        Split string by separator into array.

        Maps to ClickHouse: splitByString(separator, s)

        Args:
            separator: Separator string

        Returns:
            Function expression for splitByString(separator, expr)

        Example:
            >>> ds['tags'].str.split(',')
            >>> # SQL: splitByString(',', "tags")
        """
        from ..functions import Function
        from ..expressions import Literal

        return Function('splitByString', Literal(separator), self._expr, alias=alias)

    def split_by_char(self, separator: str, alias: str = None) -> 'Function':
        """
        Split string by single character separator into array.

        Maps to ClickHouse: splitByChar(separator, s)

        Args:
            separator: Single character separator

        Returns:
            Function expression for splitByChar(separator, expr)

        Example:
            >>> ds['path'].str.split_by_char('/')
            >>> # SQL: splitByChar('/', "path")
        """
        from ..functions import Function
        from ..expressions import Literal

        return Function('splitByChar', Literal(separator), self._expr, alias=alias)

    # ========== Regular Expressions ==========

    def match(self, pattern: str, alias: str = None) -> 'Function':
        """
        Check if string matches regex pattern.

        Maps to ClickHouse: match(s, pattern)

        Args:
            pattern: Regex pattern

        Returns:
            Function expression for match(expr, pattern)

        Example:
            >>> ds['email'].str.match(r'^[\\w]+@[\\w]+\\.[a-z]+$')
            >>> # SQL: match("email", '^[\\w]+@[\\w]+\\.[a-z]+$')
        """
        return self._create_function('match', pattern, alias=alias)

    def extract(self, pattern: str, alias: str = None) -> 'Function':
        """
        Extract first matching group from regex.

        Maps to ClickHouse: extract(s, pattern)

        Args:
            pattern: Regex pattern with capture group

        Returns:
            Function expression for extract(expr, pattern)

        Example:
            >>> ds['url'].str.extract(r'https?://([^/]+)')
            >>> # SQL: extract("url", 'https?://([^/]+)')
        """
        return self._create_function('extract', pattern, alias=alias)

    def extract_all(self, pattern: str, alias: str = None) -> 'Function':
        """
        Extract all matches from regex into array.

        Maps to ClickHouse: extractAll(s, pattern)

        Args:
            pattern: Regex pattern

        Returns:
            Function expression for extractAll(expr, pattern)

        Example:
            >>> ds['text'].str.extract_all(r'\\d+')
            >>> # SQL: extractAll("text", '\\d+')
        """
        return self._create_function('extractAll', pattern, alias=alias)

    # ========== Encoding ==========

    def base64_encode(self, alias: str = None) -> 'Function':
        """
        Encode string to Base64.

        Maps to ClickHouse: base64Encode(s)

        Returns:
            Function expression for base64Encode(expr)

        Example:
            >>> ds['data'].str.base64_encode()
            >>> # SQL: base64Encode("data")
        """
        return self._create_function('base64Encode', alias=alias)

    def base64_decode(self, alias: str = None) -> 'Function':
        """
        Decode Base64 string.

        Maps to ClickHouse: base64Decode(s)

        Returns:
            Function expression for base64Decode(expr)

        Example:
            >>> ds['encoded'].str.base64_decode()
            >>> # SQL: base64Decode("encoded")
        """
        return self._create_function('base64Decode', alias=alias)

    def hex(self, alias: str = None) -> 'Function':
        """
        Encode string to hexadecimal.

        Maps to ClickHouse: hex(s)

        Returns:
            Function expression for hex(expr)

        Example:
            >>> ds['data'].str.hex()
            >>> # SQL: hex("data")
        """
        return self._create_function('hex', alias=alias)

    def unhex(self, alias: str = None) -> 'Function':
        """
        Decode hexadecimal string.

        Maps to ClickHouse: unhex(s)

        Returns:
            Function expression for unhex(expr)

        Example:
            >>> ds['hex_data'].str.unhex()
            >>> # SQL: unhex("hex_data")
        """
        return self._create_function('unhex', alias=alias)

    # ========== URL Functions ==========

    def url_decode(self, alias: str = None) -> 'Function':
        """
        Decode URL-encoded string.

        Maps to ClickHouse: decodeURLComponent(s)

        Returns:
            Function expression for decodeURLComponent(expr)
        """
        return self._create_function('decodeURLComponent', alias=alias)

    def url_encode(self, alias: str = None) -> 'Function':
        """
        Encode string for URL.

        Maps to ClickHouse: encodeURLComponent(s)

        Returns:
            Function expression for encodeURLComponent(expr)
        """
        return self._create_function('encodeURLComponent', alias=alias)

    def extract_url_parameter(self, param_name: str, alias: str = None) -> 'Function':
        """
        Extract URL query parameter value.

        Maps to ClickHouse: extractURLParameter(URL, name)

        Args:
            param_name: Parameter name to extract

        Returns:
            Function expression for extractURLParameter(expr, param_name)

        Example:
            >>> ds['url'].str.extract_url_parameter('id')
            >>> # SQL: extractURLParameter("url", 'id')
        """
        return self._create_function('extractURLParameter', param_name, alias=alias)

    def domain(self, alias: str = None) -> 'Function':
        """
        Extract domain from URL.

        Maps to ClickHouse: domain(url)

        Returns:
            Function expression for domain(expr)

        Example:
            >>> ds['url'].str.domain()
            >>> # SQL: domain("url")
        """
        return self._create_function('domain', alias=alias)

    def path(self, alias: str = None) -> 'Function':
        """
        Extract path from URL.

        Maps to ClickHouse: path(url)

        Returns:
            Function expression for path(expr)
        """
        return self._create_function('path', alias=alias)

    # ========== Hash Functions ==========

    def md5(self, alias: str = None) -> 'Function':
        """
        Calculate MD5 hash.

        Maps to ClickHouse: MD5(s)

        Returns:
            Function expression for MD5(expr)
        """
        return self._create_function('MD5', alias=alias)

    def sha256(self, alias: str = None) -> 'Function':
        """
        Calculate SHA256 hash.

        Maps to ClickHouse: SHA256(s)

        Returns:
            Function expression for SHA256(expr)
        """
        return self._create_function('SHA256', alias=alias)

    def city_hash64(self, alias: str = None) -> 'Function':
        """
        Calculate CityHash64 (fast non-cryptographic hash).

        Maps to ClickHouse: cityHash64(s)

        Returns:
            Function expression for cityHash64(expr)
        """
        return self._create_function('cityHash64', alias=alias)
