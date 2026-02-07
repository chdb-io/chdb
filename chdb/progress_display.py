import time
import uuid
import sys


def get_notebook_display():
    try:
        from IPython import get_ipython
        from IPython.display import display, update_display, Markdown
    except Exception:
        return None

    ip = get_ipython()
    if ip is None or not hasattr(ip, "kernel"):
        return None

    return display, update_display, Markdown


def is_notebook():
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    return ip is not None and hasattr(ip, "kernel")


def get_marimo_output():
    try:
        import marimo as mo
        from marimo._runtime.context import runtime_context_installed
    except Exception:
        return None
    if not runtime_context_installed():
        return None
    output = getattr(mo, "output", None)
    if output is None:
        return None
    replace = getattr(output, "replace", None)
    if not callable(replace):
        return None
    md = getattr(mo, "md", None)
    if not callable(md):
        return None

    def _render(value):
        text = value
        return replace(md(f"```\n{text}\n```"))

    return _render


_BAR_CHARS = "█"


def _format_quantity(value):
    value = float(value)
    for suffix, limit in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if value >= limit:
            return f"{value / limit:.2f} {suffix}"
    return f"{value:.0f}"


def _format_bytes(value):
    value = float(value)
    for suffix, limit in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if value >= limit:
            return f"{value / limit:.2f} {suffix}"
    return f"{value:.0f} B"


def _render_progress_bar(current, total, elapsed_ns, bar_width):
    if total > 0:
        ratio = float(current) / float(max(total, 1))
        ratio = min(1.0, max(0.0, ratio))
        percent = int(ratio * 100.0)
        filled = int(ratio * bar_width)
        filled = min(bar_width, max(0, filled))
        bar = _BAR_CHARS * filled + " " * (bar_width - filled)
        return f"{bar} {percent:3d}%"

    # Total is unknown: render an indeterminate moving bar segment.
    pos = int((elapsed_ns // 200_000_000) % max(bar_width, 1))
    chars = [" "] * bar_width
    chars[pos] = _BAR_CHARS
    if bar_width > 1:
        chars[(pos + 1) % bar_width] = _BAR_CHARS
    return f"{''.join(chars)}   ?%"


def render_progress_lines(payload, bar_width=60):
    read_rows = int(payload.get("read_rows", 0))
    read_bytes = int(payload.get("read_bytes", 0))
    total_rows = int(payload.get("total_rows_to_read", 0))
    total_bytes = int(payload.get("total_bytes_to_read", 0))
    elapsed_ns = int(payload.get("elapsed_ns", 0))

    rate_rows = 0.0
    rate_bytes = 0.0
    if elapsed_ns > 0:
        rate_rows = read_rows * 1e9 / elapsed_ns
        rate_bytes = read_bytes * 1e9 / elapsed_ns

    line1 = (
        f"Progress: {_format_quantity(read_rows)} rows, "
        f"{_format_bytes(read_bytes)} ({_format_quantity(rate_rows)} rows/s., "
        f"{_format_bytes(rate_bytes)}/s.)"
    )

    current = read_rows if total_rows > 0 else read_bytes
    total = total_rows if total_rows > 0 else total_bytes
    line2 = _render_progress_bar(current, total, elapsed_ns, bar_width)

    if line2:
        return f"{line1}\n{line2}"
    return line1


class NotebookProgressStructuredCallback:
    def __init__(self, display, update_display, markdown, min_interval=0.2):
        self._display = display
        self._update_display = update_display
        self._display_id = f"chdb-progress-{uuid.uuid4().hex}"
        self._min_interval = min_interval
        self._last_update = 0.0
        self._started = False
        self._markdown = markdown

    def __call__(self, payload):
        now = time.monotonic()
        if self._started and (now - self._last_update) < self._min_interval:
            return
        self._last_update = now
        text = render_progress_lines(payload)
        rendered = self._markdown(f"```\n{text}\n```")
        if not self._started:
            self._display(rendered, display_id=self._display_id)
            self._started = True
        else:
            self._update_display(rendered, display_id=self._display_id)

    def close(self):
        return


class MarimoProgressStructuredCallback:
    def __init__(self, replace_line, min_interval=0.2):
        self._replace_line = replace_line
        self._min_interval = min_interval
        self._last_update = 0.0

    def __call__(self, payload):
        now = time.monotonic()
        if (now - self._last_update) < self._min_interval:
            return
        self._last_update = now
        try:
            self._replace_line(render_progress_lines(payload))
        except Exception:
            return

    def close(self):
        return


def create_auto_progress_callback(
    marimo_replace=None,
    notebook_display=None,
):
    """Create a progress callback for progress=auto mode.

    Selection order:
    1. If TTY is available outside notebook, return None and let terminal progress handle it.
    2. Prefer marimo renderer when available.
    3. Fall back to IPython/Jupyter display callback.
    """
    in_notebook = is_notebook()
    is_tty = sys.stdout.isatty() or sys.stderr.isatty()

    if not in_notebook and is_tty:
        return None

    if marimo_replace is None:
        marimo_replace = get_marimo_output()
    if marimo_replace is not None:
        return MarimoProgressStructuredCallback(marimo_replace)

    if notebook_display is None:
        notebook_display = get_notebook_display()
    if notebook_display is not None:
        display, update_display, markdown = notebook_display
        return NotebookProgressStructuredCallback(display, update_display, markdown)

    return None
