"""Terminal presentation layer for AMLN.

Everything here is pure-Python / numpy: no plotting dependency required. It
gives the demo a readable, structured console report plus text-mode graphics:

* boxed banners, stage headers, aligned key/value blocks and tables
* horizontal bar charts, sparklines and a real ASCII line-chart
* an ASCII rendering of the adaptive memory landscape (height profile with
  hill / valley threshold lines) and a "mountain range" view where each attack
  family becomes a mountain whose size follows its memory height
* sample-data inspection: feature names, example rows, per-family statistics

Unicode block glyphs are used when the console can encode them, otherwise the
module degrades to a pure-ASCII glyph set automatically.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any, Iterable

import numpy as np

WIDTH = 78

_UNICODE = {
    "full": "\u2588", "mid": "\u2593", "low": "\u2591", "peak": "\u25b2",
    "h": "\u2500", "v": "\u2502", "tl": "\u250c", "tr": "\u2510",
    "bl": "\u2514", "br": "\u2518", "lt": "\u251c", "rt": "\u2524",
    "cross": "\u253c", "dh": "\u2550", "dv": "\u2551", "dtl": "\u2554",
    "dtr": "\u2557", "dbl": "\u255a", "dbr": "\u255d",
    "arrow": "\u25b6", "dot": "\u00b7", "bullet": "\u2022",
    "up": "\u25b2", "down": "\u25bc", "ground": "\u2594",
    "spark": "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588",
    "marks": "\u25cf\u25cb\u25c6\u25b2\u25a0",
}

_ASCII = {
    "full": "#", "mid": "+", "low": ".", "peak": "^",
    "h": "-", "v": "|", "tl": "+", "tr": "+",
    "bl": "+", "br": "+", "lt": "+", "rt": "+",
    "cross": "+", "dh": "=", "dv": "|", "dtl": "+",
    "dtr": "+", "dbl": "+", "dbr": "+",
    "arrow": ">", "dot": ".", "bullet": "*",
    "up": "^", "down": "v", "ground": "_",
    "spark": "_.-=+*#@",
    "marks": "*o+x@",
}


def _probe_unicode() -> bool:
    """Try to put the console into UTF-8 mode; report whether glyphs are safe."""
    if sys.platform == "win32":
        try:  # make the Windows console interpret UTF-8 byte sequences
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    probe = "".join(v for v in _UNICODE.values())
    try:
        probe.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


UNICODE_OK = _probe_unicode()
G = _UNICODE if UNICODE_OK else _ASCII


# --------------------------------------------------------------------------
# structural primitives
# --------------------------------------------------------------------------
def banner(title: str, subtitle: str | None = None, width: int = WIDTH) -> None:
    """Double-ruled box used once at the top of a run."""
    inner = width - 2
    print()
    print(G["dtl"] + G["dh"] * inner + G["dtr"])
    print(G["dv"] + title.center(inner) + G["dv"])
    if subtitle:
        print(G["dv"] + subtitle.center(inner) + G["dv"])
    print(G["dbl"] + G["dh"] * inner + G["dbr"])


def stage(number: Any, title: str, width: int = WIDTH) -> None:
    """Stage header, e.g. ``STAGE 3  Continual landscape population``."""
    head = f" STAGE {number}  {title} "
    print()
    print(G["h"] * width)
    print(head)
    print(G["h"] * width)


def rule(label: str = "", width: int = WIDTH, indent: int = 2) -> None:
    pad = " " * indent
    if not label:
        print(pad + G["h"] * (width - indent))
        return
    text = f" {label} "
    print(pad + G["h"] * 2 + text + G["h"] * max(0, width - indent - 2 - len(text)))


def panel(title: str, lines: Iterable[str], width: int = WIDTH,
          indent: int = 2) -> None:
    """Single-ruled titled box around pre-formatted text lines."""
    pad = " " * indent
    inner = width - indent - 2
    head = f"{G['h']} {title} "
    print(pad + G["tl"] + head + G["h"] * max(0, inner - len(head)) + G["tr"])
    for line in lines:
        print(pad + G["v"] + " " + line[:inner - 1].ljust(inner - 1) + G["v"])
    print(pad + G["bl"] + G["h"] * inner + G["br"])


def kv(pairs: Sequence[tuple[str, Any]], indent: int = 2, key_width: int = 30,
       leader: bool = True) -> None:
    """Aligned ``key .......... value`` block."""
    pad = " " * indent
    fill = G["dot"] if leader else " "
    for key, value in pairs:
        k = str(key)
        dots = fill * max(1, key_width - len(k))
        print(f"{pad}{k} {dots} {_fmt(value)}")


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]],
          aligns: Sequence[str] | None = None, indent: int = 2,
          title: str | None = None, max_rows: int | None = None) -> None:
    """Aligned text table with a ruled header."""
    body = [[_fmt(c) for c in row] for row in rows]
    truncated = 0
    if max_rows is not None and len(body) > max_rows:
        truncated = len(body) - max_rows
        body = body[:max_rows]
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in body:
        for i in range(cols):
            if i < len(row):
                widths[i] = max(widths[i], len(row[i]))
    aligns = list(aligns) if aligns else ["l"] + ["r"] * (cols - 1)
    aligns += ["l"] * (cols - len(aligns))
    pad = " " * indent

    def line(cells: Sequence[str]) -> str:
        out = []
        for i in range(cols):
            cell = cells[i] if i < len(cells) else ""
            out.append(cell.rjust(widths[i]) if aligns[i] == "r"
                       else cell.ljust(widths[i]))
        return pad + "  ".join(out).rstrip()

    if title:
        print(pad + title)
    print(line(list(headers)))
    print(pad + "  ".join(G["h"] * w for w in widths))
    for row in body:
        print(line(row))
    if truncated:
        print(f"{pad}{G['dot'] * 3} {truncated} more row(s) not shown")


def note(text: str, indent: int = 2) -> None:
    print(f"{' ' * indent}{G['bullet']} {text}")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        if value == 0:
            return "0.000"
        av = abs(value)
        if av >= 1000:
            return f"{value:,.1f}"
        if av >= 1:
            return f"{value:.3f}"
        if av >= 1e-4:
            return f"{value:.4f}"
        return f"{value:.2e}"
    if isinstance(value, (np.floating,)):
        return _fmt(float(value))
    if isinstance(value, (np.integer,)):
        return str(int(value))
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10000 else str(value)
    return str(value)


# --------------------------------------------------------------------------
# text-mode charts
# --------------------------------------------------------------------------
def bar(value: float, vmax: float, width: int = 24, glyph: str | None = None
        ) -> str:
    """A single horizontal bar string scaled to ``vmax``."""
    glyph = glyph or G["full"]
    if vmax <= 0:
        return " " * width
    filled = int(round(max(0.0, min(1.0, value / vmax)) * width))
    return glyph * filled + G["low"] * 0 + " " * (width - filled)


def bar_chart(items: Sequence[tuple[str, float]], width: int = 30,
              indent: int = 2, value_fmt: str = "{:.3f}",
              glyph: str | None = None, label_width: int | None = None,
              suffix: Sequence[str] | None = None) -> None:
    """Horizontal bar chart: ``label  |#####   value``."""
    if not items:
        note("(no data)", indent)
        return
    vmax = max(v for _, v in items) or 1.0
    lw = label_width or max(len(k) for k, _ in items)
    pad = " " * indent
    for i, (label, value) in enumerate(items):
        extra = f"  {suffix[i]}" if suffix and i < len(suffix) else ""
        print(f"{pad}{label:<{lw}} {G['v']}{bar(value, vmax, width, glyph)}"
              f"{G['v']} {value_fmt.format(value)}{extra}")


def sparkline(values: Sequence[float]) -> str:
    """Compact one-line trend rendering."""
    vals = [float(v) for v in values]
    if not vals:
        return ""
    levels = G["spark"]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span <= 0:
        return levels[len(levels) // 2] * len(vals)
    return "".join(
        levels[min(len(levels) - 1, int((v - lo) / span * (len(levels) - 1) + 0.5))]
        for v in vals)


def line_chart(series: dict[str, Sequence[float]], x_values: Sequence[Any] | None = None,
               height: int = 11, width: int = 56, title: str | None = None,
               y_label: str = "", x_label: str = "", indent: int = 2) -> None:
    """ASCII line chart. Multiple series are drawn with distinct markers."""
    series = {k: [float(v) for v in vs] for k, vs in series.items() if len(vs)}
    if not series:
        note("(nothing to plot)", indent)
        return

    all_vals = [v for vs in series.values() for v in vs]
    lo, hi = min(all_vals), max(all_vals)
    if hi - lo < 1e-12:
        hi, lo = hi + 0.5, lo - 0.5
    marks = G["marks"]
    grid = [[" "] * width for _ in range(height)]

    for s_i, (name, vals) in enumerate(series.items()):
        mark = marks[s_i % len(marks)]
        n = len(vals)
        for col in range(width):
            if n == 1:
                v = vals[0]
            else:  # linear interpolation so the curve reads as a line
                pos = col * (n - 1) / (width - 1) if width > 1 else 0.0
                i0 = int(pos)
                i1 = min(n - 1, i0 + 1)
                frac = pos - i0
                v = vals[i0] * (1 - frac) + vals[i1] * frac
            row = height - 1 - int(round((v - lo) / (hi - lo) * (height - 1)))
            row = max(0, min(height - 1, row))
            if grid[row][col] == " ":
                grid[row][col] = mark

    pad = " " * indent
    if title:
        print(f"{pad}{title}")
    lab_w = 9
    for r in range(height):
        val = hi - (hi - lo) * r / (height - 1)
        show = r == 0 or r == height - 1 or r == height // 2
        axis_label = f"{val:>{lab_w}.3g}" if show else " " * lab_w
        print(f"{pad}{axis_label} {G['rt']}{''.join(grid[r])}")
    print(f"{pad}{' ' * lab_w} {G['bl']}{G['h'] * width}")

    if x_values is not None and len(x_values):
        left = _fmt(x_values[0])
        right = _fmt(x_values[-1])
        axis = left.ljust(max(1, width - len(right))) + right
        print(f"{pad}{' ' * (lab_w + 1)}{axis[:width]}")
    footer = []
    if y_label:
        footer.append(f"y: {y_label}")
    if x_label:
        footer.append(f"x: {x_label}")
    if len(series) > 1:
        legend = "   ".join(f"{marks[i % len(marks)]} {name}"
                            for i, name in enumerate(series))
        footer.append(legend)
    if footer:
        print(f"{pad}{' ' * (lab_w + 1)}{'    '.join(footer)}")


def histogram(values: Sequence[float], bins: int = 10, width: int = 30,
              indent: int = 2, label: str = "") -> None:
    """Bucketed distribution as a horizontal bar chart."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        note("(no values)", indent)
        return
    counts, edges = np.histogram(arr, bins=bins)
    rows = [(f"[{edges[i]:5.2f},{edges[i + 1]:5.2f})", float(counts[i]))
            for i in range(len(counts))]
    if label:
        print(f"{' ' * indent}{label}")
    bar_chart(rows, width=width, indent=indent, value_fmt="{:.0f}")
