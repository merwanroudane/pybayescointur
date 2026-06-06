"""
Pretty console tables for :mod:`pybayescointur` result objects.

Uses ``rich`` when available for colourful, aligned terminal output and falls
back to a plain pandas string otherwise.  Every helper also returns the
underlying :class:`pandas.DataFrame` so results can be exported to LaTeX / CSV.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

import pandas as pd

try:  # optional dependency
    from rich.console import Console
    from rich.table import Table
    from rich import box
    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


def _render(df: pd.DataFrame, title: str, highlight_col: str | None = None,
            highlight_rule=None):
    """Print ``df`` as a rich table (or plain text) and return it unchanged."""
    if not _HAS_RICH:
        print(f"\n{title}\n" + "-" * len(title))
        print(df.to_string(index=False))
        return df

    console = Console()
    table = Table(title=title, box=box.ROUNDED, header_style="bold cyan",
                  title_style="bold magenta")
    for col in df.columns:
        table.add_column(str(col), justify="right")
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            text = f"{val:.4g}" if isinstance(val, float) else str(val)
            if highlight_col and col == highlight_col and highlight_rule:
                style = highlight_rule(val)
                text = f"[{style}]{text}[/{style}]" if style else text
            cells.append(text)
        table.add_row(*cells)
    console.print(table)
    return df


# --------------------------------------------------------------------------- #
# per-result formatters                                                        #
# --------------------------------------------------------------------------- #
def por_table(result, title="Posterior Odds Ratio - panel unit-root test"):
    """Table for :class:`~pybayescointur.por_panel.PanelPORResult`."""
    df = result.to_frame()
    return _render(
        df, title,
        highlight_col="POR (beta_01)",
        highlight_rule=lambda v: "green" if (isinstance(v, float) and v < 1) else "red",
    )


def break_table(result, title="Structural-break posterior odds ratios"):
    """Table for :class:`~pybayescointur.structural_break.StructuralBreakResult`."""
    df = result.to_frame()
    return _render(
        df, title,
        highlight_col="decision",
        highlight_rule=lambda v: "green" if v == "reject null" else "yellow",
    )


def csd_table(result, title="Posterior model probabilities (cross-sectional dep.)"):
    """Table for :class:`~pybayescointur.cross_section.CSDResult`."""
    df = result.to_frame()
    return _render(
        df, title,
        highlight_col="post_prob",
        highlight_rule=lambda v: "bold green" if (isinstance(v, float) and v > 0.5) else None,
    )


def coint_rank_table(result, title="Cointegrating-rank posterior probabilities"):
    """Table for :class:`~pybayescointur.cointegration.PanelCointResult`."""
    df = result.rank_frame()
    return _render(df, title)
