"""Render the figure in the README from a daily load CSV.

Usage: python docs/make_figure.py  (writes out/strava-acwr-{light,dark}.png)
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "src")
import acwr

df = pd.read_csv("data/strava_runs_daily.csv")
load = acwr.daily_load(df, date_col="date", load_col="km")
NAMES = ["Coupled (rac)", "Uncoupled (rau)", "Exponential (ewma)"]
SHORT = ["Coupled", "Uncoupled", "Exponential"]
R = [acwr.acwr(load, method=m) for m in ("rac", "rau", "ewma")]
weekly = load.resample("W-SUN").sum()

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "grid": "#e6e5e1",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "bar": "#86b6ef",
        "rule": "#9a998f",
        "warn": "#e34948",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "grid": "#383835",
        "series": ("#3987e5", "#d95926", "#199e70"),
        "bar": "#1c5cab",
        "rule": "#6f6e66",
        "warn": "#e66767",
    },
}

for mode, T in THEMES.items():
    fig = plt.figure(figsize=(12.6, 10.6), facecolor=T["surface"])
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.3, 1.2],
        hspace=0.64,
        left=0.075,
        right=0.815,
        top=0.845,
        bottom=0.055,
    )
    ax = [fig.add_subplot(g) for g in gs]
    for a in ax:
        a.set_facecolor(T["surface"])
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(T["grid"])
            a.spines[s].set_linewidth(1.0)
        a.tick_params(colors=T["ink2"], labelsize=9, length=3, width=0.8)
        a.grid(axis="y", color=T["grid"], linewidth=0.8, alpha=0.9)
        a.set_axisbelow(True)

    fig.text(
        0.075,
        0.955,
        "Three years of running, and what the ratio makes of it",
        color=T["ink"],
        fontsize=17,
        fontweight="semibold",
    )
    fig.text(
        0.075,
        0.925,
        "Aditya Garg  ·  1,018 runs over 1,199 days  ·  23 Jun 2023 to 3 Oct 2026",
        color=T["ink2"],
        fontsize=10.5,
    )
    fig.text(
        0.075,
        0.903,
        "Load is kilometres per day. Rest days count as zero, not as missing data.",
        color=T["ink2"],
        fontsize=10.5,
    )

    # --- A: weekly volume -------------------------------------------------
    a = ax[0]
    a.bar(weekly.index, weekly.values, width=5.0, color=T["bar"], linewidth=0)
    a.set_ylabel("km per week", color=T["ink2"], fontsize=9.5)
    a.set_title(
        "Weekly volume",
        color=T["ink"],
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    a.set_xlim(load.index[0], load.index[-1])
    a.set_ylim(0, float(weekly.max()) * 1.26)
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- B: the three ratios, full span -----------------------------------
    a = ax[1]
    for s, c in zip(R, T["series"], strict=True):
        v = s.replace([np.inf, -np.inf], np.nan)
        a.plot(v.index, v.values, color=c, linewidth=1.2, solid_capstyle="round")
    a.axhline(1.0, color=T["rule"], linewidth=1.0, linestyle=(0, (4, 3)))
    a.axhline(4.0, color=T["rule"], linewidth=1.1, linestyle=(0, (1, 2.5)))
    a.set_ylabel("acute : chronic", color=T["ink2"], fontsize=9.5)
    a.set_title(
        "The same training, three definitions of the ratio",
        color=T["ink"],
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    a.set_xlim(load.index[0], load.index[-1])
    a.set_ylim(0, 9.8)
    a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a.annotate(
        "4.0, the ceiling the\ncoupled form cannot cross",
        xy=(load.index[-1], 4.0),
        xytext=(10, 2),
        textcoords="offset points",
        color=T["ink2"],
        fontsize=9,
        va="center",
    )
    a.annotate(
        "1.0",
        xy=(load.index[-1], 1.0),
        xytext=(10, 0),
        textcoords="offset points",
        color=T["ink2"],
        fontsize=9,
        va="center",
    )
    # 9.07 is the largest *finite* uncoupled value in the series, on
    # 4 January 2024, after the mid-December break. On the seven June 2025 days
    # where the coupled form caps at 4.0 the uncoupled form is infinite, since
    # its 21-day denominator covers nothing but rest, so there is no point on
    # the axis to anchor a label to there.
    a.annotate(
        "uncoupled reaches 9.07 here;\nit has no ceiling to stop at",
        xy=(R[1].replace([np.inf, -np.inf], np.nan).idxmax(), 9.07),
        xytext=(16, -2),
        textcoords="offset points",
        color=T["ink2"],
        fontsize=9,
        va="top",
        ha="left",
    )
    a.legend(
        handles=[
            Line2D([], [], color=c, lw=2.4, label=n)
            for n, c in zip(NAMES, T["series"], strict=True)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.32),
        ncol=3,
        frameon=False,
        fontsize=10,
        labelcolor=T["ink2"],
        handlelength=1.6,
        columnspacing=2.2,
        handletextpad=0.7,
    )

    # --- C: last 90 days ---------------------------------------------------
    a = ax[2]
    win = load.index[-90:]
    ends = sorted(range(3), key=lambda i: float(R[i].reindex(win).iloc[-1]))
    for i in ends:
        v = R[i].reindex(win).replace([np.inf, -np.inf], np.nan)
        c = T["series"][i]
        a.plot(v.index, v.values, color=c, linewidth=2.0, solid_capstyle="round")
        a.plot(
            [v.index[-1]],
            [v.iloc[-1]],
            "o",
            color=c,
            markersize=8,
            markeredgecolor=T["surface"],
            markeredgewidth=2,
            zorder=5,
        )
        a.annotate(
            f"{SHORT[i]}   {v.iloc[-1]:.2f}",
            xy=(v.index[-1], v.iloc[-1]),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            color=c,
            fontsize=10,
            fontweight="semibold",
        )
    a.axhline(1.0, color=T["rule"], linewidth=1.0, linestyle=(0, (4, 3)))
    a.axhline(1.5, color=T["warn"], linewidth=1.2, linestyle=(0, (4, 3)))
    a.annotate(
        "1.5, the threshold coaching material quotes",
        xy=(win[46], 1.5),
        xytext=(0, 7),
        textcoords="offset points",
        color=T["warn"],
        fontsize=9,
    )
    a.set_ylabel("acute : chronic", color=T["ink2"], fontsize=9.5)
    a.set_title(
        "Last 90 days: today the three methods disagree about "
        "whether this is a risky week",
        color=T["ink"],
        fontsize=12,
        fontweight="semibold",
        loc="left",
        pad=10,
    )
    a.set_xlim(win[0], win[-1])
    a.set_ylim(0.4, 2.05)
    a.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=6, interval=2))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    paths = [f"out/strava-acwr-{mode}.png"]
    if mode == "light":
        # The README and the docs site both embed this one, so regenerating
        # has to update it rather than leave a hand-copied file behind.
        paths.append("docs/img/acwr-real-data.png")
    for p in paths:
        pathlib.Path(p).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=170, facecolor=T["surface"])
        print("wrote", p)
    plt.close(fig)
