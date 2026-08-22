# core/plots.py
# The two showcase figures: a cumulative spaghetti plot and a reference-period
# anomaly plot. Both work on rainfall or on flow, since the structure of the
# question is identical.
#
# Colour convention, applied consistently across both figures:
#   grey    every other water year
#   blue    the wettest year on record
#   red     the driest year on record
#   black   the most recent complete water year
#
# These are deliberately publication quality rather than screen quality. A
# figure that has to be redrawn before it can go in a paper is a figure the tool
# has not really produced.

# %%
import calendar

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% colours
C_OTHER = '#C8C8C8'
C_WET = '#1F77D0'
C_DRY = '#D0245C'
C_RECENT = '#000000'
C_REFERENCE = '#E4E4E4'

CM = 1.0 / 2.54


# %%
def _month_ticks(start_month, n_days=366):
    """Tick positions and labels at the first of each month of the water year.

    Month advance is a simple modular increment. An earlier version had a
    conditional here that sent January back to December, so the labels
    oscillated Dec, Jan, Dec, Jan from the fourth tick onward.
    """
    positions, labels = [], []
    day = 1
    month = int(start_month)

    for _ in range(12):
        if day > n_days:
            break
        positions.append(day)
        labels.append(calendar.month_abbr[month])
        # 2001 is not a leap year, so a water year of 365 days lines up
        day += calendar.monthrange(2001, month)[1]
        month = (month % 12) + 1

    return positions, labels


def _figure(width_cm, height_cm, rect):
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    ax = fig.add_axes(rect)
    # ticks pointing both ways, and no top or right spine, so annotations
    # placed outside the axes are not crossed by a frame line
    ax.tick_params(axis='both', direction='inout', width=0.5, length=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    return fig, ax


# %%
def cumulative_spaghetti(wide, start_month=1, ylabel='Cumulative rainfall (mm)',
                         title=None, subtitle=None, credit=None,
                         width_cm=19.0, height_cm=11.0, annotate=True):
    """Cumulative curves for every water year, with the extremes picked out.

    Parameters
    ----------
    wide : DataFrame from rainfall.cumulative_by_water_year, one column per
        water year, indexed by day of the water year.
    """
    if wide is None or wide.empty:
        return None

    totals = wide.ffill().iloc[-1]
    wettest = int(totals.idxmax())
    driest = int(totals.idxmin())
    recent = int(max(wide.columns))

    fig, ax = _figure(width_cm, height_cm, [0.09, 0.13, 0.88, 0.78])

    for year in wide.columns:
        if year in (wettest, driest, recent):
            continue
        ax.plot(wide.index, wide[year], color=C_OTHER, linewidth=0.7, zorder=1)

    highlights = [(driest, C_DRY, 1.8), (wettest, C_WET, 1.8), (recent, C_RECENT, 2.4)]
    for year, colour, width in highlights:
        ax.plot(wide.index, wide[year], color=colour, linewidth=width, zorder=3,
                label=f'{year}-{str(year + 1)[-2:]}')

    if annotate:
        for year, colour, _ in highlights:
            series = wide[year].dropna()
            if series.empty:
                continue
            ax.annotate(f'{series.iloc[-1]:.0f} mm\n({year}-{str(year + 1)[-2:]})',
                        xy=(series.index[-1], series.iloc[-1]),
                        xytext=(6, 0), textcoords='offset points',
                        color=colour, fontsize=8, va='center', ha='left')

    positions, labels = _month_ticks(start_month, int(wide.index.max()))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, wide.index.max() * 1.02)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Month')
    ax.set_ylabel(ylabel)

    grey_handle = plt.Line2D([], [], color=C_OTHER, linewidth=0.7, label='Other years')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[grey_handle] + handles, loc='lower right', frameon=True,
              framealpha=0.9, fontsize=8)

    if title:
        ax.set_title(title, loc='left', fontsize=12, pad=14 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color='#555555', va='bottom')
    if credit:
        ax.text(1.0, -0.13, credit, transform=ax.transAxes, fontsize=7,
                color='#BBBBBB', ha='right', va='top')

    return fig


# %%
def anomaly_bars(anomaly, year_column, anomaly_column,
                 reference_start=None, reference_end=None,
                 moving_column=None, ylabel='Anomaly (mm)',
                 title=None, subtitle=None, credit=None,
                 width_cm=19.0, height_cm=10.0):
    """Departure from a reference period mean, one bar per water year.

    Blue above the line, red below, black for the most recent complete year, and
    the reference period shaded so the baseline is visible rather than implied.
    """
    if anomaly is None or anomaly.empty:
        return None

    years = anomaly[year_column].to_numpy()
    values = anomaly[anomaly_column].to_numpy()
    recent = int(years.max())

    colours = [C_RECENT if y == recent else (C_WET if v >= 0 else C_DRY)
               for y, v in zip(years, values)]

    fig, ax = _figure(width_cm, height_cm, [0.09, 0.14, 0.88, 0.76])

    if reference_start is not None and reference_end is not None:
        ax.axvspan(reference_start - 0.5, reference_end + 0.5, color=C_REFERENCE,
                   zorder=0, label='Reference period')

    ax.bar(years, values, color=colours, edgecolor='black', linewidth=0.4, zorder=2)
    ax.axhline(0.0, color='black', linewidth=1.0, zorder=3)

    if moving_column and moving_column in anomaly.columns:
        ax.plot(years, anomaly[moving_column].to_numpy(), color='black',
                linewidth=1.8, zorder=4,
                label=moving_column.replace('Anomaly_MA', '').strip() + ' year moving mean')

    ax.set_xlabel('Water year')
    ax.set_ylabel(ylabel)

    handles, labels = ax.get_legend_handles_labels()
    recent_handle = plt.Rectangle((0, 0), 1, 1, facecolor=C_RECENT, edgecolor='black',
                                  linewidth=0.4,
                                  label=f'{recent}-{str(recent + 1)[-2:]}')
    ax.legend(handles=handles + [recent_handle], loc='upper left', frameon=False,
              fontsize=8)

    if title:
        ax.set_title(title, loc='left', fontsize=12, pad=14 if subtitle else 6)
    if subtitle:
        ax.text(0.0, 1.015, subtitle, transform=ax.transAxes, fontsize=8,
                color='#555555', va='bottom')
    if credit:
        ax.text(1.0, -0.14, credit, transform=ax.transAxes, fontsize=7,
                color='#BBBBBB', ha='right', va='top')

    return fig


# %%
def rainfall_runoff_cumulative(rain_wide, flow_wide, year, start_month=1,
                               credit=None, width_cm=19.0, height_cm=10.0):
    """Cumulative rainfall and cumulative runoff for one water year, together.

    The vertical gap between the two curves is water that has fallen on the
    catchment and has not yet left it, so the shape of that gap through the year
    is the catchment's storage behaviour drawn directly. Widening through the
    wet season is filling; a slow closing through the dry season is release.
    """
    if rain_wide is None or flow_wide is None:
        return None
    if year not in rain_wide.columns or year not in flow_wide.columns:
        return None

    rain = rain_wide[year].dropna()
    flow = flow_wide[year].dropna()

    fig, ax = _figure(width_cm, height_cm, [0.10, 0.14, 0.86, 0.76])

    ax.plot(rain.index, rain.to_numpy(), color=C_WET, linewidth=2.0,
            label='Cumulative rainfall')
    ax.plot(flow.index, flow.to_numpy(), color=C_RECENT, linewidth=2.0,
            label='Cumulative runoff')

    common = rain.index.intersection(flow.index)
    ax.fill_between(common, flow.loc[common], rain.loc[common],
                    color=C_WET, alpha=0.12, label='Rainfall not yet discharged')

    positions, labels = _month_ticks(start_month, int(rain.index.max()))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlim(1, rain.index.max())
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Month')
    ax.set_ylabel('Cumulative depth (mm)')
    ax.set_title(f'Water year {year}-{str(year + 1)[-2:]}', loc='left', fontsize=12)
    ax.legend(loc='upper left', frameon=False, fontsize=8)

    if credit:
        ax.text(1.0, -0.14, credit, transform=ax.transAxes, fontsize=7,
                color='#BBBBBB', ha='right', va='top')

    return fig
