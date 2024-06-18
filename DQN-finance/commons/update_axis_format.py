from matplotlib import dates
from matplotlib.figure import Figure


def update_axis_format(fig: Figure, event):
    ax = event.inaxes

    try:
        xlim = ax.get_xlim()
    except AttributeError:
        return
    start, end = dates.num2date(xlim[0]), dates.num2date(xlim[1])
    range_days = (end - start).days
    if range_days < 30:
        # less 30 day -> show days
        ax.xaxis.set_minor_locator(dates.DayLocator())
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%d"))
    elif range_days < 600:
        # less 600 day -> show months, days
        ax.xaxis.set_minor_locator(dates.AutoDateLocator())
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%b %d"))
    else:
        # more 600 day -> show years
        ax.xaxis.set_minor_locator(dates.YearLocator())
        ax.xaxis.set_minor_formatter(dates.DateFormatter("%Y"))

    fig.canvas.draw_idle()
