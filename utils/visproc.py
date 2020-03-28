#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import scipy.signal
import matplotlib.pyplot as plt


def plot_data(t, y, smooth=True, label="Covid data plot",
              color="r", ls="-", save=False, fname='confirmed_coronavirus.png'):
    """
    Plot data.

    Parameters
    ----------
    t : array
        COVID-19 time data.
    y: array
        Data to plot.
    label : str
        Plot label.
    color : str
        Plot color.
    ls : str
        Line style.
    save : bool, optional
        Boolean describing whether to save plot or not. The default is False.
    fname : str, optional
        Name of plot. The default is "death_rate.png".
    """
    # plot data
    if smooth: y = scipy.signal.savgol_filter(y, 7, 3)
    plt.plot(t, y, linestyle=ls, color=color, label=label)
    _ = plt.xticks(rotation=45)
    plt.legend()

    # save file
    if save:
        plt.savefig(fname)


def plot_points_cloud(df, title, x_label, y_label, id_label,
                      marker='x', color='red'):
    """
    Plot points labels graph.

    Parameters
    ----------
    df : pandas.Dataframe
        data to plot.
    title : str, optional
        Plot title. The default is "Covid-19 death rate".
    x_label : str
        X-data column name.
    y_label : str
        Y-data column name.
    id_label : str
        country id.
    marker : TYPE, optional
        DESCRIPTION. The default is 'x'.
    color : TYPE, optional
        DESCRIPTION. The default is 'red'.
    """
    ax = df.plot.scatter(x=x_label, y=y_label, c=color)
    df[[x_label, y_label, id_label]].apply(lambda x: ax.text(*x), axis=1)
    plt.title(title)

    # scale axis
    plt.xscale('log', basex=10)
    plt.yscale('log', basey=10)

    plt.show()
