#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import scipy
import scipy.stats
import numpy as np
import scipy.signal
from visproc import plot_data
import matplotlib.pyplot as plt
from logisticfit import LogisticFit
from dataproc import (get_country_data, compute_estimated_infected_population)



class CovidCountry:
    def __init__(self, country="Tunisia", g=14, j=1, ws=7, po=3):
        """
        Init the CovidCountry class.

        Parameters
        ----------
        country : str, optional
            Processed country name. The default is "Tunisia".
        g : TYPE, optional
            Assumed average number of days taken for a COVID-19 case to lead to death. The default is 15.
        j : TYPE, optional
            Assumed number of days to estmate the rates on. The default is 3.
        ws : TYPE, optional
            The smoothing window in days. The default is 7.
        po : TYPE, optional
            the smoothing polynomial order. The default is 3.
        """
        self.country = country
        self.confirmed_cases_df = get_country_data(self.country, "confirmed_cases")
        self.death_cases_df     = get_country_data(self.country, "death_cases")
        self.recovered_cases_df = get_country_data(self.country, "recovered_cases")

        # init infections estimations vars
        self.g, self.j = g, j

        # window size in days and polynomial order
        self.ws, self.po = ws, po


    def parse_data(self):
        """
        Get COVID-19 data for a certain country [source: CSSE at Johns Hopkins University].

        Parameters
        ----------
        country : str
            Country to collect data for.
        type_of_data : str
            Type of data to collect.

        Returns
        -------
        df : pandas.Dataframe
            Dataframe with COVID-19 information.
        """
        # merge data columns
        self.covid_df = self.confirmed_cases_df
        self.covid_df["death_cases"] = self.death_cases_df["confirmed_cases"]
        self.covid_df["recovered_cases"] = self.recovered_cases_df["confirmed_cases"]
        self.covid_df["new_cases"] = abs(self.covid_df["confirmed_cases"] - self.covid_df["confirmed_cases"].shift(1).fillna(0))


    def compute_death_rate(self, smooth=True,
                           plot=True, title="Covid-19 death rate",
                           save=False, fname="death_rate.png"):
        """
        Compute and plot the death rate.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Covid-19 death rate".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "death_rate.png".
        """
        self.covid_df["death_rate"]  = self.covid_df["death_cases"] / self.covid_df["confirmed_cases"]
        self.covid_df["death_rate"]  = self.covid_df["death_rate"].fillna(0)
        self.covid_df["death_rate"][self.covid_df["death_rate"] < 0] = 0

        # plot data
        if plot:
            plot_data(self.covid_df.date, self.covid_df["death_rate"],
                      smooth=smooth, label="Death rate of COVID-19",
                      color="grey", ls="--", save=False)
            plt.title(title)
            plt.show()

        # save plot to file
        if save:
            plt.savefig(fname)


    def compute_recovery_rate(self, smooth=True,
                           plot=True, title="Covid-19 recovery rate",
                           save=False, fname="recovery_rate.png"):
        """
        Compute and plot the recovery rate.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Covid-19 recovery rate".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "recovery_rate.png".
        """
        self.covid_df["recovery_rate"]  = self.covid_df["recovered_cases"] / self.covid_df["confirmed_cases"]
        self.covid_df["recovery_rate"]  = self.covid_df["recovery_rate"].fillna(0)
        self.covid_df["recovery_rate"][self.covid_df["recovery_rate"] < 0] = 0

        # plot data
        if plot:
            plot_data(self.covid_df.date, self.covid_df["recovery_rate"],
                      smooth=smooth, label="Recovery rate of COVID-19",
                      color="grey", ls="--", save=False)
            plt.title(title)
            plt.show()

        # save plot to file
        if save:
            plt.savefig(fname)


    def compute_estimations(self, smooth=True,
                            plot=True, title="Estimated number of Covid-19 infections",
                            save=False, fname="estimated_cases.png"):
        """
        Estimate number of COVID-19 cases from confirmed cases.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Estimated number of Covid-19 infections".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "estimated_cases.png".

        Returns
        -------
        df : pandas.Dataframe
            Dataframe with COVID-19 information.
        """
        # compute estimated infected population
        I = compute_estimated_infected_population(self.confirmed_cases_df, self.death_cases_df, g=self.g, j=self.j)
        self.covid_df["estimated_cases"] = I

        # smoothen results: window size 7 (1 week), polynomial order 3
        self.covid_df["estimated_cases"] = scipy.signal.savgol_filter(I, self.ws, self.po)
        self.covid_df["estimated_cases"][self.covid_df["estimated_cases"] < 0] = 0

        # plot data
        if plot:
            plot_data( self.covid_df.date, self.confirmed_cases_df["confirmed_cases"],
                      label="Confirmed COVID-19 cases", color="orange")
            plot_data(self.covid_df.date, self.covid_df["estimated_cases"],
                      smooth=smooth, label="Estimated COVID-19 cases",
                      color="purple", ls="--")
            plt.title(title)
            plt.show()

        # save plot to file
        if save:
            plt.savefig(fname)


    def compute_daily_growth(self, smooth=True,
                             plot=True, title="Covid-19 daily growth",
                             save=False, fname="daily_growth.png"):
        """
        Compute and plot the daily growth.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Covid-19 daily growth".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "daily_growth.png".
        """
        # compute linear growth rate
        self.covid_df["daily_growth"]  = abs(self.covid_df["confirmed_cases"] - self.covid_df["confirmed_cases"].shift(1).fillna(0))
        self.covid_df["daily_growth"] /= self.covid_df["confirmed_cases"].shift(1).fillna(0)
        self.covid_df["daily_growth"]  = self.covid_df["daily_growth"].fillna(0)
        self.covid_df["daily_growth"]  = self.covid_df["daily_growth"].replace(np.inf, self.covid_df["new_cases"])

        # remove negative values
        self.covid_df["daily_growth"][self.covid_df["daily_growth"] < 0] = 0

        # plot data
        if plot:
            plot_data(self.covid_df.date, self.covid_df["daily_growth"],
                      smooth=smooth, label="Daily growth of COVID-19 cases",
                      color="grey", ls="--", save=False)
            plt.title(title)
            plt.show()

        # save plot to file
        if save:
            plt.savefig(fname)


    def compute_growth_factor(self, smooth=True,
                              plot=True, title="Growth factor of Covid-19",
                              save=False, fname="growth_factor.png"):
        """
        Compute and plot the grwoth factor.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Growth factor of Covid-19".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "growth_factor.png".
        """
        self.covid_df["growth_factor"] = self.covid_df["new_cases"] / self.covid_df["new_cases"].shift(1).fillna(0)
        self.covid_df["growth_factor"] = self.covid_df["growth_factor"].fillna(0)
        self.covid_df["growth_factor"] = self.covid_df["growth_factor"].replace(np.inf, self.covid_df["new_cases"])

        # smoothen results: window size 7 (1 week), polynomial order 3
        if smooth: self.covid_df["growth_factor"] = scipy.signal.savgol_filter(self.covid_df["growth_factor"], self.ws, self.po)

        # remove negative values
        self.covid_df["growth_factor"][self.covid_df["growth_factor"] < 0] = 0

        # plot growth factor
        if plot:
            plot_data(self.covid_df.date, self.covid_df["growth_factor"],
                      smooth=smooth, label="Growth factor of COVID-19 cases",
                      color="grey", ls="--", save=False)
            plt.title(title)
            plt.show()

        # save plot to file
        if save:
            plt.savefig(fname)


    def logisitc_fit(self, p0=[0, 1, 1, 1],
                     plot=True, title='Least-squares 4PL fit to covid-19 data',
                     save=False, fname="logistic_fit.png"):
        """
        Fit to a logistic curve model.

        Parameters
        ----------
        smooth : bool, optional
            Boolean describing whether or not to smooth the curve. The default is True.
        plot : bool, optional
            Boolean describing whether to plot data or not. The default is True.
        title : str, optional
            Plot title. The default is "Least-squares 4PL fit to covid-19 data".
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Name of plot. The default is "logistic_fit.png".
        """
        # init data
        t = np.arange(0, self.covid_df.shape[0])
        v = self.covid_df["confirmed_cases"].values
       
        # fit
        lgf = LogisticFit(t, v, p0)
        lgf.fit_data()

        # plot fit
        if plot:
            lgf.plot_results(self.covid_df["date"].values, save, fname, title)
