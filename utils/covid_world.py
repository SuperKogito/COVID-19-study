#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from .dataproc import get_world_data
from .visproc import plot_points_cloud


class CovidWorld:
    def __init__(self):
        """
        Init Covid world class.
        """
        # world data
        self.world_confirmed_cases_df = get_world_data(data_type="confirmed_cases")
        self.world_death_cases_df     = get_world_data(data_type="death_cases")
        self.world_recovered_cases_df = get_world_data(data_type="recovered_cases")


    def parse_data(self):
        """
        Parse dataframe data.
        """
        # merge data columns
        self.world_covid_df = self.world_confirmed_cases_df
        self.world_covid_df["death_cases"] = self.world_death_cases_df["confirmed_cases"]
        self.world_covid_df["recovered_cases"] = self.world_recovered_cases_df["confirmed_cases"]

        # compute death rates
        self.world_covid_df["death_rate"]  = self.world_covid_df["death_cases"] / self.world_covid_df["confirmed_cases"]
        self.world_covid_df["death_rate"]  = self.world_covid_df["death_rate"].fillna(0)


    def plot_countries(self, filter_date="2020-03-25",
                       filter_countries=["China", "France", "Germany", "Italy", "Spain", "Tunisia", "US"],
                       title="Covid-19 Confirmed cases to death cases on "):
        """
        Plot countries point clouds.

        Parameters
        ----------
        filter_date : str, optional
            Date to filter data for. The default is "2020-03-25".
        filter_countries : list, optional
            Countries to filter data for. The default is ["China", "France", "Germany", "Italy", "Spain", "Tunisia", "US"].
        title : str, optional
            Plot title. The default is "Covid-19 Confirmed cases to death cases on ".
        """
        # filter data on date
        self.world_covid_df = self.world_covid_df[self.world_covid_df.date == filter_date]
        self.world_covid_df = self.world_covid_df[self.world_covid_df.country.isin(filter_countries)]
        self.world_covid_df = self.world_covid_df.groupby("country").sum()
        self.world_covid_df["death_rate"] = self.world_covid_df["death_cases"] / self.world_covid_df["confirmed_cases"]
        self.world_covid_df["country"]    = self.world_covid_df.index

        # plot cloud
        plot_points_cloud(self.world_covid_df, title + filter_date,
                          "death_cases", "confirmed_cases", "country",
                          color='red')
