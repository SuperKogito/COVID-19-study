#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import pandas as pd


data_urls = {"confirmed_cases": 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
             "recovered_cases": 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
             "death_cases"    : 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'}

def get_country_data(country, data_type):
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
    # read/download csv
    df = pd.read_csv(data_urls[data_type])

    # filter data
    df = df[df['Country/Region'] == country]

    # remove unnecessary columns
    df.drop(['Province/State', 'Lat','Long'], axis=1, inplace=True)

    # unpivot aka "melt"
    df = df.melt(id_vars=['Country/Region'], var_name='date', value_name='confirmed_cases')

    # rename column
    df.rename(columns={'Country/Region': 'country'}, inplace=True)

    # convert date column to date-type
    df.date = pd.to_datetime(df.date)

    # return data
    return df


def get_world_data(data_type):
    """
    Get COVID-19 data for the world [source: CSSE at Johns Hopkins University].

    Parameters
    ----------
    type_of_data : str
        Type of data to collect.

    Returns
    -------
    df : pandas.Dataframe
        Dataframe with COVID-19 information.
    """
    # read/download csv
    df = pd.read_csv(data_urls[data_type])

    # remove unnecessary columns
    df.drop(['Province/State', 'Lat','Long'], axis=1, inplace=True)

    # unpivot aka "melt"
    df = df.melt(id_vars=['Country/Region'], var_name='date', value_name='confirmed_cases')

    # rename column
    df.rename(columns={'Country/Region': 'country'}, inplace=True)

    # convert date column to date-type
    df.date = pd.to_datetime(df.date)

    # return data
    return df


def compute_estimated_infected_population(confirmed_cases_df, death_cases_df, g=8, j=20):
    """
    Compute the estimated infected population.

    Parameters
    ----------
    confirmed_cases_df : pandas.Dataframe
        Dataframe of confirmed COVID-19 cases.
    death_cases_df : pandas.Dataframe
        Dataframe of death cases.
    g : int, optional
        Assumed average number of days taken for a COVID-19 case to lead to death. The default is 15.
    j : TYPE, optional
        Assumed number of days to estmate the rates on. The default is 3.

    Returns
    -------
    I : list
        List with estimated number of cases values.

    """
    # deaths D, confirmed cases C
    D = death_cases_df.confirmed_cases
    C = confirmed_cases_df.confirmed_cases

    # compute the case fatality rate
    CFR = D / C

    # replace NAN = 0/0 by 0: no fatalities, no cases
    np.nan_to_num(CFR, 0)


    # compute the estimated number cases
    I = (D/ CFR)

    # replace NAN = 0/0 by 0: no fatalities, 0 death rate
    np.nan_to_num(I, 0)

    # estimate # cases
    I = I.shift(j).fillna(1) * (1 + g)**j
    return I
