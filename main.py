#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import warnings
import datetime
import matplotlib
import pandas as pd
import arabic_reshaper
import matplotlib.pyplot as plt
from utils.sirfit import SirFit
from bidi import algorithm as bidialg
from utils.covid_world import CovidWorld
from utils.covid_country import CovidCountry


# hide warnings
warnings.filterwarnings("ignore") 

# set style
matplotlib.rcParams.update({'font.size': 9})
plt.style.use('ggplot')
pd.set_option('expand_frame_repr', False)

# world 
world = CovidWorld()
world.parse_data()

# define list of dates
num_of_days = 7
base = datetime.datetime.today() - datetime.timedelta(days=num_of_days)
dates_list = [base + datetime.timedelta(days=x) for x in range(num_of_days)]


for date in dates_list:
    try:
        world.parse_data()
        world.plot_countries(filter_date=str(date.date()),
                             filter_countries=["Algeria", "Iran", "Sweden", "China", "Italy",
                                               "France", "Spain", "Tunisia", "Turkey",
                                               "Germany", "US", "United Kingdom"], 
                             title= bidialg.get_display(arabic_reshaper.reshape(u" فيروس كورونا حول العالم")))
    except Exception as e: 
        print(e)
        
        
# define list of dates
num_of_days = 1
base = datetime.datetime.today() - datetime.timedelta(days=num_of_days)
dates_list = [base + datetime.timedelta(days=x) for x in range(num_of_days)]


for date in dates_list:
    try:
        world.parse_data()
        world.plot_countries(filter_date=str(date.date()),
                             filter_countries=["Algeria", "Iran", "Sweden", "China", "Italy",
                                               "France", "Spain", "Tunisia", "Turkey",
                                               "Germany", "US", "United Kingdom"], 
                             title= bidialg.get_display(arabic_reshaper.reshape(u" فيروس كورونا حول العالم")))
    except Exception as e: 
        print(e)



# Country
tn = CovidCountry(country="Tunisia")
tn.parse_data()

# death and recovery rates
tn.compute_death_rate(smooth=False, plot=True, title=bidialg.get_display(arabic_reshaper.reshape(u"معدل وفيات فيروس كورونافي تونس")))
tn.compute_recovery_rate(smooth=False, plot=True, title=bidialg.get_display(arabic_reshaper.reshape(u"معدل التعافي من فيروس كورونافي تونس")))

# infections estimations and growth
tn.compute_estimations(smooth=True, plot=True, title=bidialg.get_display(arabic_reshaper.reshape(u"العدد التقديري لعدوى فيروس كورونافي تونس")))
tn.compute_daily_growth(smooth=False, plot=True, title=bidialg.get_display(arabic_reshaper.reshape(u"النمو اليومي لعدوى فيروس كورونافي تونس")))
tn.compute_growth_factor(smooth=False, plot=True, title=bidialg.get_display(arabic_reshaper.reshape(u"عامل النمواليومي لعدوى فيروس كورونافي تونس")))



# modelling and fits
# logistic curve model
tn.logisitc_fit(p0=[0, 1, 1, 1], plot=True, title='Least-squares 4PL fit to covid-19 data for Tunisia')
plt.show()


# SIR model
ij = list(zip(tn.covid_df.confirmed_cases.values, tn.covid_df.recovered_cases.values))
for i, j in ij[-7:]:
    sf = SirFit(total_population=12000000, I0=i, R0=j, 
                contract_rate=.5, recovery_rate=1/14,
                number_of_days=120)
    t, S, I, R = sf.fit()
    sf.plot_fit(t, S, I, R, title="SIR model applied on Covid data in Tunisia")
plt.show()


# Country
de = CovidCountry(country="Germany")
de.parse_data()

# death and recovery rates
de.compute_death_rate(smooth=False, plot=True, title="Covid-19 death rate")
de.compute_recovery_rate(smooth=False, plot=True, title="Covid-19 recovery rate")

# infections estimations and growth
de.compute_estimations(smooth=True, plot=True, title="Estimated number of Covid-19 infections")
de.compute_daily_growth(smooth=False, plot=True, title="Covid-19 daily growth")
de.compute_growth_factor(smooth=False, plot=True, title="Growth factor of Covid-19")

# modelling and fits
# logistic curve model
de.logisitc_fit(p0=[0, 1, 1, 1], plot=True, title='Least-squares 4PL fit to covid-19 data in Germany')
plt.show()


# SIR model
ij = list(zip(de.covid_df.confirmed_cases.values, de.covid_df.recovered_cases.values))
for i, j in ij[-7:]:
    sf = SirFit(total_population=83000000, I0=i, R0=j, 
                contract_rate=.5, recovery_rate=1/14,
                number_of_days=120)
    t, S, I, R = sf.fit()
    sf.plot_fit(t, S, I, R, title="SIR model applied on Covid data in Germany")
plt.show()
