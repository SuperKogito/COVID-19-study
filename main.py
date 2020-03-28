#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import warnings
import datetime
import matplotlib
from sirfit import SirFit
import matplotlib.pyplot as plt
from covid_world import CovidWorld
from covid_country import CovidCountry


"""
covid-19 هذاتحليل لحالة
في تونس استناداً إلى البيانات المتاحة من جامعة جونز هوبكنز

MIT licensed by Ayoub Malek
"""
import warnings
import datetime
import matplotlib
from sirfit import SirFit
import matplotlib.pyplot as plt
from covid_world import CovidWorld
from covid_country import CovidCountry


# hide warnings
warnings.filterwarnings("ignore") 

# set style
matplotlib.rcParams.update({'font.size': 9})
plt.style.use('ggplot')    
    
# Country
tn = CovidCountry(country="Germany")
tn.parse_data()

# death and recovery rates
tn.compute_death_rate(smooth=False, plot=True, title="Covid-19 death rate")
tn.compute_recovery_rate(smooth=False, plot=True, title="Covid-19 recovery rate")

# infections estimations and growth
tn.compute_estimations(smooth=True, plot=True, title="Estimated number of Covid-19 infections")
tn.compute_daily_growth(smooth=False, plot=True, title="Covid-19 daily growth")
tn.compute_growth_factor(smooth=False, plot=True, title="Growth factor of Covid-19")

# modelling and fits
# logistic curve model
tn.logisitc_fit(p0=[0, 1, 1, 1], plot=True, title='Least-squares 4PL fit to covid-19 data')
plt.show()

# SIR model
ij = list(zip(tn.covid_df.confirmed_cases.values, tn.covid_df.recovered_cases.values))
for i, j in ij[-7:]:
    sf = SirFit(total_population=12000000, I0=i, R0=j, 
                contract_rate=.2, recovery_rate=1/21,
                number_of_days=120)
    t, S, I, R = sf.fit()
    sf.plot_fit(t, S, I, R)
plt.show()