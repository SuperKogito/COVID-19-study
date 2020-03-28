#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class SirFit:
    def __init__(self, total_population, I0, R0, contract_rate, recovery_rate,
                 number_of_days):
        """
        Init SIR model class.

        Parameters
        ----------
        total_population : int
            Total count of the study population.
        I0 : int
            Initial number of infected.
        R0 : int
            Initial number of recoveries.
        contract_rate : float
            Contract/ disease propagation ratte in persons (= number of people infected by one patient).
        recovery_rate : float
            Rate of recoveries.
        number_of_days : int
            Number od days to foresee in the model.
        """
        # init total population
        self.N = total_population

        # Initial number of infected and recovered individuals, I0 and R0.
        self.I0, self.R0 = I0, R0

        # Everyone else, S0, is susceptible to infection initially.
        self.S0 = self.N - self.I0 - self.R0

        # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
        self.beta,  self.gamma = contract_rate, recovery_rate

        # A grid of time points (in days)
        self.t = np.linspace(0, number_of_days, number_of_days)


    def fit(self):
        """
        Fit data to SIR model

        Returns
        -------
        t : array
            Time array in days.
        S : array
            Susceptible predicted count.
        I : array
            Infected predicted count.
        R : array
            Recovered predicted count.
        """
        # Initial conditions vector
        y0 = self.S0, self.I0, self.R0

        # Integrate the SIR equations over the time grid, t.
        ret = odeint(self.deriv, y0, self.t, args=(self.N, self.beta, self.gamma))
        S, I, R = ret.T
        return self.t, S, I, R


    def deriv(self, y, t, N, beta, gamma):
        """
        Compute SIR derivatives

        Parameters
        ----------
        y : array
            array wit SIR data.
        t : array
            Time array in days.
        N : int
            Total population.
        beta : float
            Contract/ disease propagation ratte in persons (= number of people infected by one patient).
        gamma : float
            Rate of recoveries.

        Returns
        -------
        dSdt : float
            dS / dt.
        dIdt : float
            dI / dt.
        dRdt : float
            dR / dt .
        """
        # The SIR model differential equations.
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt


    def plot_fit(self, t, S, I, R, title="SIR model applied on Covid data"):
        """
        Plot fit/model results.

        Parameters
        ----------
        t : array
            Time array in days.
        S : array
            Susceptible predicted count.
        I : array
            Infected predicted count.
        R : array
            Recovered predicted count.
        title : str, optional
            Plot title. The default is "SIR model applied on Covid data".
        """
        # Plot the data on three separate curves for S(t), I(t) and R(t)
        plt.plot(t, S, 'b', alpha=0.5, lw=1)
        plt.plot(t, I, 'r', alpha=0.5, lw=1)
        plt.plot(t, R, 'g', alpha=0.5, lw=1)
        plt.xlabel('Number of days')
        plt.ylabel('Number of individuals')
        plt.grid(b=True, which='major', c='w', lw=1, ls='-')
        plt.legend(['Susceptible', 'Infected', 'Recovered'], loc='right middle')
        plt.title(title)
