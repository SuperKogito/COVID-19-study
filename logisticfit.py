#!/usr/bin/python3
"""
Copyright (c) 2020 Ayoub Malek

This source code is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import datetime
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import leastsq


warnings.filterwarnings("ignore", category=RuntimeWarning)


class LogisticFit:
    def __init__(self, x, y, p0):
        """
        Init LogisticFit class.

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        p0 : TYPE
            DESCRIPTION.
        """
        self.x  = x
        self.y  = y
        self.p0 = p0


    def logistic4(self, x, a, b, c, d):
        """
        4PL lgoistic equation.


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.
        d : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return ((a - d) / (1.0 + ((x / c)**b))) + d


    def residuals(self, p, y, x):
        """
        Deviations of data from fitted 4PL curve.


        Parameters
        ----------
        p : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        err : TYPE
            DESCRIPTION.

        """
        a, b, c, d = p
        err = y - self.logistic4(x, a, b, c, d)
        return err


    def peval(self, x, p):
        """
        Evaluated value at x with current parameters.


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        a, b, c, d = p
        return self.logistic4(x, a, b, c, d)


    def fit_data(self):
        """
        Fit data to curve, determine curve parameters based on given data.
        keyword: curve fitting.

        Returns
        -------
        array
            Logistic fit parameters.

        """
        # Fit equation using least squares optimization
        self.plsq = leastsq(self.residuals, self.p0, args=(self.y, self.x))
        return self.plsq


    def plot_results(self, dates, save=False, fname="logistic_fit.png",
                     title='Least-squares 4PL fit to covid-19 data'):
        """
        Plot data points and fitted curve with the predicted form for the upcoming
        inputs.

        Parameters
        ----------
        dates: array
            x-axis dates.
        save : bool, optional
            Boolean describing whether to save plot or not. The default is False.
        fname : str, optional
            Plot filename. The default is "logistic_fit.png".
        title : str, optional
            Plot title. The default is 'Least-squares 4PL fit to covid-19 data'.
        """
        # Plot results
        plt.plot(self.x, self.y, 'x')
        plt.title(title)

        # add future prediction
        x_pred = [i for i in range(2*len(self.x))]
        y_pred = self.peval(x_pred, self.plsq[0])
        plt.plot(x_pred, y_pred, "-.")
    
        # define list of dates
        num_of_days = 2*len(self.x)
        base = datetime.date(2020, 1, 22)
        dates_list = [base + datetime.timedelta(days=x) for x in range(num_of_days)]

        # format axis
        plt.xticks(x_pred[::14], [str(i)  for i in dates_list[::14]], rotation=45)
        plt.ylabel("predicted number of infections")
        
        # add legend
        plt.legend(['Data', 'Prediction'], loc='upper left')

        # save plot
        if save:
            plt.savefig(fname)
