# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 00:33:13 2015

@author: meggierhayes
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

loansData = pd.read_csv('/Users/meggierhayes/Thinkful_Python_DataScience/Unit2/2.5_MultipleRegression/LoanStats3a.csv')

loansData['int_rate'] = loansData['int_rate'].map(lambda x: round(float(str(x).rstrip('%'))/100, 4))

loansData['annual_inc']

regression_loans=loansData[['int_rate', 'annual_inc']]
regression_loans = regression_loans.dropna()

regression_loans

intrate = regression_loans['int_rate']
income = regression_loans['annual_inc']

y = np.matrix(intrate).transpose()
x1 = np.matrix(income).transpose()

x = np.column_stack([x1])
print x[0:5]

X = sm.add_constant(x)
print X
model = sm.OLS(y,X)
f = model.fit()

print 'Coefficients: ', f.params[0:1]
print 'Intercept: ', f.params[1]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

#######################

multi_loans=loansData[['int_rate', 'annual_inc', 'home_ownership']]
multi_loans
multi_loans = multi_loans.dropna()
multi_loans

multi_loans['home_ownership_ord'] = pd.Categorical(multi_loans.home_ownership).labels

from IPython.core.display import HTML
def short_summary(est):
    return HTML(est.summary().tables[1].as_html())

est = smf.ols(formula='int_rate ~ C(home_ownership)+annual_inc', data=multi_loans).fit()
short_summary(est)

#######################


interact = smf.ols(formula='int_rate ~ annual_inc * C(home_ownership)', data=multi_loans).fit()
short_summary(interact)
