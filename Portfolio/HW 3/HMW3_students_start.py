# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

Data = pd.ExcelFile('C:/Users/rohil/Downloads/DataH3MSF6321.xlsx')

df = Data.parse('Industry Portfolios', skiprows=2, index_col=None, na_values=['NA'])

Ret = df.iloc[:,1:7] # These are the excess returns of the 6 assets
N = len(Ret.columns)

E0 = Ret.mean(axis = 0, skipna = True).T # Here E0 is just the in sample mean of the EXCESS returns
SIGMA = np.cov(Ret.T)

RF = 0.4 

E0 = E0 + RF

'Get the predictors'
DP = df.iloc[:,7]
TERM = df.iloc[:,8]
DEF = df.iloc[:,9]
INF = df.iloc[:,10]
UNE = df.iloc[:,11]

'Just for fun, here are some basic characteristics of the assets'
SDs = (SIGMA.diagonal()**0.5).T
SRs=(E0.T-RF)/SDs
     
Table_Characteristics = np.array([E0.T , SDs , SRs])
print(' ')
print(' Mean, Sdev and Sharpe ratio of the 6 assets')
row = ( 'Mean', 'Sdev', 'Sharpe ratio' )
head = ('Asset 1','Asset 2','Asset 3','Asset 4','Asset 5','Asset 6') 
fmt = '%6.2f'
Table_Characteristics = pd.DataFrame(Table_Characteristics, index = row, columns = head )
print(Table_Characteristics.to_latex(float_format = fmt))

      
      
