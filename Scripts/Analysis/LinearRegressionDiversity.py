#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime as dt
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from scipy.stats import normaltest
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 250)

aggr = pd.read_csv('/Data/ByTeamsAndMembers.csv', sep=',',skipinitialspace=True)
aggr


# In[38]:


#VISUAL OUTLIARS IDENTIFICATION
x= aggr['PRSMean']
# print(x)
y=aggr['PRLTMean']
# print(y)
plt.scatter(x, y)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('PRS Mean vs PRLT Mean')
plt.xlabel('PRS MEAN')
plt.ylabel('PRLT MEAN')
plt.show()

#OUTLIARS REMOVAL

aggr_no_outliars = aggr[aggr.PRSMean < 800]
aggr_no_outliars = aggr_no_outliars[aggr.PRLTMean < 15]

x= aggr_no_outliars['PRSMean']
# print(x)
y=aggr_no_outliars['PRLTMean']
# print(y)
plt.scatter(x, y)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
plt.title('PRS Mean vs PRLT Mean')
plt.xlabel('PRS MEAN')
plt.ylabel('PRLT MEAN')
plt.show()


ds = aggr_no_outliars


# In[39]:


#aggr


# In[4]:


#LINEAR REGRESSION  BLAU VS SPR - ALL
x= ds['BlauIndex']
# print(x)
y=ds['PRSMean']
# print(y)


# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
# plt.show()



# In[5]:


#LINEAR REGRESSION  TEAMMEMBERS VS SPR
x= ds['#TEAMMEMBERS']
# print(x)
y=ds['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[40]:


#LINEAR REGRESSION  TEAMMEMBERS VS SPR
x= ds[['BlauIndex', '#TEAMMEMBERS']]
# print(x)
y=ds['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[6]:


#LINEAR REGRESSION  BLAU VS PRLT - ALL
x= ds['BlauIndex']
# print(x)
y=ds['PRLTMean']
# print(y)

x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[7]:


#MULTIPLE LINEAR REGRESSION  BLAU VS SPR
x= ds[['BlauIndex', '#TEAMMEMBERS']]
# print(x)
y=ds['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()



# In[8]:


#LINEAR REGRESSION  TEAM VS PRLT - ALL
x= ds['#TEAMMEMBERS']
# print(x)
y=ds['PRLTMean']
# print(y)

x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[9]:


# ==============================
# SMALL TEAMS
# ==============================


# In[10]:


#LINEAR REGRESSION  BLAU VS SPR
#teamid small teams T04 T01 T03 T05 T07
print('==================================\n BLAU VS SPR - SMALL\n==================================')
ds_small = ds.loc[aggr['TEAMID'].isin(['T04', 'T01', 'T03', 'T05', 'T07'])]
# print(aggrtemp)
x= ds_small['BlauIndex']
# print(x)
y=ds_small['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[11]:


print('==================================\n BLAU VS PRLT - SMALL\n==================================')
ds_small = ds.loc[aggr['TEAMID'].isin(['T04', 'T01', 'T03', 'T05', 'T07'])]
# print(aggrtemp)
x= ds_small['BlauIndex']
# print(x)
y=ds_small['PRLTMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[12]:


# ==============================
# MEDIUM TEAMS
# ==============================


# In[13]:


#LINEAR REGRESSION  BLAU VS SPR
#teamid small teams T04 T01 T03 T05 T07
print('==================================\n BLAU VS SPR - MEDIUM\n==================================')
ds_medium = ds.loc[aggr['TEAMID'].isin(['T11', 'T10', 'T09', 'T06'])]
# print(aggrtemp)
x= ds_medium['BlauIndex']
# print(x)
y=ds_medium['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[14]:


#LINEAR REGRESSION  BLAU VS SPR
#teamid small teams T04 T01 T03 T05 T07
print('==================================\n BLAU VS PRLT - MEDIUM\n==================================')
ds_medium = ds.loc[aggr['TEAMID'].isin(['T11', 'T10', 'T09', 'T06'])]
# print(aggrtemp)
x= ds_medium['BlauIndex']
# print(x)
y=ds_medium['PRLTMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[15]:


# ==============================
# LARGE TEAMS
# ==============================


# In[18]:


#LINEAR REGRESSION  BLAU VS SPR
#teamid large teams T14 T13 T12 T02 T08

print('==================================\n BLAU VS SPR - LARGE\n==================================')
ds_large = ds.loc[aggr['TEAMID'].isin(['T14', 'T13', 'T12', 'T02', 'T08'])]
# print(aggrtemp)
x= ds_large['BlauIndex']
# print(x)
y=ds_large['PRSMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[17]:



print('==================================\n BLAU VS PRLT - LARGE\n==================================')
ds_large = ds.loc[aggr['TEAMID'].isin(['T14', 'T13', 'T12', 'T02', 'T08'])]
# print(aggrtemp)
x= ds_large['BlauIndex']
# print(x)
y=ds_large['PRLTMean']
# print(y)
x=sm.add_constant(x)
# print(x)

model = sm.OLS(y,x)
results = model.fit()
y_pred = results.predict(x)

print(results.summary())

fig = plt.figure(figsize=(3, 9))
fig = sm.graphics.plot_ccpr_grid(results, fig=fig)
# fig.tight_layout(pad=1)
plt.show()


# In[ ]:
