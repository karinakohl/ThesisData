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
get_ipython().run_line_magic('matplotlib', 'inline')

# pd.set_option('display.max_rows', 200)
# pd.set_option('display.max_columns', 50)
prs = pd.read_csv('/PYTHON SCRIPTS/GITHUB/CSVs/02 Data Team/02 getPullRequests.csv', sep=';', skipinitialspace=True)
prsSize = pd.read_csv('/PYTHON SCRIPTS/GITHUB/CSVs/02 Data Team/02 getPullRequestsFiles.csv', sep=';',skipinitialspace=True)
genders = pd.read_csv('//PYTHON SCRIPTS/GITHUB/CSVs/02 Data Team/00_T02_Genders.csv', sep=',',skipinitialspace=True)


# In[2]:


#filtra dados para considerar apenas dados de 2020
start_date = "2020-01-01"
end_date = "2020-12-31"

after_start_date = prs["pullCreatedAt"] >= start_date
before_end_date = prs["pullMergedAt"] <= end_date
between_two_dates = after_start_date & before_end_date
prsFiltered2020 = prs.loc[between_two_dates]
# prsFiltered2020.count()
#Add a new column for calculated data
prsFiltered2020['daysToMerge'] = 0
#calculates the number of days to merge a PR
for i in prsFiltered2020.index:

    try:

        tmpCreatedAt = prsFiltered2020.at[i, 'pullCreatedAt']
        tmpCreatedAt = dt.datetime.strptime(tmpCreatedAt, '%Y-%m-%d')
        tmpMergedAt = prsFiltered2020.at[i, 'pullMergedAt']
        #print("merged at:"+str(tmpMergedAt))

        tmpMergedAt = dt.datetime.strptime(str(tmpMergedAt), '%Y-%m-%d')
        #print(tmp)
        dias = tmpMergedAt - tmpCreatedAt

        prsFiltered2020.at[i,'daysToMerge'] = int(dias.days)

    except:

        continue


# faz o merge com genero
prsFiltered2020Gendered = pd.merge(genders, prsFiltered2020, on="pullUserLogin")

#filtra o arquivo de tamanhos pra pegar só os pullid de 2020
prsSizebyGender = prsSize[prsSize['pullid'].isin(prsFiltered2020Gendered['pullid'])]
prsSizebyGender = pd.merge(genders, prsSizebyGender, on="pullUserLogin")


#junta o arquivo de lead time e o de tamanho num só
prsAll = pd.merge(prsFiltered2020Gendered, prsSizebyGender, on='pullid')


# In[3]:


#PRLT
PRLT=prsFiltered2020Gendered[['gender','pullid', 'userLoginAnonymous', 'daysToMerge']].groupby(['userLoginAnonymous']).agg(minPRLT=('daysToMerge', 'min'), maxPRLT=('daysToMerge', 'max'), meanPRLT=('daysToMerge', 'mean'), stdPRLT=('daysToMerge', 'std'))
#PRLT


# In[4]:


PRS = prsSizebyGender[['gender','pullid', 'userLoginAnonymous', 'changes']].groupby(['userLoginAnonymous']).agg(minChanges=('changes', 'min'), maxChanges=('changes', 'max'), meanChanges=('changes', 'mean'), stdChanges=('changes', 'std'))
#PRS


# In[5]:


#PRLT
PRLTTeam=prsFiltered2020Gendered[['gender','pullid', 'userLoginAnonymous', 'daysToMerge']].agg(minPRLT=('daysToMerge', 'min'), maxPRLT=('daysToMerge', 'max'), meanPRLT=('daysToMerge', 'mean'), stdPRLT=('daysToMerge', 'std'))
print(PRLTTeam.transpose())
PRSTeam = prsSizebyGender[['gender','pullid', 'userLoginAnonymous', 'changes']].agg(minChanges=('changes', 'min'), maxChanges=('changes', 'max'), meanChanges=('changes', 'mean'), stdChanges=('changes', 'std'))
print(PRSTeam.transpose())


# In[ ]:
