#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

import imblearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

sns.set_style(style='darkgrid')


# In[2]:


df1 = pd.read_csv('bond_ratings.csv')
df2 = pd.read_csv('fund_allocations.csv')
df3 = pd.read_csv('fund_config.csv')
df4 = pd.read_csv('fund_ratios.csv')
df5 = pd.read_csv('fund_specs.csv')
df6 = pd.read_csv('other_specs.csv')
df7 = pd.read_csv('return_3year.csv')
df8 = pd.read_csv('return_5year.csv')
df9 = pd.read_csv('return_10year.csv')


# In[3]:


def replace_outliers(v_df): 
    Q1 = v_df[:].quantile(0.25)
    Q2 = v_df[:].quantile(0.50)
    Q3 = v_df[:].quantile(0.75)
    IQR = Q3 - Q1
    
    gr_num = v_df.select_dtypes(exclude ='object')
    for feature in gr_num.columns:
        if IQR[feature] != 0:
            gr_num[feature] = np.where((gr_num[feature] < (Q1[feature] - 1.5 * IQR[feature])), Q1[feature], gr_num[feature])
            gr_num[feature] = np.where((gr_num[feature] > (Q3[feature] + 1.5 * IQR[feature])), Q3[feature], gr_num[feature])                           
    gr_df = v_df.select_dtypes(include ='object')
    gr_df = gr_df.join(gr_num)
    return gr_df


# ### Dataframe 1

# In[4]:


df1.info()


# In[5]:


df1[:].fillna((df1[:].median()), inplace = True) #fixing null values
df1.info()
gr_1 = replace_outliers(df1.drop(columns='tag')) #fixing outliers
gr1 = gr_1.join(pd.DataFrame(df1[['tag']]))
gr1.skew()


# In[7]:


df1.skew()


# ### Dataframe 2

# In[8]:


df2.info()


# In[9]:


df2[:].fillna((df2[:].median()), inplace = True) #fixing null values
df2.info()
gr_2 = replace_outliers(df2.drop(columns='id')) #fixing outliers
gr2 = gr_2.join(pd.DataFrame(df2[['id']]))
gr2.skew()


# ### Dataframe 3

# In[10]:


df3.info()


# ### Dataframe 4

# In[11]:


df4.info()


# In[12]:


# Converting the datatype from object to float by removing the unwanted commas
df4['ps_ratio'] = df4['ps_ratio'].str.replace(",","").astype(float)
df4['mmc'] = df4['mmc'].str.replace(",","").astype(float)
df4['pc_ratio'] = df4['pc_ratio'].str.replace(",","").astype(float)
df4['pe_ratio'] = df4['pe_ratio'].str.replace(",","").astype(float)


# In[14]:


df4[:].fillna((df4[:].median()), inplace = True) #fixing the null values
df4.info()
gr_4 = replace_outliers(df4.drop(['fund_id','tag'], axis=1)) #fixing outliers
gr4 = gr_4.join(pd.DataFrame(df4[['fund_id','tag']]))
gr4.skew()


# ### Dataframe 6

# In[15]:


df6.drop(columns='greatstone_rating',inplace = True)


# In[16]:


df6.info()


# In[17]:


# Converting the datatype from object to float by removing the unwanted commas
df6['ps_ratio'] = df6['ps_ratio'].str.replace(",","").astype(float)
df6['mmc'] = df6['mmc'].str.replace(",","").astype(float)
df6['pc_ratio'] = df6['pc_ratio'].str.replace(",","").astype(float)
df6['pe_ratio'] = df6['pe_ratio'].str.replace(",","").astype(float)


# In[19]:


df6[:].fillna((df6[:].median()), inplace = True) #fixing the null values
df6.info()
gr_6 = replace_outliers(df6.drop(['tag'], axis=1)) #fixing outliers
gr6 = gr_6.join(pd.DataFrame(df6[['tag']]))
gr6.skew()


# ### Dataframe 7

# In[20]:


df7.info()


# In[21]:


# Converting the datatype from object to float by removing the unwanted commas
df7['3yrs_treynor_ratio_fund'] = df7['3yrs_treynor_ratio_fund'].str.replace(",","").astype(float)


# In[23]:


df7[:].fillna((df7[:].median()), inplace = True) #fixing the null values
df7.info()
gr_7 = replace_outliers(df7.drop(['tag'], axis=1)) #fixing outliers
gr7 = gr_7.join(pd.DataFrame(df7[['tag']]))
gr7.skew()


# ### Dataframe 8

# In[24]:


df8.info()


# In[25]:


# Converting the datatype from object to float by removing the unwanted commas
df8['5yrs_treynor_ratio_fund'] = df8['5yrs_treynor_ratio_fund'].str.replace(",","").astype(float)


# In[119]:


df8[:].fillna((df8[:].median()), inplace = True) #fixing the null values
df8.info()
gr_8 = replace_outliers(df8.drop(['tag'], axis=1)) #fixing outliers
gr8 = gr_8.join(pd.DataFrame(df8[['tag']]))
gr8.skew()


# ### Dataframe 9

# In[26]:


df9.info()


# In[27]:


# Converting the datatype from object to float by removing the unwanted commas
df9['10yrs_treynor_ratio_fund'] = df9['10yrs_treynor_ratio_fund'].str.replace(",","").astype(float)


# In[32]:


df9[:].fillna((df9[:].median()), inplace = True) #fixing the null values


# In[33]:


df9.info()
gr_9 = replace_outliers(df9.drop(['fund_id'], axis=1)) #fixing outliers
gr9 = gr_9.join(pd.DataFrame(df9[['fund_id']]))
gr9.skew()


# ### Dataframe 5

# In[34]:


df5 = pd.read_csv('fund_specs.csv')
df5.info()


# In[35]:


df5['investment_class'] = df5['investment_class'].fillna(df5['investment_class'].value_counts().index[0])
df5['fund_size'] = df5['fund_size'].fillna(df5['fund_size'].value_counts().index[0])
df5.info()


# In[36]:


df5['total_assets'].fillna((df5['total_assets'].median()), inplace = True)
df5['yield'].fillna((df5['yield'].median()), inplace = True)
df5['return_ytd'].fillna((df5['return_ytd'].median()), inplace = True) #fixing the null values
df5.info()
gr_5 = replace_outliers(df5.drop(['tag','greatstone_rating'], axis=1)) #fixing outliers
gr5 = gr_5.join(pd.DataFrame(df5[['tag','greatstone_rating']]))
gr5.skew()


# In[37]:


gr5.isna().sum()


# In[120]:


df_temp1 = pd.merge(gr1, gr2, left_on="tag", right_on="id")
df_temp1.info()


# In[130]:


df_temp1.drop(columns='id', inplace = True)


# In[131]:


gr4.info()
df_temp2 = pd.merge(df_temp1, gr4)


# In[132]:


df_temp3 = pd.merge(df_temp2, df3)
df_temp3.info()


# In[133]:


df_temp4 = pd.merge(df_temp3, gr5)
df_temp4.info()


# In[134]:


df_temp5 = pd.merge(df_temp4, gr6)
df_temp5.info()


# In[135]:


df_temp6 = pd.merge(df_temp5, gr7)
df_temp6.info()


# In[136]:


df_temp7 = pd.merge(df_temp6, gr8)
df_temp7.info()


# In[137]:


df = pd.merge(df_temp7, gr9)
df.info()


# In[138]:


df.shape


# In[139]:


df.isnull().values.any() 


# In[140]:


df.describe().T


# In[141]:


df_cat = df.select_dtypes(exclude = 'float')
df_cat.head()


# # Separating out the test data where GR is null

# In[280]:


df_test = df[df.greatstone_rating.isnull()][:]


# In[281]:


df_test.describe()


# In[144]:


gr_df = df


# In[145]:


gr_df = gr_df.merge(df_test, how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
 
gr_df.info()


# In[146]:


gr_df['greatstone_rating'] = gr_df['greatstone_rating'].astype(str)


# In[147]:


gr_df['greatstone_rating'].dtypes


# In[148]:


gr_df.greatstone_rating.value_counts()


# - Getting the final dataframe which will be used for model building

# In[149]:


gr_df.info()


# In[150]:


gr_df.head()


# ## Univariate Analysis

# ### Columns of DF1 

# In[311]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['bb_rating','us_govt_bond_rating','below_b_rating','others_rating','maturity_bond','b_rating','a_rating','aaa_rating','aa_rating','bbb_rating','duration_bond']:   # for-loop to iterate over every attribute whose distribution is to be visualized
    plt.subplot(4, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[312]:


sns.pairplot(gr_df, vars = ['bb_rating','us_govt_bond_rating','below_b_rating','others_rating','maturity_bond','b_rating','a_rating','aaa_rating','aa_rating','bbb_rating','duration_bond'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[304]:


corr = abs(gr_df[gr_1.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# - 'us_govt_bond_rating' and 'maturity_bond' are very lightly correlated, thus can be ignored

# ### Columns of DF2

# In[225]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['portfolio_communication_allocation','portfolio_financial_services','portfolio_industrials_allocation','portfolio_tech_allocation','portfolio_materials_basic_allocation','portfolio_energy_allocation','portfolio_consumer_defence_allocation','portfolio_healthcare_allocation','portfolio_property_allocation','portfolio_utils_allocation','portfolio_cyclical_consumer_allocation']:   # for-loop to iterate over every attribute whose distribution is to be visualized
    plt.subplot(4, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[314]:


sns.pairplot(gr_df, vars = ['portfolio_communication_allocation','portfolio_financial_services','portfolio_industrials_allocation','portfolio_tech_allocation','portfolio_materials_basic_allocation','portfolio_energy_allocation','portfolio_consumer_defence_allocation','portfolio_healthcare_allocation','portfolio_property_allocation','portfolio_utils_allocation','portfolio_cyclical_consumer_allocation'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[315]:


corr = abs(gr_df[gr_2.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# ### Columns of DF4

# In[246]:


plt.figure(figsize= (30,20))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['fund_ratio_net_annual_expense','pb_ratio','ps_ratio','mmc','pc_ratio','pe_ratio']:
    plt.subplot(2, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[316]:


sns.pairplot(gr_df, vars = ['fund_ratio_net_annual_expense','pb_ratio','ps_ratio','mmc','pc_ratio','pe_ratio'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[317]:


corr = abs(gr_df[gr_4.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# - good correlation between pb_ratio, ps_ratio, pc_ratio

# ### Columns of DF5 

# In[231]:


plt.figure(figsize= (30,10))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['total_assets','yield','return_ytd']:
    plt.subplot(1, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[318]:


sns.pairplot(gr_df, vars = ['total_assets','yield','return_ytd'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[319]:


corr = abs(gr_df[gr_5.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# Avarage correlation between return_ytd and yeild, rest are very less

# ### Columns of DF6

# In[235]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['2014_category_return','2012_return_category','years_up','2018_return_category','category_return_1year','cash_percent_of_portfolio','pc_ratio','2011_return_category','ytd_return_fund','years_down','2014_return_fund','category_return_1month','2013_return_fund','fund_return_3months','ytd_return_category','pb_ratio','2017_category_return','1_year_return_fund','pe_ratio','2015_return_fund']:
    plt.subplot(7, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[320]:


sns.pairplot(gr_df, vars = ['2014_category_return','2012_return_category','years_up','2018_return_category','category_return_1year','cash_percent_of_portfolio','pc_ratio','2011_return_category','ytd_return_fund','years_down','2014_return_fund','category_return_1month','2013_return_fund','fund_return_3months','ytd_return_category','pb_ratio','2017_category_return','1_year_return_fund','pe_ratio','2015_return_fund'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[329]:


gr_temp = gr_df[['2014_category_return','2012_return_category','years_up','2018_return_category','category_return_1year','cash_percent_of_portfolio','pc_ratio','2011_return_category','ytd_return_fund','years_down','2014_return_fund','category_return_1month','2013_return_fund','fund_return_3months','ytd_return_category','pb_ratio','2017_category_return','1_year_return_fund','pe_ratio','2015_return_fund']].copy()


# In[330]:


corr = abs(gr_temp.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# In[236]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['portfolio_convertable','3_months_return_category','portfolio_others','2016_return_fund','mmc','stock_percent_of_portfolio','2016_return_category','ps_ratio','2011_return_fund','2010_return_fund','fund_return_3years','2012_fund_return','2018_return_fund','2017_return_fund','category_ratio_net_annual_expense','category_return_2015','1_month_fund_return','bond_percentage_of_porfolio','portfolio_preferred','2010_return_category','2013_category_return']:
    plt.subplot(7, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[331]:


sns.pairplot(gr_df, vars = ['portfolio_convertable','3_months_return_category','portfolio_others','2016_return_fund','mmc','stock_percent_of_portfolio','2016_return_category','ps_ratio','2011_return_fund','2010_return_fund','fund_return_3years','2012_fund_return','2018_return_fund','2017_return_fund','category_ratio_net_annual_expense','category_return_2015','1_month_fund_return','bond_percentage_of_porfolio','portfolio_preferred','2010_return_category','2013_category_return'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[332]:


gr_temp = gr_df[['portfolio_convertable','3_months_return_category','portfolio_others','2016_return_fund','mmc','stock_percent_of_portfolio','2016_return_category','ps_ratio','2011_return_fund','2010_return_fund','fund_return_3years','2012_fund_return','2018_return_fund','2017_return_fund','category_ratio_net_annual_expense','category_return_2015','1_month_fund_return','bond_percentage_of_porfolio','portfolio_preferred','2010_return_category','2013_category_return']].copy()


# In[333]:


corr = abs(gr_temp.corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# ### Columns of DF7

# In[244]:


plt.figure(figsize= (30,20))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['3yrs_treynor_ratio_fund','3_years_alpha_fund','3years_category_std','3yrs_sharpe_ratio_fund','3yrs_treynor_ratio_category','3_years_return_mean_annual_fund','fund_beta_3years','3years_fund_r_squared','3years_fund_std','category_beta_3years','fund_return_3years','3_years_alpha_category','3_years_return_mean_annual_category','3yrs_sharpe_ratio_category','3years_category_r_squared','3_years_return_category']:
    plt.subplot(6, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[334]:


sns.pairplot(gr_df, vars = ['3yrs_treynor_ratio_fund','3_years_alpha_fund','3years_category_std','3yrs_sharpe_ratio_fund','3yrs_treynor_ratio_category','3_years_return_mean_annual_fund','fund_beta_3years','3years_fund_r_squared','3years_fund_std','category_beta_3years','fund_return_3years','3_years_alpha_category','3_years_return_mean_annual_category','3yrs_sharpe_ratio_category','3years_category_r_squared','3_years_return_category'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[335]:


corr = abs(gr_df[gr_7.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# ### Columns of DF8

# In[242]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['category_r_squared_5years','5yrs_sharpe_ratio_fund','5_years_alpha_fund','5years_fund_r_squared','5years_fund_std','5yrs_sharpe_ratio_category','5_years_beta_fund','5yrs_treynor_ratio_fund','5_years_return_mean_annual_fund','5_years_return_mean_annual_category','5yrs_treynor_ratio_category','5_years_return_fund','5_years_alpha_category','5_years_beta_category','5years_category_std','5_years_return_category']:
    plt.subplot(6, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[336]:


sns.pairplot(gr_df, vars = ['category_r_squared_5years','5yrs_sharpe_ratio_fund','5_years_alpha_fund','5years_fund_r_squared','5years_fund_std','5yrs_sharpe_ratio_category','5_years_beta_fund','5yrs_treynor_ratio_fund','5_years_return_mean_annual_fund','5_years_return_mean_annual_category','5yrs_treynor_ratio_category','5_years_return_fund','5_years_alpha_category','5_years_beta_category','5years_category_std','5_years_return_category'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[337]:


corr = abs(gr_df[gr_8.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# ### Columns of DF9

# In[245]:


plt.figure(figsize= (30,40))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['10years_category_r_squared','10yrs_sharpe_ratio_fund','10_years_alpha_fund','10years_fund_r_squared','10years_fund_std','10yrs_sharpe_ratio_category','10_years_beta_fund','10yrs_treynor_ratio_fund','10_years_return_mean_annual_category','10yrs_treynor_ratio_category','10_years_return_fund','10_years_alpha_category','10_years_beta_category','10years_category_std','10_years_return_mean_annual_fund','10_years_return_category']:
    plt.subplot(6, 3, pos)   # plot grid
    sns.distplot(gr_df[feature], kde= True )
    pos += 1


# In[338]:


sns.pairplot(gr_df, vars = ['10years_category_r_squared','10yrs_sharpe_ratio_fund','10_years_alpha_fund','10years_fund_r_squared','10years_fund_std','10yrs_sharpe_ratio_category','10_years_beta_fund','10yrs_treynor_ratio_fund','10_years_return_mean_annual_category','10yrs_treynor_ratio_category','10_years_return_fund','10_years_alpha_category','10_years_beta_category','10years_category_std','10_years_return_mean_annual_fund','10_years_return_category'], hue='greatstone_rating', diag_kind='kde')    # pairplot
plt.show()


# In[339]:


corr = abs(gr_df[gr_9.columns].corr()) # correlation matrix
lower_triangle = np.tril(corr, k = -1)  # select only the lower triangle of the correlation matrix
mask = lower_triangle == 0  # to mask the upper triangle in the following heatmap

plt.figure(figsize = (15,10))  # setting the figure size
sns.set_style(style = 'white')  # Setting it to white so that we do not see the grid lines
sns.heatmap(lower_triangle, center=0.5, cmap= 'YlGnBu', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
            cbar= False, linewidths= 1, mask = mask)   # Da Heatmap
#plt.xticks(rotation = 50)   # Aesthetic purposes
#plt.yticks(rotation = 10)   # Aesthetic purposes
plt.show()


# ### Categorical Variables 

# In[151]:


df_cat.columns


# In[152]:


plt.figure(figsize= (20,20))  # Set the figure size
pos = 1    # a variable to manage the position of the subplot in the overall plot
for feature in ['greatstone_rating', 'investment_class', 'fund_size']: #'tag', 'fund_id', 'category', 'parent_company','fund_name','currency', 'inception_date' are not required
    plt.subplot(3, 1, pos)   # plot grid
    sns.countplot(gr_df[feature], palette= 'Blues')
    pos += 1


# In[153]:


gr_df['greatstone_rating'] = gr_df['greatstone_rating'].astype(float)

#pd.to_numeric(gr_df['greatstone_rating'])


# In[154]:


gr_df['greatstone_rating'].describe()


# In[155]:


correlation_values=gr_df.corr()['greatstone_rating']
corr_df = correlation_values.abs().sort_values(ascending=False)


# In[68]:


corr_df.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\Correlation file.csv')


# In[156]:


gr_df.drop(columns =['tag', 'category', 'parent_company','fund_name','currency', 'inception_date', 'us_govt_bond_rating', '_merge'], inplace = True)


# In[157]:


gr_df.shape


# ## Convert Strings to columns 

# In[158]:


gr_df = pd.get_dummies(gr_df, columns=['fund_size','investment_class'])
gr_df.set_index('fund_id', inplace = True)


# In[161]:


gr_df.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\Final_file.csv')


# ## Split the data into train and test

# In[162]:


x = gr_df.drop(columns='greatstone_rating',axis=1)    # Predictors
y = gr_df.loc[:,'greatstone_rating'] # target


# In[163]:


#con_df = X.select_dtypes(exclude ='object')
from scipy.stats import zscore
X1 = x.select_dtypes(include ='object')


# In[164]:


X1.shape


# In[165]:


X = x.apply(zscore)


# In[166]:


X.shape


# In[167]:


X_null= X.isnull().sum()


# In[168]:


X_null.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\check_null.csv')


# In[169]:


X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=1) # Splitting the data in 80:20 ratio
print("Size of the training dataframe is ",len(X_train))
print("Size of the test dataframe is ",len(X_validate))
print("{0:0.2f}% data is in training set".format((len(X_train)/len(gr_df.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(X_validate)/len(gr_df.index)) * 100))


# In[205]:


total_rating_0 = len(gr_df.loc[gr_df['greatstone_rating'] == 0.0])
total_rating_1 = len(gr_df.loc[gr_df['greatstone_rating'] == 1.0])
total_rating_2 = len(gr_df.loc[gr_df['greatstone_rating'] == 2.0])
total_rating_3 = len(gr_df.loc[gr_df['greatstone_rating'] == 3.0])
total_rating_4 = len(gr_df.loc[gr_df['greatstone_rating'] == 4.0])
total_rating_5 = len(gr_df.loc[gr_df['greatstone_rating'] == 5.0])
print("Total rating_0: {0} ({1:2.2f}%)".format(total_rating_0, (total_rating_0 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("Total rating_1: {0} ({1:2.2f}%)".format(total_rating_1, (total_rating_1 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("Total rating_2: {0} ({1:2.2f}%)".format(total_rating_2, (total_rating_2 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("Total rating_3: {0} ({1:2.2f}%)".format(total_rating_3, (total_rating_3 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("Total rating_4: {0} ({1:2.2f}%)".format(total_rating_4, (total_rating_4 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("Total rating_5: {0} ({1:2.2f}%)".format(total_rating_5, (total_rating_5 / (total_rating_0 + total_rating_1 + total_rating_2 + total_rating_3 + total_rating_4 + total_rating_5)) * 100 ))
print("")
print("Training rating_0: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0.0]), (len(y_train[y_train[:] == 0.0])/len(y_train)) * 100))
print("Training rating_1: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1.0]), (len(y_train[y_train[:] == 1.0])/len(y_train)) * 100))
print("Training rating_2: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 2.0]), (len(y_train[y_train[:] == 2.0])/len(y_train)) * 100))
print("Training rating_3: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 3.0]), (len(y_train[y_train[:] == 3.0])/len(y_train)) * 100))
print("Training rating_4: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 4.0]), (len(y_train[y_train[:] == 4.0])/len(y_train)) * 100))
print("Training rating_5: {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 5.0]), (len(y_train[y_train[:] == 5.0])/len(y_train))* 100))
print("")
print("Test rating_0: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 0.0]), (len(y_validate[y_validate[:] == 0.0])/len(y_validate)) * 100))
print("Test rating_1: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 1.0]), (len(y_validate[y_validate[:] == 1.0])/len(y_validate)) * 100))
print("Test rating_2: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 2.0]), (len(y_validate[y_validate[:] == 2.0])/len(y_validate)) * 100))
print("Test rating_3: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 3.0]), (len(y_validate[y_validate[:] == 3.0])/len(y_validate)) * 100))
print("Test rating_4: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 4.0]), (len(y_validate[y_validate[:] == 4.0])/len(y_validate)) * 100))
print("Test rating_5: {0} ({1:0.2f}%)".format(len(y_validate[y_validate[:] == 5.0]), (len(y_validate[y_validate[:] == 5.0])/len(y_validate)) * 100))
print("")


# In[110]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score


# In[196]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)
regression_model.coef_[0:]


# In[199]:


print(regression_model.score(X_train, y_train))
print(regression_model.score(X_validate, y_validate))


# In[197]:


ridge = Ridge(alpha=.3)
ridge.fit(X_train,y_train)
print ("Ridge model:", (ridge.coef_))


# In[200]:


print(ridge.score(X_train, y_train))
print(ridge.score(X_validate, y_validate))


# In[198]:


lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print ("Lasso model:", (lasso.coef_))


# In[201]:


print(lasso.score(X_train, y_train))
print(lasso.score(X_validate, y_validate))


# In[277]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2, interaction_only=True)


# In[265]:


X_poly = poly.fit_transform(X)
X_train_poly, X_validate_poly, y_train_poly, y_validate_poly = train_test_split(X_poly, y, test_size=0.30, random_state=1)
X_train_poly.shape


# In[266]:


regression_model.fit(X_train_poly, y_train_poly)
print(regression_model.score(X_train_poly, y_train_poly))
print(regression_model.score(X_validate_poly, y_validate_poly))


# In[267]:


ridge = Ridge(alpha=.3)
ridge.fit(X_train_poly,y_train_poly)
print(ridge.score(X_train_poly, y_train_poly))
print(ridge.score(X_validate_poly, y_validate_poly))


# In[268]:


lasso = Lasso(alpha=0.01)
lasso.fit(X_train_poly,y_train_poly)
print(lasso.score(X_train_poly, y_train_poly))
print(lasso.score(X_validate_poly, y_validate_poly))


# In[214]:


X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.3, random_state=1) 


# In[216]:


from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=1, max_depth=3)

regressor.fit(X_train , y_train)
feature_importances = regressor.feature_importances_


feature_names = X_train.columns
print(feature_names)

k = 8

print(feature_importances)
top_k_idx = (feature_importances.argsort()[-k:][::-1])

print(feature_names[top_k_idx], feature_importances[top_k_idx])


# In[218]:


dTree = DecisionTreeClassifier(random_state=1, max_depth=5)

dTree.fit(X_train , y_train)
feature_importances = dTree.feature_importances_


feature_names = X_train.columns
print(feature_names)

k = 8

print(feature_importances)
top_k_idx = (feature_importances.argsort()[-k:][::-1])

print(feature_names[top_k_idx], feature_importances[top_k_idx])


# In[220]:


dcTree = DecisionTreeClassifier()
dcTree.fit(X_train , y_train)
print (pd.DataFrame(dcTree.feature_importances_, columns = ["Imp"], index = X_train.columns, ))


# In[221]:


clf = RandomForestClassifier(n_estimators=50)


# In[223]:


# specify parameters and distributions to sample from
from scipy.stats import randint as sp_randint
param_dist = {"max_depth": [3, 5, 7, None],
              "max_features": sp_randint(1, 20),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


# In[225]:


# run randomized search
from sklearn.model_selection import RandomizedSearchCV
samples = 10  # number of random samples 
randomCV = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=samples) #default cv = 3


# In[226]:


randomCV.fit(X_train, y_train) 
print(randomCV.best_params_)


# In[227]:


randomCV.best_estimator_


# In[259]:


rfcl = RandomForestClassifier(bootstrap=False, class_weight=None, criterion='gini',
                       max_depth=None, max_features=6, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=50,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)


# In[260]:


rfcl.fit(X_train, y_train) 
rfcl.score(X_train, y_train)


# In[261]:


rfcl.fit(X_validate, y_validate) 
rfcl.score(X_validate, y_validate)


# In[237]:


X_train.shape


# In[282]:


df_test.shape


# In[283]:


df_test['greatstone_rating'] = df_test['greatstone_rating'].astype(str)


# In[284]:


df_test = pd.get_dummies(df_test, columns=['fund_size','investment_class'])
df_test.set_index('fund_id', inplace = True)


# In[285]:


df_test.drop(columns =['tag', 'category', 'parent_company','fund_name','currency', 'inception_date', 'us_govt_bond_rating'], inplace = True)


# In[242]:


df_test.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\Test_file.csv')


# In[286]:


x_test = df_test.drop(columns='greatstone_rating',axis=1)    # Predictors
y_test = df_test.loc[:,'greatstone_rating'] # target


# In[287]:


y_test.dtype


# In[288]:


X1_test = x_test.select_dtypes(include ='object')
X1_test.shape


# In[289]:


X_test = x_test.apply(zscore)
X_test.shape


# In[290]:


y_test = rfcl.predict(X_test)


# In[291]:


y_test


# In[292]:


y_test.shape


# In[293]:


df_pred = pd.DataFrame(y_test,index=X_test.index)


# In[294]:


df_pred.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\Target2_file.csv')


# In[295]:


X_test_poly = poly.fit_transform(X_test)


# In[296]:


X_test_poly.shape


# In[297]:


y_test2 = lasso.predict(X_test_poly)


# In[298]:


y_test2


# In[273]:


df_pred2 = pd.DataFrame(y_test2,index=X_test.index)


# In[274]:


df_pred2.to_csv(r'C:\Users\CHANDRIMA CHATTARAJ\Documents\Python Scripts\Python Workbooks\Hackathon\external\Target3_file.csv')


# In[316]:


from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca.fit(X)


# In[317]:


print(pca.components_)


# In[318]:


print(pca.explained_variance_ratio_)


# In[319]:


plt.bar(list(range(1,51)),pca.explained_variance_ratio_,alpha=0.5, align='center')
plt.ylabel('Variation explained')
plt.xlabel('eigen Value')
plt.show()


# In[320]:


plt.step(list(range(1,51)),np.cumsum(pca.explained_variance_ratio_), where='mid')
plt.ylabel('Cum of variation explained')
plt.xlabel('eigen Value')
plt.show()


# In[321]:


pca50 = PCA(n_components=50)
pca50.fit(X)
print(pca50.components_)
print(pca50.explained_variance_ratio_)
Xpca50 = pca50.transform(X)


# In[322]:


Xpca_train, Xpca_validate, ypca_train, ypca_validate = train_test_split(Xpca50, y, test_size=0.3, random_state=1)
print("Size of the training dataframe is ",len(Xpca_train))
print("Size of the test dataframe is ",len(Xpca_validate))
print("{0:0.2f}% data is in training set".format((len(Xpca_train)/len(gr_df.index)) * 100))
print("{0:0.2f}% data is in test set".format((len(Xpca_validate)/len(gr_df.index)) * 100))


# In[324]:


def fit_n_print(model, Xtrain, Xtest, ytrain, ytest):  # take the model, and data as inputs
        
    model.fit(Xtrain, ytrain)   # fir the model with the train data
    
    score_train = round(model.score(Xtrain, ytrain), 3) 
    score_validate = round(model.score(Xtest, ytest), 3)   # compute accuracy score for test set
    
    return score_train,score_validate


# In[330]:


result = pd.DataFrame({'Model':[],'Accuracy_train':[], 'Accuracy_val':[]})
accuracy_train,accuracy_validate = fit_n_print(SVC(), Xpca_train, Xpca_validate, ypca_train, ypca_validate)
result = result.append(pd.Series({'Model':'SVC_pca', 'Accuracy_train':accuracy_train, 'Accuracy_val':accuracy_validate}), ignore_index=True)
accuracy_train,accuracy_validate = fit_n_print(SVC(C=3), Xpca_train, Xpca_validate, ypca_train, ypca_validate)
result = result.append(pd.Series({'Model':'SVC_pca_C3', 'Accuracy_train':accuracy_train, 'Accuracy_val':accuracy_validate}), ignore_index=True)
accuracy_train,accuracy_validate = fit_n_print(SVC(kernel = 'linear'), Xpca_train, Xpca_validate, ypca_train, ypca_validate)
result = result.append(pd.Series({'Model':'SVC_pca_linear', 'Accuracy_train':accuracy_train, 'Accuracy_val':accuracy_validate}), ignore_index=True)
accuracy_train,accuracy_validate = fit_n_print(SVC(kernel = 'linear', C=3), Xpca_train, Xpca_validate, ypca_train, ypca_validate)
result = result.append(pd.Series({'Model':'SVC_pca_linear_C3', 'Accuracy_train':accuracy_train, 'Accuracy_val':accuracy_validate}), ignore_index=True)
print(result)


# In[340]:


pipelines=[]
pipelines.append(('Logisitic Regression',Pipeline([('LogisticRegression',LogisticRegression())])))
pipelines.append(('KNN',Pipeline([('KNN',KNeighborsClassifier())])))
pipelines.append(('Naive bayes',Pipeline([('scaled Naive Bayes',GaussianNB())])))
pipelines.append(('DecisionTree',Pipeline([('decision',DecisionTreeClassifier(random_state = 0))])))


# In[341]:


result2 = pd.DataFrame({'Model':[],'Accuracy_train':[], 'Accuracy_val':[]})
for name,model in pipelines:
    accuracy_train,accuracy_validate = fit_n_print(model, Xpca_train, Xpca_validate, ypca_train, ypca_validate )
    result2 = result2.append(pd.Series({'Model':name, 'Accuracy_train':accuracy_train, 'Accuracy_val':accuracy_validate}), ignore_index=True)
print(result2)


# In[339]:


# List of values to try for max_depth:
max_depth_range = list(range(1, 20))
# List to store the average RMSE for each value of max_depth:
accuracy = []

for depth in max_depth_range:
    
    clf = DecisionTreeClassifier(max_depth = depth, criterion='entropy', max_features=10,
                             random_state = 0)
    clf.fit(Xpca_train, ypca_train)
    score = clf.score(Xpca_validate, ypca_validate)
    accuracy.append(score)   
    print(depth,score)


# In[343]:


from sklearn.cluster import KMeans

ks = range(1, 10)
inertias = [] # initializing an empty array

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


# In[344]:


def k_means(n_clust, data_frame, true_labels):
    """
    Function k_means applies k-means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels 
    and clustering performance parameters.
    
    Input:
    n_clust - number of clusters (k value)
    data_frame - dataset we want to cluster
    true_labels - original labels
    
    Output:
    1 - crosstab of cluster and actual labels
    2 - performance table
    """
    k_means = KMeans(n_clusters = n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
    %(k_means.inertia_,
      homogeneity_score(true_labels, y_clust),
      completeness_score(true_labels, y_clust),
      v_measure_score(true_labels, y_clust),
      adjusted_rand_score(true_labels, y_clust),
      adjusted_mutual_info_score(true_labels, y_clust),
      silhouette_score(data_frame, y_clust, metric='euclidean')))


# In[ ]:


k_means = KMeans(n_clusters = 2, random_state=123, n_init=30)
k_means.fit(data_frame)

