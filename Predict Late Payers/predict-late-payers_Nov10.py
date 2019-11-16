#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
# import statsmodels.api as sm

# Preprocessing and Pipeline libraries
# from sklearn.compose import ColumnTransformer
# from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle


# In[2]:


pd.options.display.max_rows = 1500
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


# In[3]:


# INPUT_DIR = 'Data/'
# tr = pd.read_csv(INPUT_DIR + 'peerLoanTest.csv')


# In[4]:


print(f"\nLoading training data...")
# load training data
train_data = pd.read_csv("Data/peerLoanTraining.csv", engine='python', header=0)

# Separate out X and y
X_train = train_data.loc[:, train_data.columns != 'is_late']
y_train = train_data['is_late']

print("\nLoading test data...")
# load test data
test_data = pd.read_csv("Data/peerLoanTest.csv", engine='python', header=0)

# Separate out X and y
X_test = test_data.loc[:, test_data.columns != 'is_late']
y_test = test_data['is_late']

print("\nLoaded !!")


# In[5]:


def SummaryTable(df):
    print('This dataset has ' + str(df.shape[0]) + ' rows, and ' + str(df.shape[1]) + ' columns')
    print("\n","TOP FEW OBSERVATIONS:")
    print(display(df.head(5)))
    print("\n","BOTTOM FEW OBSERVATIONS:")
    print(display(df.tail(5)))
    print("\n","SUMMARY of Quantitative Data:")
    print(display(df.describe()),"\n")
    summary = pd.DataFrame(df.dtypes,columns=['DataType'])
    summary = summary.reset_index()
    summary['VariableName'] = summary['index']
    summary = summary[['VariableName','DataType']]
    summary['Missing'] = df.isnull().sum().values
    summary['MissingPercentage'] = (summary['Missing']/len(df)*100).round(2)
    summary['Uniques'] = df.nunique().values
#     summary['First Value'] = df.loc[0].values
#     summary['Second Value'] = df.loc[1].values
#     summary['Third Value'] = df.loc[2].values
#     summary['Fourth Value'] = df.loc[3].values
#     summary['Fifth Value'] = df.loc[4].values
    
#     for name in summary['VariableName'].value_counts().index:
#         summary.loc[summary['VariableName'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 
    categorical_features = df.select_dtypes(include = np.object)
    print("Frequency of Categorical Data:","\n")
    for i in categorical_features.columns:
        print(i + ":" + str(categorical_features[i].nunique()))
        print(categorical_features[i].value_counts())
        print('\n')
    
    return summary


# In[ ]:


print("Summary of Training Data:")
SummaryTable(X_train)


# In[ ]:


# Check for duplicate data
columns_without_id = [col for col in X_train.columns if col!='SK_ID_CURR']
#Checking for duplicates in the data.
X_train[X_train.duplicated(subset = columns_without_id, keep=False)]
print('The no of duplicates in the data:',X_train[X_train.duplicated(subset = columns_without_id, keep=False)].shape[0])


# In[ ]:


SummaryTable(X_test)


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "import warnings\nimport multiprocessing\nwarnings.simplefilter('ignore')\nfiles = ['Data/peerLoanTraining.csv', \n         'Data/peerLoanTest.csv']\n\ndef load_data(file):\n    return reduce_mem_usage(pd.read_csv(file))\n\nwith multiprocessing.Pool() as pool:\n    peerLoanTraining, peerLoanTest = pool.map(load_data, files)")


# In[ ]:


SummaryTable(peerLoanTraining)


# In[ ]:


def corr1(col):
    N = None #10000
    num_vars = [f for f in tr.columns if tr[f].dtype != 'object']
    trx = tr.head(N) if N is not None else tr.copy()
    corrs = trx[num_vars].corrwith(trx[col]).reset_index().sort_values(0, ascending=False).reset_index(drop=True).rename({'index':'Column',0:'Correlation with ' + col}, axis=1)
    h('<b>Most correlated values with ' + col + ':</b>')
    trx = pd.concat([corrs.head(6), corrs.dropna().tail(5)])
    def linkx(val):
        return '<a href="#c_{}">{}</a>'.format(val, val) if val in included_cols else val
    trx['Column'] = trx['Column'].apply(linkx)
    h(trx.to_html(escape=False))


# In[ ]:


def corr1(col):
    N = None #10000
    num_vars = [f for f in tr.columns if tr[f].dtype != 'object']
    trx = tr.head(N) if N is not None else tr.copy()
    corrs = trx[num_vars].corrwith(trx[col]).reset_index().sort_values(0, ascending=False).reset_index(drop=True).rename({'index':'Column',0:'Correlation with ' + col}, axis=1)
    h('<b>Most correlated values with ' + col + ':</b>')
    trx = pd.concat([corrs.head(6), corrs.dropna().tail(5)])
    def linkx(val):
        return '<a href="#c_{}">{}</a>'.format(val, val) if val in included_cols else val
    trx['Column'] = trx['Column'].apply(linkx)
    h(trx.to_html(escape=False))


# #### Missing Value Analysis
# * revol_util - Use Median
# * emp_length - Create a new category

# In[ ]:


revol_util_median = X_train['revol_util'].median()
X_train['revol_util']=X_train['revol_util'].fillna(revol_util_median)
X_test['revol_util']=X_test['revol_util'].fillna(revol_util_median)

# emp_length_mode = X_train['emp_length'].mode()
# X_train['emp_length']=X_train['emp_length'].fillna(emp_length_mode[0])
# X_test['emp_length']=X_test['emp_length'].fillna(emp_length_mode[0])

X_train['emp_length']=X_train['emp_length'].fillna("Missing")
X_test['emp_length']=X_test['emp_length'].fillna("Missing")


# #### Categorical Data 
# * Grade & emp_length are Ordinal vars
# * Grade is related to Interest rate
# * 
# 

# In[ ]:


# purpose, grade, emp_length, home_ownership are Categorical vars
categorical_features = X_train.select_dtypes(include = np.object)


# In[ ]:


# A [0.5,0.08], B 0.06,[0.0943,0.1213], C 0.06,[0.1261,0.1646], D 0.06,[0.1709,0.2185], E 0.06,[0.229,0.2677], F [0.2872,0.3075], G [0.3079,0.3099]
pd.crosstab(X_train['grade'],X_train['int_rate'])


# In[ ]:


# Grade is converted to Continuous using Ordinal technique
grade = sorted(list(set(X_train['grade'].values)))
grade = dict(zip(grade, [x+1 for x in range(len(grade))]))
X_train.loc[:, 'grade'] = X_train['grade'].apply(lambda x: grade[x]).astype(int)
X_test.loc[:, 'grade'] = X_test['grade'].apply(lambda x: grade[x]).astype(int)


# In[ ]:


chart = sns.boxplot(x="grade", y="int_rate", data=X_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=85)


# In[ ]:


chart = sns.boxplot(x="purpose", y="loan_amnt", data=X_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=85)


# In[ ]:


chart = sns.boxplot(x="purpose", y="loan_amnt", data=X_train)
chart.set_xticklabels(chart.get_xticklabels(), rotation=85)


# In[ ]:


grade = sorted(list(set(X_train['grade'].values)))
grade = dict(zip(grade, range(len(grade))))


# In[ ]:





# In[ ]:





# In[ ]:


SummaryTable(X_train)


# #### Target Label Viz

# In[13]:


# Get number of positve and negative examples
pos = train_data[train_data["is_late"] == 1].shape[0]
neg = train_data[train_data["is_late"] == 0].shape[0]
print(f"Positive examples = {pos}")
print(f"Negative examples = {neg}")
print(f"Proportion of positive to negative examples = {(pos / neg) * 100:.2f}%")

plt.figure(figsize=(8, 6))
sns.countplot(train_data["is_late"])
plt.xticks((0, 1), ["Paid", "Not paid"])
plt.xlabel("")
plt.ylabel("Count")
plt.title("Class counts", y=1, fontdict={"fontsize": 20});


# In[20]:


print("FREQUENCY OF TARGET: \n",pd.value_counts(y_train))

plt.subplot(121)
plot_tr = sns.countplot(train_data["is_late"])
plot_tr.set_title("Fraud Transactions Distribution \n 0: Not Late | 1: Late", fontsize=12)
plot_tr.set_xlabel("Is Late?", fontsize=8)
plot_tr.set_ylabel('Count', fontsize=8)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/(len(train_data))*100),
            ha="center", fontsize=10) 


# In[10]:


print("FREQUENCY OF TARGET: \n",pd.value_counts(y_train))
y_train.value_counts().plot.bar()


# In[22]:


print("Distribution of data points among output class\n")

print(pd.value_counts(y_train))

import matplotlib.pyplot as plt

# The slices will be ordered and plotted counter-clockwise.
labels = train_data["is_late"].value_counts().index
sizes = train_data["is_late"].value_counts().values
# colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0, 0, 0)  # explode a slice if required

plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.axis('equal')
plt.show()  


# # Data is HIGHLY IMBALANCED

# In[ ]:





# In[ ]:


import category_encoders as ce


# In[ ]:





# In[ ]:


X_train = X_train.assign(ID=pd.Series(range(1,len(X_train))))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


numerical_features = X_train.select_dtypes(include = np.number)
categorical_features = X_train.select_dtypes(include = np.object)
print("Numerical Features:", numerical_features.shape[1] ,"\n", numerical_features.columns)
print('\n')
print("Cateogrical Features:", categorical_features.shape[1] ,"\n", categorical_features.columns)


# In[ ]:


numerical_features.columns


# In[ ]:


def woe(X, y, cont=True):
    tmp = pd.DataFrame()
    tmp["variable"] = X
    if cont:
        tmp["variable"] = pd.qcut(tmp["variable"], 255, duplicates="drop")
    tmp["target"] = y
    var_counts = tmp.groupby("variable")["target"].count()
    var_events = tmp.groupby("variable")["target"].sum()
    var_nonevents = var_counts - var_events
    tmp["var_counts"] = tmp.variable.map(var_counts)
    tmp["var_events"] = tmp.variable.map(var_events)
    tmp["var_nonevents"] = tmp.variable.map(var_nonevents)
    events = sum(tmp["target"] == 1)
    nonevents = sum(tmp["target"] == 0)
    tmp["woe"] = np.log(((tmp["var_nonevents"])/nonevents)/((tmp["var_events"])/events))
    tmp["woe"] = tmp["woe"].replace(np.inf, 0).replace(-np.inf, 0)
    tmp["iv"] = (tmp["var_nonevents"]/nonevents - tmp["var_events"]/events) * tmp["woe"]
    iv = tmp.groupby("variable")["iv"].last().sum()
    return tmp["woe"], tmp["iv"], iv


# ## Numerical
# * Apply long transformations on right skewed data
# 

# In[ ]:


numerical_features = train_data.select_dtypes(include = np.number)


# In[ ]:


# Histogram for Numerical Vars
fig, axes = plt.subplots(1,7 , figsize = (16,4))
fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
axes = axes.ravel()
for i,j in zip([i for i in numerical_features.columns[0:]], range(10)):
    axes[j].hist(numerical_features[i])
    axes[j].set_title(i+' skew: '+str(np.round(numerical_features[i].skew(),2)))


# In[ ]:


# Numerical Vars
for i in numerical_features.columns: #['loan_amnt','int_rate']
    sns.FacetGrid(train_data, hue='is_late', size=4)         .map(sns.distplot, i)         .add_legend()
    plt.show()


# In[ ]:


# Box Plot
for i in numerical_features:
    plt.figure()
    plt.clf()
    sns.boxplot(numerical_features[i])
    plt.title(i)
    plt.show()


# In[ ]:


# Box Plot of Numerical Variable by Target Variable
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,10))
AX = ax.ravel()

for i,j in zip([i for i in numerical_features.columns[0:]],AX):
    sns.boxplot(x = 'is_late', y = i, data =numerical_features,ax=j)


# ## Categorical Data

# In[ ]:


categorical_features = X_train.select_dtypes(include = np.object)


# In[ ]:


# Cross Tab of Categorical Vars by Target Variable: Freq & Percentage
for i in categorical_features.columns:
    print(i + ":" + str(categorical_features[i].nunique()))
#     print(pd.concat([pd.crosstab(train_data[i], train_data['is_late'], margins=True),
#                    pd.crosstab(train_data[i], train_data['is_late'], normalize='index')]))
    Df = pd.concat([pd.DataFrame(pd.crosstab(train_data[i], train_data['is_late'], margins=True)),
                    pd.DataFrame(train_data[i].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'),
                   pd.DataFrame(pd.crosstab(train_data[i], train_data['is_late'], normalize='index')).mul(100).round(1).astype(str) + '%'], axis=1, sort=False)
    Df.columns = ['Target_0', 'Target_1', 'Freq','FreqPercent',"Target_Percent0","Target_Percent1"]
    
    print(Df)
    print('\n')


# In[ ]:


# Frequency table for each catergorical variable
# Cross tab of Categorical variable vs Target Variable
for i in categorical_features.columns:
    print(i + ":" + str(categorical_features[i].nunique()))
    print(categorical_features[i].value_counts(), "\n",pd.crosstab(train_data[i], train_data['is_late']))
    print('\n')


# In[ ]:


# (Percentage of 1s & 0s with each category of Categorical varible)
# Cross Tab - Categorical Variable by Target Variable (Stacked Bar Plot) 
fig, axes = plt.subplots(2,2, figsize = (16,8))
axes = axes.ravel()

for i,j in zip([i for i in categorical_features.columns], axes):
    temp = pd.crosstab(train_data[i], train_data['is_late'], normalize='index')
    temp.plot(kind = 'bar', stacked = True, color = ['red', 'green'] , grid = False, ax = j)


# In[ ]:


# Frequncy - Bar Graph for Categorical Variables
for i in categorical_features: #['purpose','grade']
    chart = sns.countplot(data=train_data,x=i)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


# In[ ]:


# Cross Tab - Bar Graph for Categorical Variables by Target Variable
for i in categorical_features: #['purpose','grade']
    chart = sns.countplot(data=train_data,x=i, hue="is_late") #, order=train_data[i]
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


# ### Observations
# * Purpose
#     * Within is_late - debt_consilation has more number of 1 but within categories - small business have higher % of 1
# * Grade
#     * Within is_late - C has more number of 1 but within categories - F have higher % of 1

# In[ ]:


# For is_late=0 is considered population and within it what are the proportions for various catergories within a categorical variable
for i in categorical_features: #['purpose','grade']
    chart = train_data[i].groupby(train_data["is_late"]).value_counts(normalize=True).rename("proportion").reset_index().pipe((sns.barplot, "data"), x=i, y="proportion", hue="is_late")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


# In[ ]:


# Cross Tab - Categorical Variable by Target Variable (Stacked Bar Plot)
fig, axes = plt.subplots(2,2, figsize = (16,8))
axes = axes.ravel()

for i,j in zip([i for i in categorical_features.columns], axes):
    temp = pd.crosstab(train_data[i], train_data['is_late'])
    temp.plot(kind = 'bar', stacked = True, color = ['red', 'green'] , grid = False, ax = j)


# In[ ]:


import pandas_profiling as pp


# In[ ]:





# In[ ]:




