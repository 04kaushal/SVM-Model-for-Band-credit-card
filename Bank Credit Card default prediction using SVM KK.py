#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\ezkiska\Videos\Imarticus\Python\5th Week 4th & 5th Jan\SVM  project band credit")
data_original = pd.read_csv('BankCreditCard.csv')


# In[2]:


data = data_original


# In[3]:


data.head()


# In[4]:


# list all columns (for reference)
data.columns


# In[5]:


#  Default_Payment (response)
data = data.drop(['Customer ID'], axis = 1)
data.isnull().sum()


# In[6]:


data.describe()


# In[7]:


catCols = ["Gender", "Academic_Qualification", "Marital", "Repayment_Status_Jan", "Repayment_Status_Feb",
            "Repayment_Status_March", "Repayment_Status_April", "Repayment_Status_May", "Repayment_Status_June",
            "Default_Payment"]


# ## EDA

# In[8]:


## Target variable -------------------------------------------------------------------

sns.countplot(x=data['Default_Payment'])


# In[9]:


# percentage of 0's and 1's
np.round(data.Default_Payment.value_counts()/data.shape[0],2)
# to get percentage simply divide feature value count with data shape or dimension and we can also round off
## imbalanced data: We observe that data is imbalanced here


# In[10]:


# """EDA categorical independent features"""# ------------------------------------------------


# In[12]:



# Set up the matplotlib figure
sns.set(style='darkgrid', palette='RdBu', font='sans-serif', font_scale=1, color_codes=True, rc=None)

#'''https://seaborn.pydata.org/generated/seaborn.set.html'''

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False) #Controls sharing of properties among x (sharex) 
#or y (sharey) axes:

#'''  True or 'all': x- or y-axis will be shared among all subplots.
#False or 'none': each subplot x- or y-axis will be independent.
#'row': each subplot row will share an x- or y-axis.
#'col': each subplot column will share an x- or y-axis.'''

sns.despine(left=False)

#'''seaborn.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
#Remove the top and right spines from plot(s).'''

sns.countplot(x=data['Gender'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Marital'],  hue = 'Default_Payment', data = data, ax = axes[0][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[13]:


#"""  BOX PLOT APPROACH for above but to show X vs Y i.e. Dependent vs Independent variable  """

sns.set(style='darkgrid', palette='RdBu', font='sans-serif', font_scale=1, color_codes=True, rc=None)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False) #Controls sharing of properties among x (sharex) 
sns.despine(left=False)
sns.barplot(x=data['Gender'], y=data['Default_Payment'], data = data, ax = axes[0][0])
sns.barplot(x=data['Academic_Qualification'], y=data['Default_Payment'], data = data, ax = axes[1][0])
sns.barplot(x=data['Marital'], y=data['Default_Payment'], data = data, ax = axes[0][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# Now We will create Function to check proportions of our data for each categorical variable 

# In[14]:


def createProportions(df,colName, dependentColName):
    
    tab = pd.crosstab(df[colName],  df[dependentColName],margins = False)
    prop = []
    for i in range(tab.shape[0]):
        value = tab.iloc[i,1]/tab.iloc[i,0]
        prop.append(value)
    tab['prop'] = prop

    return tab


# In[15]:


createProportions(data,'Marital', 'Default_Payment')


# In[16]:


createProportions(data,'Academic_Qualification', 'Default_Payment')


# In[17]:


createProportions(data,'Gender', 'Default_Payment')


# In[18]:


# Set up the matplotlib figure
sns.set(style="white", palette="RdBu", color_codes=True)
f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=False)
sns.despine(left=True)
sns.countplot(x=data['Repayment_Status_Jan'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Repayment_Status_Feb'], hue = 'Default_Payment', data = data, ax = axes[0][1])
sns.countplot(x=data['Repayment_Status_March'],  hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Repayment_Status_April'], hue = 'Default_Payment', data = data, ax = axes[1][1])
sns.countplot(x=data['Repayment_Status_May'], hue = 'Default_Payment', data = data, ax = axes[2][0])
sns.countplot(x=data['Repayment_Status_June'],  hue = 'Default_Payment', data = data, ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[19]:


createProportions(data,'Repayment_Status_Jan', 'Default_Payment')


# In[20]:


createProportions(data,'Repayment_Status_Feb', 'Default_Payment')


# In[21]:


createProportions(data,'Repayment_Status_March', 'Default_Payment')


# In[22]:


createProportions(data,'Repayment_Status_April', 'Default_Payment')


# In[23]:


createProportions(data,'Repayment_Status_May', 'Default_Payment')


# In[24]:


createProportions(data,'Repayment_Status_June', 'Default_Payment')


# In[25]:


# group the academic levels as 1,2 and >= 3  OR 0(1) and 1(2,3,4) but 2nd technique will not be proper 
# due to distribution of target values in 2 and 3, let's see what happens afetr technique 1
# 
''' So We will create a clubbing fuctions and club the levels
0,1 will remain and 2 on owards will be clubbed as 2 itself for all 
Rapayment status months'''
#


# In[26]:


def club(df, feature, a, b, newValue):
    
    for i in range(a, b):
        df[feature][df[feature] == i] = newValue
    
    x = df[feature].value_counts()
    
    return x

def labelCh(df, feature, a, b):
    
    for i in range(a, b):
        df[feature][df[feature] == i] = i-1
    
    x = df[feature].value_counts()
    
    return x


# In[27]:


club(data, 'Repayment_Status_Jan', 2, 7, 1)


# In[28]:


club(data, 'Repayment_Status_Feb', 2, 7, 1)


# In[29]:


club(data, 'Repayment_Status_March', 2, 7, 1)


# In[30]:


club(data, 'Repayment_Status_April', 2, 7, 1)


# In[31]:


club(data, 'Repayment_Status_May', 2, 7, 1)


# In[32]:


club(data, 'Repayment_Status_June', 2, 7, 1)


# In[33]:


print(data['Academic_Qualification'].value_counts())


# In[34]:


club(data, 'Academic_Qualification', 4, 7, 3)


# In[35]:


# Academic qualification has been clubbed and reduced to 3 levels, 1& 2 from initial values
#while 3 to 6 blevels have been clubbed to 3
sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data)
plt.tight_layout()


# In[36]:


data[['Academic_Qualification', 'Default_Payment']].groupby(['Academic_Qualification'],
    as_index=False).mean().sort_values(by='Academic_Qualification', ascending=True)


# In[37]:


createProportions(data,'Academic_Qualification', 'Default_Payment') #call the function from above to check proportion


# In[38]:


# Here percentage conversion is the same for 2 and 3 levels
# group the academic levels as 0(1) and 1(2,3): makes sense post grad 
#approx.== professionals and elite classes
data['Academic_Qualification'][data['Academic_Qualification']==1] = 0
club(data, 'Academic_Qualification', 2, 4, 1)


# In[39]:


'''Working on Gender Encoding'''
# group the Gender levels as 0 and 1 instead of 1 and 2
labelCh(data, 'Gender', 1, 3)


# In[40]:


sns.countplot(x=data['Gender'], hue = 'Default_Payment', data = data)
plt.tight_layout()


# In[41]:


data[['Gender', 'Default_Payment']].groupby(['Gender'],
    as_index=False).mean().sort_values(by='Gender', ascending=True)


# In[42]:


'''Working on Marital Status Categorical data'''
# group the marital levels as (0,1) as 0 (due to proprotion of blue)
# and (2,3) as 1; (due to proprotion of red)
print(data['Marital'].value_counts())
club(data, 'Marital', 1, 2, 0)
club(data, 'Marital', 2, 4, 1)


# In[43]:


sns.countplot(x=data['Marital'], hue = 'Default_Payment', data = data)
plt.tight_layout()


# In[44]:


data[['Marital', 'Default_Payment']].groupby(['Marital'],
    as_index=False).mean().sort_values(by='Marital', ascending=True)
'''After all the changes on Categorical plots We will summarize all the plots'''


# In[45]:


## Rechecking the plots
# Set up the matplotlib figure
sns.set(style='darkgrid', palette='muted', font='sans-serif', font_scale=1, color_codes=True, rc=None)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=False) #Controls sharing of properties among x (sharex) 
sns.despine(left=False)
sns.countplot(x=data['Gender'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Academic_Qualification'], hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Marital'],  hue = 'Default_Payment', data = data, ax = axes[0][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[46]:


# Set up the matplotlib figure
sns.set(style='darkgrid', palette='husl', font='sans-serif', font_scale=1, color_codes=True, rc=None)
f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex=False) #Controls sharing of properties among x (sharex) 
sns.despine(left=False)
sns.countplot(x=data['Repayment_Status_Jan'], hue = 'Default_Payment', data = data, ax = axes[0][0])
sns.countplot(x=data['Repayment_Status_Feb'], hue = 'Default_Payment', data = data, ax = axes[0][1])
sns.countplot(x=data['Repayment_Status_March'],  hue = 'Default_Payment', data = data, ax = axes[1][0])
sns.countplot(x=data['Repayment_Status_April'], hue = 'Default_Payment', data = data, ax = axes[1][1])
sns.countplot(x=data['Repayment_Status_May'], hue = 'Default_Payment', data = data, ax = axes[2][0])
sns.countplot(x=data['Repayment_Status_June'],  hue = 'Default_Payment', data = data, ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[47]:


'''Now we will drop the dependent variable from list of categorical features'''

dataCAT = data[catCols].drop(['Default_Payment'], axis = 1)


# '''Let's Look at Numerical features now '''

# In[48]:


### EDA - numerical features ----------------------------------------------------------------------
cols = data.columns.tolist()
numCols = [cols[i] for i in range(len(cols)) if i == 0 or i == 4] #names(data3)[c(1,5)
numCols2 = cols[11:-1] # Returns columns 12:23)] from data 
numCols.extend(numCols2)


# In[49]:


'''Python list extend() is an inbuilt function that adds the specified list elements 
(or any iterable) to the end of the current list. 
The extend() extends the list by adding all items of the list (passed as an argument) to an end'''

dataNUM = data[numCols]


# In[50]:


'''We will now see univariate Analysis on Numerical Data'''

# Univariate analysis
# Set up the matplotlib figure
'''  Credit_amount till June_bill_amount '''
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 3, figsize=(7, 7), sharex=False)
sns.despine(left=True)
sns.distplot(dataNUM.iloc[:,0], ax = axes[0][0]) #dist plot all rown and a particular column here  0 which is credit Amount
sns.distplot(dataNUM.iloc[:,1], ax = axes[0][1])
sns.distplot(dataNUM.iloc[:,2], ax = axes[0][2])
sns.distplot(dataNUM.iloc[:,3], ax = axes[1][0])
sns.distplot(dataNUM.iloc[:,4], ax = axes[1][1])
sns.distplot(dataNUM.iloc[:,5], ax = axes[1][2])
sns.distplot(dataNUM.iloc[:,6], ax = axes[2][0])
sns.distplot(dataNUM.iloc[:,7], ax = axes[2][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[51]:


''' All previous_payments  '''
sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 3, figsize=(7, 7), sharex=False)
sns.despine(left=True)
sns.distplot(dataNUM.iloc[:,8], ax = axes[0][0])
sns.distplot(dataNUM.iloc[:,9], ax = axes[0][1])
sns.distplot(dataNUM.iloc[:,10], ax = axes[0][2])
sns.distplot(dataNUM.iloc[:,11], ax = axes[1][0])
sns.distplot(dataNUM.iloc[:,12], ax = axes[1][1])
sns.distplot(dataNUM.iloc[:,13], ax = axes[1][2])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[52]:


''' Let Us check Auto Correlattion between Numerical Features'''
# Credit, Jan-Bill, March-bill, April-bill, all Previous_payments
#sns.heatmap(dataNUM, cmap="YlGnBu")
corr = dataNUM.corr()
ax = sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[53]:


#correlation is high among Bill_Amount, not Previous_Payments
dataNUM[['Jan_Bill_Amount','Feb_Bill_Amount']].corr().iloc[1,0]


# In[54]:


#drop all but June_Bills and June Previous_Payments
dataNUM.columns
# dropped columns are more
dataNUM = dataNUM[['Credit_Amount', 'Age_Years','June_Bill_Amount', 'Previous_Payment_June']]


# In[55]:


''''Now we will do outlier check'''

sns.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)
sns.boxplot(y = dataNUM.iloc[:,0], x = data['Default_Payment'], data = data, ax = axes[0][0])
sns.boxplot(y = dataNUM.iloc[:,1], x = data['Default_Payment'], data = data, ax = axes[0][1])
sns.boxplot(y = dataNUM.iloc[:,2], x = data['Default_Payment'], data = data, ax = axes[1][0])
sns.boxplot(y = dataNUM.iloc[:,3], x = data['Default_Payment'], data = data, ax = axes[1][1])
plt.setp(axes, yticks=[])
plt.tight_layout()


# In[56]:


dataNUM_list = [dataNUM]

for dataset in dataNUM_list:
    dataset.loc[dataset.Credit_Amount < np.percentile(dataNUM['Credit_Amount'].values, 1), 
                'Credit_Amount' ] = np.percentile(dataNUM['Credit_Amount'].values, 1)
    dataset.loc[dataset.Credit_Amount > np.percentile(dataNUM['Credit_Amount'].values, 99), 
                'Credit_Amount' ] = np.percentile(dataNUM['Credit_Amount'].values, 99)
    
    dataset.loc[dataset.Age_Years < np.percentile(dataNUM['Age_Years'].values, 1), 
                'Age_Years' ] = np.percentile(dataNUM['Age_Years'].values, 1)
    dataset.loc[dataset.Age_Years > np.percentile(dataNUM['Age_Years'].values, 99),
                'Age_Years' ] = np.percentile(dataNUM['Age_Years'].values, 99)
    
    dataset.loc[dataset.June_Bill_Amount < np.percentile(dataNUM['June_Bill_Amount'].values, 1), 
                'June_Bill_Amount' ] = np.percentile(dataNUM['June_Bill_Amount'].values, 1)
    dataset.loc[dataset.June_Bill_Amount > np.percentile(dataNUM['June_Bill_Amount'].values, 99), 
                'June_Bill_Amount' ] = np.percentile(dataNUM['June_Bill_Amount'].values, 99)
    
    dataset.loc[dataset.Previous_Payment_June < np.percentile(dataNUM['Previous_Payment_June'].values, 1),
                'Previous_Payment_June' ] = np.percentile(dataNUM['Previous_Payment_June'].values, 1)
    dataset.loc[dataset.Previous_Payment_June > np.percentile(dataNUM['Previous_Payment_June'].values, 99),
                'Previous_Payment_June' ] = np.percentile(dataNUM['Previous_Payment_June'].values, 99)


# In[57]:


def clubLabelEncoder(df, feature, k):
    
    df[feature +'_band'] = pd.qcut(df[feature], k)
    x = df[feature + '_band'].value_counts().index.tolist()
    
    intervals = []
    for i in range(len(x)):
        leftInt = x[i].left
        rtInt = x[i].right
        intervals.append(leftInt)
        intervals.append(rtInt)
    
    intervals_ = sorted(list(set(intervals)))
    
    for i in range(len(intervals_)-1):
        
        df.loc[(df[feature] > intervals_[i]) & (df[feature] <= intervals_[i+1]), feature] = i
    
    df = df.iloc[:,:-1]
    
    return df[feature].value_counts()


# In[58]:


clubLabelEncoder(dataNUM, 'Age_Years', 4)
dataNUM['Age_Years'].value_counts()


# In[59]:


clubLabelEncoder(dataNUM, 'Credit_Amount', 4)
dataNUM['Credit_Amount'].value_counts()


# In[60]:


clubLabelEncoder(dataNUM, 'June_Bill_Amount', 4)
dataNUM['June_Bill_Amount'].value_counts()


# In[61]:


clubLabelEncoder(dataNUM, 'Previous_Payment_June', 4)
dataNUM['Previous_Payment_June'].value_counts()


# In[62]:


dataNUM = dataNUM.iloc[:,:-4] #removing four band columns from the last


# In[64]:


dataCAT.iloc[:,0].value_counts() # already brinary,  no need for dummy encoding


# In[65]:


df = pd.concat([dataCAT, dataNUM], axis = 1) # only independent deatures
df['Default_Payment'] = data['Default_Payment'] # add the target column


# In[66]:


#------------------------- rectified data for SVM, scaled numercial features-------------
dataNUM_ = data[['Credit_Amount', 'Age_Years', 'June_Bill_Amount', 'Previous_Payment_June']]
df_ = pd.concat([dataCAT, dataNUM_], axis = 1)
df_['Default_Payment'] = data['Default_Payment'] 


# In[67]:


## ---------------------------------------Modelling with SMOTE--------------------------------------------------------------------------------

'''SMOTE will take care of the oversampling stuff''' 

'''df_ has the numerical variables as they were originally along with the other discrete features #(clubbed them)'''
'''df has the categorized numerical features along with the other discrete features #(clubbed them)'''
'''df2 has the numerical variables as they were along with the other discrete features #(unclubbed)'''


from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE 
from sklearn.svm import SVC #Support vector classifier
from sklearn.metrics import roc_auc_score, roc_curve
import time


X = df_.iloc[:,0:-1]
y = df_['Default_Payment']

sm = SMOTE(random_state = 2) 
X_sm, y_sm = sm.fit_sample(X, y.ravel()) 

print('With imbalance treatment:'.upper())
print('Before SMOTE:',X.shape, y.shape)
print('After SMOTE, Xs: {}'.format(X_sm.shape)) 
print('After SMOTE, y: {}'.format(y_sm.shape)) 
print("After SMOTE, counts of '1': {}".format(sum(y_sm == 1))) 
print("After SMOTE, counts of '0': {}".format(sum(y_sm == 0))) 
print("Before SMOTE, counts of '1': {}".format(sum(y == 1))) 
print("Before SMOTE, counts of '0': {}".format(sum(y == 0))) 
print('\n')
print('*'*80)


#X_smU = [X_sm[i][:9] for i in range(len(X_sm))] #cat
#X_smS = [X_sm[i][9:] for i in range(len(X_sm))] #num


# In[68]:


X_sm.info()


# In[69]:


X_smU= X_sm[['Gender','Academic_Qualification','Marital','Repayment_Status_Jan','Repayment_Status_Feb',
             'Repayment_Status_March','Repayment_Status_April','Repayment_Status_May','Repayment_Status_June']]  #categorical columns
X_smS= X_sm[['Credit_Amount','Age_Years','June_Bill_Amount','Previous_Payment_June']] #numerical columns


# In[70]:


# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range  =(-1,1)).fit(X_smS)
Xs_smS = scaler.transform(X_smS)


# In[71]:


#X_sm = Xs_smS.tolist()

df1 = pd.DataFrame(columns = dataCAT.columns.tolist() , data = X_smU)
df2 = pd.DataFrame(columns = dataNUM.columns.tolist() , data = Xs_smS)

DF = pd.concat([df1, df2], axis = 1)
X_sm = DF.values

X_scaled = DF.iloc[:,0:-1] # scaled version
y_sm


# In[72]:


#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_sm, y_sm, test_size = 0.3, random_state = 0)
# for scaled data use X_scaled instead of X_sm 


t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')


# In[73]:


# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train_sm, y_train_sm.ravel()) # for probability

pred_proba = svm_p.predict_proba(X_val_sm)[::,1]
fpr, tpr, _ = roc_curve(y_val_sm,  pred_proba)
auc = roc_auc_score(y_val_sm, pred_proba)
plt.plot(fpr,tpr,label="Smote, auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[74]:


# --------------------------------homework------------------------------------------------------

# repeat the modelling for df2 with scaling

# columns should be 14

#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_scaled, y_sm, test_size = 0.3, random_state = 0)

t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')


# In[75]:


#-------------------------------------------------------------------------------------------
# repeat the modelling for df2 with scaling for all numerical --- create df2_ without clubbing
dataCAT_original = data_original[catCols]
data_allNumeric = pd.concat([dataCAT_original, dataNUM_], axis = 1)
data_allNumeric['Default_Payment'] = data['Default_Payment']

X = data_allNumeric.iloc[:,0:-1]
y = data_allNumeric['Default_Payment']

X = df_.iloc[:,0:-1]   # why has this been used heer  ?????????????????????????
y = df_['Default_Payment']        # why has this been used heer  ?????????????????????????

sm = SMOTE(random_state = 2) 
X_sm, y_sm = sm.fit_sample(X, y.ravel()) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range  =(-1,1)).fit(X_sm)
Xscaled = scaler.transform(X_sm)


#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(Xscaled, y_sm, test_size = 0.3, random_state = 0)

t1 = time.time()
# Naked Modelling

svm = SVC() #SVC(kernel='linear') 

clf = svm.fit(X_train_sm, y_train_sm.ravel()) 
score1 = svm.score(X_train_sm, y_train_sm)
score2 = svm.score(X_val_sm, y_val_sm)
pred = svm.predict(X_val_sm) 
  
# print classification report 
print('With SMOTE:'.upper())
print('train accuracy: ', score1)
print('test accuracy: ', score2)
print('F1 score:\n', classification_report(y_val_sm, pred)) 
print('*'*80)
#print('\n')


# In[80]:


#### GRID SEARCH WITH SMOTE for SVM---------------------------------------------------
t3 = time.time()

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import metrics

parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
svm_gs = GridSearchCV(svm,parameters)
svm_gs.fit(X_train_sm,y_train_sm)
preds = svm_gs.predict(X_val_sm)
score3 = svm_gs.score(X_train_sm, y_train_sm)
score4 = svm_gs.score(X_val_sm, y_val_sm)
predG = svm_gs.predict(X_val_sm)

print('GRID SEARCH WITH SMOTE:')
print('Using best parameters:',svm_gs.best_params_)
print('Train accuracy:', np.round(score3,3))
print('Test accuracy:', np.round(score4,3))
print('F1 score:\n', classification_report(y_val_sm, predG)) 

# ROC Curve
svm_p = SVC(probability = True) # for probability
clf_p = svm_p.fit(X_train_sm, y_train_sm.ravel()) 
pred_proba_gs = svm_p.predict_proba(X_val_sm)[::,1]
fpr_gs, tpr_gs, _ = roc_curve(y_val_sm,  pred_proba_gs)
auc_gs = roc_auc_score(y_val_sm, pred_proba_gs)
plt.plot(fpr_gs,tpr_gs,label="Gs-Smote, auc="+str(np.round(auc_gs,3)))
plt.legend(loc=4)
plt.tight_layout()
print('*'*80)


# In[81]:


#### GRID SEARCH WITH SMOTE with CROSS VALIDATION---------------------------------
t5 = time.time()

def SVM_gridSearch(X,y,nfolds):
    #create a dictionary of all values we want to test
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma' : [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
    # decision tree model
    svm = SVC()
    #use gridsearch to val all values
    svm_gscv = GridSearchCV(svm, parameters, cv=nfolds)
    #fit model to data
    svm_gscv.fit(X, y)
    #find score
    accuracy = svm_gscv.score(X, y)
    
    return svm_gscv.best_params_, accuracy, svm_gscv

print('GRID SEARCH WITH SMOTE & CROSS VALIDATION -- DT:')
best_param, score5, svm_gscv = SVM_gridSearch(X_train_sm,y_train_sm, 4)
score6 = svm_gscv.score(X_val_sm, y_val_sm)
predGC = svm_gscv.predict(X_val_sm)
print('Using best parameters:',best_param)
print('Train accuracy:', np.round(score5,3))
print('Test accuracy:', np.round(score6,3))
print('F1 score:\n', classification_report(y_val_sm, predGC))

## ROC curve
svm_pp = SVC(probability = True) # for probability
clf_p = svm_pp.fit(X_train_sm, y_train_sm.ravel()) 
pred_proba_gsp = svm_pp.predict_proba(X_val_sm)[::,1]

fpr_gscv, tpr_gscv, _ = roc_curve(y_val_sm,  pred_proba_gsp)
auc_gscv = metrics.roc_auc_score(y_val_sm, pred_proba_gsp)
plt.plot(fpr_gscv,tpr_gscv,label="Gs-Smote-CV, auc="+str(np.round(auc_gscv,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[ ]:




