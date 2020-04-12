import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

for files in os.listdir(os.getcwd()):
	print(os.path.join(os.getcwd(),files))




'''
auto = TPOTClassifier(config_dict=None, crossover_rate=0.1, cv=5,
               disable_update_check=False, early_stop=None, generations=100,
               max_eval_time_mins=5, max_time_mins=None, memory=None,
               mutation_rate=0.9, n_jobs=1, offspring_size=None,
               periodic_checkpoint_folder=None, population_size=100,
               random_state=None, scoring=None, subsample=1.0, template=None,
               use_dask=False, verbosity=0, warm_start=False)
'''



# Import Data
train = pd.read_csv(r'C:\Users\sumasark\Downloads\python-UpGrad\AutoML\TPOT\train.csv')
test = pd.read_csv(r'C:\Users\sumasark\Downloads\python-UpGrad\AutoML\TPOT\test.csv')


# Impute Missing Values
train.loc[train.Age.isnull(),'Age'] = train.Age.median()
test.loc[test.Age.isnull(),'Age'] = test.Age.median()

test.loc[test.Fare.isnull(),'Fare'] = test.Fare.median()

train.loc[train.Cabin.isnull(),'Cabin'] = 'M'
test.loc[test.Cabin.isnull(),'Cabin'] = 'M'

train['Cabin'] = train.Cabin.astype(str).str[0]
test['Cabin'] = test.Cabin.astype(str).str[0]

train.Cabin = train.Cabin[train.Cabin!='T']

train.loc[train.Embarked.isnull(),'Embarked'] = 'S'

train['Sex'] = train.Sex.map({'male':1,'female':0})
test['Sex'] = test.Sex.map({'male':1,'female':0})



'''
print(train.isnull().sum()/len(train.index))
print(test.isnull().sum()/len(train.index))
'''

# Feature Engineering
dumm_train = pd.get_dummies(train[['Cabin','Embarked']],drop_first=True)
dumm_test = pd.get_dummies(test[['Cabin','Embarked']],drop_first=True)

train = pd.concat([train,dumm_train],axis='columns')
test = pd.concat([test,dumm_test],axis='columns')

train.Cabin = train.Cabin[train.Cabin!='T']



# Dropping Unuses Columns
train = train.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis='columns')
test = test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis='columns')


print('train_shape={},test_shape={}'.format(train.shape,test.shape))



# Doing and train validation split
y = train.pop('Survived')
X = train

train_X,validation_X,train_y,validation_y = train_test_split(X,y,test_size=0.3,random_state=42)


# Fitting a TPOT classification model
# Change max_time_mins for the amount of time you want to train 
tpot = TPOTClassifier(verbosity = 2,max_time_mins=1)
tpot.fit(train_X,train_y)
print(tpot.score(validation_X,validation_y))


print(tpot.fitted_pipeline_)

tpot.export('tpot_titanic.py')
