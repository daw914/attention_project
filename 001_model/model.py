import pickle
import pandas as pd 
import matplotlib as mat
import matplotlib.pyplot as plt    
import numpy as np
import seaborn as sns
import shap


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")
import shap
from sklearn.model_selection import train_test_split

df = pd.read_csv('ABKK_attributes.csv')

def menu_order(x):#split the menu
    x=str(x)
    x=list(x)
    return x

def l1_order(x):#define the position of lottery 1, e.g. if res=5, lottery 1 is positioned the first after the outside option. position is -100000000 if lottery 1 not in the menu
    res=-100000000
    if x[0]=="1":
        res=5
    elif len(x)>=2 and x[1]=="1":
        res=4
    elif len(x)>=3 and x[2]=="1":
        res=3
    elif len(x)>=4 and x[3]=="1":
        res=2
    elif len(x)>=5 and x[4]=="1":
        res=1
    return res

def l2_order(x):
    res=-100000000
    if x[0]=="2":
        res=5
    elif len(x)>=2 and x[1]=="2":
        res=4
    elif len(x)>=3 and x[2]=="2":
        res=3
    elif len(x)>=4 and x[3]=="2":
        res=2
    elif len(x)>=5 and x[4]=="2":
        res=1
    return res    

def l3_order(x):
    res=-100000000
    if x[0]=="3":
        res=5
    elif len(x)>=2 and x[1]=="3":
        res=4
    elif len(x)>=3 and x[2]=="3":
        res=3
    elif len(x)>=4 and x[3]=="3":
        res=2
    elif len(x)>=5 and x[4]=="3":
        res=1
    return res

def l4_order(x):
    res=-100000000
    if x[0]=="4":
        res=5
    elif len(x)>=2 and x[1]=="4":
        res=4
    elif len(x)>=3 and x[2]=="4":
        res=3
    elif len(x)>=4 and x[3]=="4":
        res=2
    elif len(x)>=5 and x[4]=="4":
        res=1
    return res

def l5_order(x):
    res=-100000000
    if x[0]=="5":
        res=5
    elif len(x)>=2 and x[1]=="5":
        res=4
    elif len(x)>=3 and x[2]=="5":
        res=3
    elif len(x)>=4 and x[3]=="5":
        res=2
    elif len(x)>=5 and x[4]=="5":
        res=1
    return res

df['menu_ordered']=df.order.map(menu_order)
df['order_l1']=df.menu_ordered.map(l1_order)
df['order_l2']=df.menu_ordered.map(l2_order)
df['order_l3']=df.menu_ordered.map(l3_order)
df['order_l4']=df.menu_ordered.map(l4_order)
df['order_l5']=df.menu_ordered.map(l5_order)
df=df.rename(columns = {'alt_asked':'position_1','altasked':'position_2','v24':'position_3','v25':'position_4','v26':'position_5'})
#split the data into train (80%) and test data (20%)
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
keys = set(df['ethnicity'])
res = {}
for key in keys:
    res[key] = [0]*len(df['ethnicity'])    
for i, item in enumerate(df['ethnicity']):
    if item in keys:
        res[item][i] = 1
df_eth=pd.DataFrame(res)
df_eth.index=df.index
df_eth=df_eth.rename(columns = {1:'white',2:'hispanic',3:'black',4:'asian',5:'Native_Hawaiian',6:'other'})
df = pd.concat([df, df_eth], axis=1)



def bin0(X):
    if X==0:
        return 1
    else:
        return 0
    
df['choice']=df.choice.map(bin0)

features = list(df.columns)
target = 'choice'
features.remove(target)
features.remove('latitud')
features.remove('longitud')
    
X= df[features]
y = df[target]

#bulid a data transformer to make data ready for machine learning
class DataTransformer(object):
    
    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['gender']=X.gender
        df['age']=X.age
        df['education']=X.education
        df['income']=X.income
        df['marital_status']=X.marital_status.map(self.married) #1 if ever married
        df['employment']=X.employment.map(self.employed)
        df['cost']=X.cost
        df['survey_duation']=X.duration
        df['firstclick']=X.firstclick
        df['stopping_time']=X.lastclick
        df['clickcount']=X.clickcount
        df['hispanic']=X.hispanic
        df['black']=X.black
        df['asian']=X.asian
        df['Native_Hawaiian']=X.Native_Hawaiian
        df['other_race']=X.other
        #df['position_1'] = X.position_1.map(self.none_position)
        #df['position_2'] = X.position_2.map(self.none_position)
        #df['position_3'] = X.position_3.map(self.none_position)
        #df['position_4'] = X.position_4.map(self.none_position)
        #df['position_5'] = X.position_5.map(self.none_position)
        df['var_pos1']=X.position_1.map(self.var)
        df['var_pos2']=X.position_2.map(self.var)
        df['var_pos3']=X.position_3.map(self.var)
        df['var_pos4']=X.position_4.map(self.var)
        df['var_pos5']=X.position_5.map(self.var)
        df['expect_pos1']=X.position_1.map(self.expect)
        df['expect_pos2']=X.position_2.map(self.expect)
        df['expect_pos3']=X.position_3.map(self.expect)
        df['expect_pos4']=X.position_4.map(self.expect)
        df['expect_pos5']=X.position_5.map(self.expect)
        df['order_l1']=X.order_l1
        df['order_l2']=X.order_l2
        df['order_l3']=X.order_l3
        df['order_l4']=X.order_l4
        df['order_l5']=X.order_l5
        df['steps1']=X.position_1.map(self.steps)
        df['steps2']=X.position_2.map(self.steps)
        df['steps3']=X.position_3.map(self.steps)
        df['steps4']=X.position_4.map(self.steps)
        df['steps5']=X.position_5.map(self.steps)
        self.mean = df.mean()
        
    def transform(self, X, y=None):
        df = pd.DataFrame()
        df['gender']=X.gender
        df['age']=X.age
        df['education']=X.education
        #df['ethnicity']=X.ethnicity
        df['income']=X.income
        df['marital_status']=X.marital_status.map(self.married) #1 if ever married
        df['employment']=X.employment.map(self.employed)
        df['cost']=X.cost
        df['survey_duation']=X.duration
        df['firstclick']=X.firstclick
        df['stopping_time']=X.lastclick
        df['clickcount']=X.clickcount
        df['hispanic']=X.hispanic
        df['black']=X.black
        df['asian']=X.asian
        df['Native_Hawaiian']=X.Native_Hawaiian
        df['other_race']=X.other
        #df['position_1'] = X.position_1.map(self.none_position)
        #df['position_2'] = X.position_2.map(self.none_position)
        #df['position_3'] = X.position_3.map(self.none_position)
        #df['position_4'] = X.position_4.map(self.none_position)
        #df['position_5'] = X.position_5.map(self.none_position)
        df['var_pos1']=X.position_1.map(self.var)
        df['var_pos2']=X.position_2.map(self.var)
        df['var_pos3']=X.position_3.map(self.var)
        df['var_pos4']=X.position_4.map(self.var)
        df['var_pos5']=X.position_5.map(self.var)
        df['expect_pos1']=X.position_1.map(self.expect)
        df['expect_pos2']=X.position_2.map(self.expect)
        df['expect_pos3']=X.position_3.map(self.expect)
        df['expect_pos4']=X.position_4.map(self.expect)
        df['expect_pos5']=X.position_5.map(self.expect)
        df['order_l1']=X.order_l1
        df['order_l2']=X.order_l2
        df['order_l3']=X.order_l3
        df['order_l4']=X.order_l4
        df['order_l5']=X.order_l5
        df['steps1']=X.position_1.map(self.steps)
        df['steps2']=X.position_2.map(self.steps)
        df['steps3']=X.position_3.map(self.steps)
        df['steps4']=X.position_4.map(self.steps)
        df['steps5']=X.position_5.map(self.steps)
        #df=df.drop(['position_1', 'position_2','position_3','position_4','position_5'], axis=1)
        #self.mean = df.mean()
        return df.fillna(self.mean)
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def none_position(self, x):
        if x=='-':
            x=-10000
        else:
            x=float(x)
        return x
    
    def var(self, x):
        if x!='-':
            x=float(x)
        if x==1:
            v=625
        elif x==2:
            v=100
        elif x==3:
            v=368.75
        elif x==4:
            v=511.73
        elif x==5:
            v=251.11
        else:
            v=0
        return v 
    
    def expect(self, x):
        if x!='-':
            x=float(x)
        if x==1:
            v=25
        elif x==2:
            v=20
        elif x==3:
            v=22.5
        elif x==4:
            v=24.125
        elif x==5:
            v=21.625
        else:
            v=0
        return v  
    
    def steps(self,x):
        if x=='-':
            x=-10000
        else:
            x=float(x)
        if x==1:
            b=2
        elif x==2:
            b=2
        elif x==3 or x==4:
            b=4
        elif x==5:
            b=5
        else: b=10000
        return b
    
    def married(self, X):
        if X==1:
            return 0
        else:
            return 1
    def employed(self, X):
        if X==1:
            return 1
        else:
            return 0

params = {'objective': 'binary:logistic',
 'eval_metric': 'logloss',
 'max_depth': 5,
 'gamma': 0.3,
 'lambda': 5,
 'learning_rate': 0.05,
 'max_depth': 3,
 'min_child_weight': 5,
 'n_estimators': 100}
steps=[('tf', DataTransformer()),
      ('xgb', XGBClassifier(**params))]
model=Pipeline(steps)


model=model.fit(X,y)

with open('model.pkl','wb') as f:
    pickle.dump(model,f)