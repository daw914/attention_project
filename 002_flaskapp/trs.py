import pandas as pd

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