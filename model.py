#!/usr/bin/env python
# coding: utf-8

# # Business Problem :
# Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.
# 
# With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home
# 
# 
# # problem statement :
# Predicting the home prices based on customer requirement in residential homes in Ames, Iowa

# # Descriprtion of data fields

# In[2]:




#Data fields
#Here's a brief version of what you'll find in the data description file.

#SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
#MSSubClass: The building class
#MSZoning: The general zoning classification
#LotFrontage: Linear feet of street connected to property
#LotArea: Lot size in square feet
#Street: Type of road access
#Alley: Type of alley access
#LotShape: General shape of property
#LandContour: Flatness of the property
#Utilities: Type of utilities available
#LotConfig: Lot configuration
#LandSlope: Slope of property
#Neighborhood: Physical locations within Ames city limits
#Condition1: Proximity to main road or railroad
#Condition2: Proximity to main road or railroad (if a second is present)
#BldgType: Type of dwelling
#HouseStyle: Style of dwelling
#OverallQual: Overall material and finish quality
#OverallCond: Overall condition rating
#YearBuilt: Original construction date
#YearRemodAdd: Remodel date
#RoofStyle: Type of roof
#RoofMatl: Roof material
#Exterior1st: Exterior covering on house
#Exterior2nd: Exterior covering on house (if more than one material)
#MasVnrType: Masonry veneer type
#MasVnrArea: Masonry veneer area in square feet
#ExterQual: Exterior material quality
#ExterCond: Present condition of the material on the exterior
#Foundation: Type of foundation
#BsmtQual: Height of the basement
#BsmtCond: General condition of the basement
#BsmtExposure: Walkout or garden level basement walls
#BsmtFinType1: Quality of basement finished area
#BsmtFinSF1: Type 1 finished square feet
#BsmtFinType2: Quality of second finished area (if present)
#BsmtFinSF2: Type 2 finished square feet
#BsmtUnfSF: Unfinished square feet of basement area
#TotalBsmtSF: Total square feet of basement area
#Heating: Type of heating
#HeatingQC: Heating quality and condition
#CentralAir: Central air conditioning
#Electrical: Electrical system
#1stFlrSF: First Floor square feet
#2ndFlrSF: Second floor square feet
#LowQualFinSF: Low quality finished square feet (all floors)
#GrLivArea: Above grade (ground) living area square feet
#BsmtFullBath: Basement full bathrooms
#BsmtHalfBath: Basement half bathrooms
#FullBath: Full bathrooms above grade
#HalfBath: Half baths above grade
#Bedroom: Number of bedrooms above basement level
#Kitchen: Number of kitchens
#KitchenQual: Kitchen quality
#TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
#Functional: Home functionality rating
#Fireplaces: Number of fireplaces
#FireplaceQu: Fireplace quality
#GarageType: Garage location
#GarageYrBlt: Year garage was built
#GarageFinish: Interior finish of the garage
#GarageCars: Size of garage in car capacity
#GarageArea: Size of garage in square feet
#GarageQual: Garage quality
#GarageCond: Garage condition
#PavedDrive: Paved driveway
#WoodDeckSF: Wood deck area in square feet
#OpenPorchSF: Open porch area in square feet
#EnclosedPorch: Enclosed porch area in square feet
#3SsnPorch: Three season porch area in square feet
#ScreenPorch: Screen porch area in square feet
#PoolArea: Pool area in square feet
#PoolQC: Pool quality
#Fence: Fence quality
#MiscFeature: Miscellaneous feature not covered in other categories
#MiscVal: $Value of miscellaneous feature
#MoSold: Month Sold
#YrSold: Year Sold
#SaleType: Type of sale
#SaleCondition: Condition of sale


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import sklearn


# In[4]:


# data_dictionary=pd.read_csv(rD:\My python Projects\house-prices-advanced-regression-techniques\data_description.txt")


# In[5]:


#df = pd.read_csv(r"D:\My python Projects\house-prices-advanced-regression-techniques\train.csv")

df=pd.read_csv("train.csv")
# In[6]:


df.head()


# In[7]:


df.iloc[0:59,60:].isna().sum()


# In[8]:


sns.heatmap(df[df.describe().columns])


# # Memory optimization

# In[9]:



initial_memoryusage=df.memory_usage().sum()/(1024*1024)
def memory_management(df):
    for i in range(0,len(df.columns)):
        if "int" in str(df.dtypes.values[i]):
            min_value=df[df.columns[i]].min()
            max_value=df[df.columns[i]].max()
            if min_value>np.iinfo(np.int8).min and max_value<np.iinfo(np.int8).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.int8)
            elif min_value>np.iinfo(np.int16).min and max_value<np.iinfo(np.int16).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.int16)
            elif min_value>np.iinfo(np.int32).min and max_value<np.iinfo(np.int32).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.int32)
            elif min_value>np.iinfo(np.int64).min and max_value<np.iinfo(np.int64).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.int64)
        elif "float" in str(df.dtypes.values[i]):
            #print(df.columns[i],df.dtypes.values[i])
            min_value=df[df.columns[i]].min()
            max_value=df[df.columns[i]].max()
            if min_value>np.finfo(np.float16).min and max_value<np.finfo(np.float16).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.float16)
            elif min_value>np.finfo(np.float32).min and max_value<np.finfo(np.float32).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.float32)
            elif min_value>np.finfo(np.float64).min and max_value<np.finfo(np.float64).max:
                df[df.columns[i]]=df[df.columns[i]].astype(np.float64)
        else:
            pass
        return df


# In[10]:


new_memory_allocated=memory_management(df)


# In[11]:


new_memory_allocated=df.memory_usage().sum()/(1024*1024)
print("Memory before optimizaiton",initial_memoryusage)
print("Memory after optimizaiton",new_memory_allocated)
print("optimization percentagge",((initial_memoryusage-new_memory_allocated)/(new_memory_allocated)*100))


# # Null value treatment

# In[12]:


def null_value_treatment(df,threshold):
    ### calculating null value percentage
    print("null values before treatmenet")
    print(" ")
    print(df.isna().sum())
    null_value_percentage=(df.isna().sum()/df.shape[0])*100
    ### segregating null value percentage based on threshold
    drop_columns=null_value_percentage[null_value_percentage>threshold].index
    fill_columns=null_value_percentage[null_value_percentage<=threshold].index
    ### drop columns
    df.drop(drop_columns,axis=1,inplace=True) ### Fill columns #print(df[fill_columns].describe().columns) #print(df[fill_columns].describe(include='object').columns) #print(df[fill_columns].describe(include='category').columns)
    ### fill null values for numerical
    for i in df[fill_columns].describe().columns:
        df[i].fillna(df[i].median(),inplace=True) ### fill null values for object
    for i in df[fill_columns].describe(include='object').columns:
        df[i].fillna(df[i].value_counts().index[0],inplace=True)
   ### fill null values for categorical variables
   ### for i in house_data[fill_columns].describe(include='category').columns:
        # house_data[i].fillna(house_data[i].value_counts().index[0],inplace=True)
    print("null values after treatmenet")
    print(" ")
    print(df.isna().sum())
    return df

base_house_data=null_value_treatment(df,30)


# # Outliers

# In[55]:


def outlier(table_name):
    numerical_columns=table_name[table_name.describe().columns]
    cont_numerical_columns=[]
    discrete_numerical_columns=[]
    for i in numerical_columns:
        if numerical_columns[i].nunique()>8:
            cont_numerical_columns.append(i)
        else:
            discrete_numerical_columns.append(i)
    for i in cont_numerical_columns:
        fqrt=np.quantile(table_name[i].values,0.25)
        tqrt=np.quantile(table_name[i].values,0.75)
        #print(fqrt,tqrt)
        iqr=tqrt-fqrt
        utv=tqrt+1.5*iqr
        ltv=fqrt-1.5*iqr
        data=[]
    for i in table_name[i].values:
        if i>utv and i<ltv:
            print('outlier found')
            data.append(table_name[i].median())
        else:
            data.append(i)



outlier(base_house_data)



from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for i in base_house_data.describe(include='object').columns:
    lb.fit(base_house_data[i])
    data_points=lb.transform(base_house_data[i])
    base_house_data[i]=data_points


# # Splitting the Dataset for Model

base_house_data=base_house_data[['LotArea','HouseStyle','SaleCondition','SalePrice']]
#base_house_data.loc['LotArea','HouseStyle','yrsold','SaleCondition','SalePrice']
y=base_house_data['SalePrice']
x=base_house_data.drop('SalePrice',axis=1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=121,test_size=0.2) #### 80 Percent and 20 Percent of the data
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)


# # Model 1 KNN Regression



from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV



params = {'n_neighbors':[2,3,4,5,6,7,8,9]}

knn =KNeighborsRegressor()

model = GridSearchCV(knn, params, cv=5)
model.fit(x_train,y_train)
model.best_params_

K=9
model =KNeighborsRegressor(n_neighbors = K)
model.fit(x_train,y_train) 
knn_predicted_values=model.predict(x_test)

pickle.dump(model,open('model.pkl','wb'))
