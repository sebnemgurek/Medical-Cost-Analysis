# Aim
"""Predict insurance costs"""

# Variables
"""
age: age of primary beneficiary
sex: insurance contractor gender, female, male
bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
children: Number of children covered by health insurance / Number of dependents
smoker: Smoking
region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
charges: Individual medical costs billed by health insurance
"""

# Import Required Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import warnings
warnings.simplefilter(action="ignore")

# Loading the dataset

df_= pd.read_csv("insurance/insurance.csv")
df=df_.copy()
df.head()


# Discoverer Data Analysis

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

check_df(df)

"""
Veri setinde 7 değişken ve 1337 adet gözlem bulunmaktadır.
Bağımlı değişken:charges
Bağımsız değişkenler:age, sex, bmi, children, smoker,region
Veri setinde eksik gözlem bulunmamaktadır.
"""
#############################################################

# Identifying numeric and categorical variables

def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}',f'{cat_cols}')
    print(f'num_cols: {len(num_cols)}',f'{num_cols}')
    print(f'cat_but_car: {len(cat_but_car)}',f'{cat_but_car}')
    print(f'num_but_cat: {len(num_but_cat)}',f'{num_but_cat}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

"""
Observations: 1338
Variables: 7
cat_cols: 4 ['sex', 'smoker', 'region', 'children']
num_cols: 3 ['age', 'bmi', 'charges']
cat_but_car: 0 []
num_but_cat: 1 ['children']
"""

############################################################

# Frequencies of Classes of Categorical Variables

for i in cat_cols:
    print(df[i].value_counts())
    print("###########################")

"""
#Sex
male      676
female    662
###########################
#Smoker
no     1064
yes     274
###########################
#Region
southeast    364
southwest    325
northwest    325
northeast    324
###########################
#Children
0    574
1    324
2    240
3    157
4     25
5     18
"""

##########################################################

# Digitizing categorical variables

# Label encoder
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df=label_encoder(df, col)

df.head()

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df=one_hot_encoder(df, ohe_cols)

#################################################################################################

#Q-1
# Bmi(Vücut Kitle İndeksi)’nin dağılımı

plt.hist(df["bmi"]);
#sns.histplot(data = df["bmi"])


###################################################################
#Q-2
# “smoker” ile “charges” arasındaki ilişki

df_new=df[["smoker","charges"]]
df_new.corr(method ='pearson')

"""
           smoker   charges
smoker   1.000000  0.787251
charges  0.787251  1.000000

"""

#Korelasyon grafiği
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_new.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdBu",vmin=-1)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# Sigara kullanan ve kullanmayan kişilerin ortalama masrafları

df.groupby("smoker")["charges"].mean()
sns.barplot(x="smoker", y="charges", data=df)

"""
0     8434.268298
1    32050.231832
"""

###################################################################   olmadı sorrr
#Q-3
# “smoker” (Sigara tüketen) ile “region”(Bölge) arasındaki ilişkiyi inceleyiniz.

ax = sns.countplot(x="region", hue="smoker", data=df)

df_sr=df[["smoker","region"]]
df_sr.head()
df_sr.corr(method ='pearson')

###################################################################
#Q-4
# “bmi” ile “sex”(Cinsiyet) arasındaki ilişkiyi inceleyiniz.

df_sb=df[["sex","bmi"]]
df_sb.corr(method ='pearson')

"""
          sex       bmi
sex  1.000000  0.046371
bmi  0.046371  1.000000
"""

#Korelasyon grafiği
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_sb.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdBu",vmin=-1)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


###################################################################
#Q-5
# En çok “children”’a sahip “region”’ı bulunuz.

df.groupby("region").agg({"children":sum})

"""Southeast=382"""

###################################################################
#Q-6
# “Age” ile “bmi” arasındaki ilişkiyi inceleyiniz.

df_ab=df[["age","bmi"]]
df_ab.corr(method ='pearson')

"""
          age       bmi
age  1.000000  0.109272
bmi  0.109272  1.000000
"""

#Korelasyon grafiği

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_ab.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdBu",vmin=-1)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


###################################################################
#Q-7
# “bmi” ile “children” arasındaki ilişkiyi inceleyiniz.

df_cb=df[["children","bmi"]]
df_cb.corr(method ='pearson')

"""
          children       bmi
children  1.000000  0.012759
bmi       0.012759  1.000000
"""

#Korelasyon grafiği
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_ab.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdBu",vmin=-1)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

###################################################################
#Q-8
# “bmi” değişkeninde outlier var mıdır? İnceleyiniz.
sns.boxplot(x=df["bmi"])

"""Evet aykırı değer mevcut"""

##################################################################
#Q-9
# “bmi” ile “charges” arasındaki ilişkiyi inceleyiniz.

df_chb=df[["bmi","charges"]]
df_chb.corr(method ='pearson')

"""
              bmi   charges
bmi      1.000000  0.198341
charges  0.198341  1.000000
"""

#Korelasyon grafiği
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_chb.corr(), annot=True, fmt=".2f", ax=ax, cmap="RdBu",vmin=-1)
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#############################################################################
#Q-10
# “region”, “smoker” ve “bmi” arasındaki ilişkiyi bar plot kullanarak inceleyiniz

sns.barplot(x="region", y="bmi", hue="smoker", data=df)

###################################################################################

# Analysis of Categorical Variables

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

############################################################################

# Analysis of Numerical Variables

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

#####################################################

# Outlier observation analysis

#1-Setting a threshold
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


#2-Check for outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df,col))



########################################################

# 4-Data Preprocessing

# Selection of dependent and independent variable:
X = df.drop(['charges'], axis=1)
y = df[["charges"]]

# Splitting the dataset into train-test:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


####################################################################################################

# 5-Model Selection and Examining the performance of selected models using cross validation:

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("LightGBM", LGBMRegressor())
          ]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 6068.3355 (LR) 
RMSE: 6068.3148 (Ridge) 
RMSE: 6067.8588 (Lasso) 
RMSE: 4810.2618 (RF) 
RMSE: 12684.3279 (SVR) 
RMSE: 4530.9505 (GBM) 
RMSE: 4724.545 (LightGBM) 
"""

# Best performing model
"""
RMSE: 4530.9505 (GBM) 
"""
############################################################

# 6-Optimizing parameters with Grid Search

rf_params = {"max_depth": [7,8,9 ,None],
             "max_features": [4,5,6, "auto"],
             "min_samples_split": [7,8,9],
             "n_estimators": [990,1000,1010]}


lightgbm_params = {"learning_rate": [0.09, 0.1, 0.15],
                   "n_estimators": [490, 500, 510],
                   "colsample_bytree":[0.4, 0.5, 0.6]}


regressors = [("RF", RandomForestRegressor(warm_start=True), rf_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
             ]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

"""
########## RF ##########
RMSE: 4861.134 (RF) 
RMSE (After): 4581.3358 (RF) 
RF best params: {'max_depth': 7, 'max_features': 'auto', 'min_samples_split': 9, 'n_estimators': 1010}

########## LightGBM ##########
RMSE: 4724.545 (LightGBM) 
RMSE (After): 5096.9425 (LightGBM) 
LightGBM best params: {'colsample_bytree': 0.5, 'learning_rate': 0.09, 'n_estimators': 490}
"""

# 7-Model Evaluation
# Regresyon modeli değerlendirme metriklerini kullanarak optimize edilmiş olan modelin değerlendirmesini yapınız.
# (Ör. Mean Squared Error, Mean Absolute Error vb.)

rf_model = RandomForestRegressor().fit(X_train, y_train)
y_pred = rf_model.predict(X_train)

# Train RKARE
rf_model.score(X_train, y_train)
"""
Eğitim veri setindeki bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesi  %97 dir. 
"""

# Train RMSE
y_pred = rf_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
"""
Eğitim Hatası:1891.41
"""

# Test RKARE
rf_model.score(X_test, y_test)
"""
Modelin bağımsız değişkenlerin bağımlı değişkeni açıklama yüzdesi  %84 dir. 
"""

# Test RMSE
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
"""
Test hatası:4251.28
"""

