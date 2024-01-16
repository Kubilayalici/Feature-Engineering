import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from datetime import date
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x : "%.3f" % x)
pd.set_option("display.width", 500)

df = pd.read_csv("Telco-Customer-Churn.csv")

dff = df.copy()

def eda(dataframe, head=5):
    print(15*"#" + " " + "Head" + " " + 15*"#")
    print(dataframe.head(head))
    print(15*"#" + " " + "Shape" + " " + 15*"#")
    print(dataframe.shape)
    print(15*"#" + " " + "Info" + " " + 15*"#")
    print(dataframe.info())
    print(15*"#" + " " + "Null" + " " + 15*"#")
    print(dataframe.isnull().sum())
    print(15*"#" + " " + "Tail" + " " + 15*"#")
    print(dataframe.tail(head))
    print(15*"#" + " " + "Describe" + " " + 15*"#")
    print(dataframe.describe().T)
    
eda(dff)


dff["TotalCharges"] = pd.to_numeric(dff["TotalCharges"], errors= "coerce")

dff["Churn"] = dff["Churn"].apply(lambda x : 1 if x == "Yes" else 0)

# Grab categorical and numeric columns

def grab_col_names(dataframe, cat_th = 10, car_th = 20):
    cat_col = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_col = [col for col in num_col if col not in num_but_cat]

    print(f"Observation: {dataframe.shape[1]}")
    print(f"cat_col: {len(cat_col)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_col: {len(num_col)}")

    return cat_col, cat_but_car,num_col

cat_col, cat_but_car, num_col = grab_col_names(dff)




###############################
# Categoric Variables Analysis
###############################

# Categorical Variables Analysis depent on Target Variable

def target_summary_with_cat(dataframe, target,categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts()/ len(dataframe)}), end="\n\n\n")

for col in cat_col:
    target_summary_with_cat(dff, "Churn", col)



def cat_summary(dataframe, col_name, plot = False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio":100*dataframe[col_name].value_counts()/len(dataframe)}))
    print("#########################")
    if plot:
        sns.countplot(x= dataframe[col_name], data= dataframe)
        plt.show()


for col in cat_col:
    print(cat_summary(dff,col))



###############################
# Numeric Variables Analysis 
###############################

# Numeric Variables Analysis depent on Target Variable

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end = "\n\n\n")

for col in num_col:
    target_summary_with_num(dff, "Churn", col)



def num_summary(dataframe, numerical_col, plot = False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins = 20)
        plt.xlabel(numerical_col)
        plt.ylabel(numerical_col)
        plt.show()

for col in num_col:
    print(num_summary(dff, col, plot=True))



#################
# Correlation Matrix
###################
    
dff[num_col].corr()

f,ax = plt.subplots(figsize = [18,13])
sns.heatmap(dff[num_col].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation_Matrix", fontsize = 20)
plt.show()




##############################
# Missing Value Analysis
##############################

dff.isnull().sum()

def missing_value_table(dataframe, na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum()>0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio= (dataframe[na_columns].isnull().sum()/dataframe.shape[0]*100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1, keys=["n_miss","ratio"])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_value_table(dff, na_name=True)

dff["TotalCharges"].fillna(dff["TotalCharges"].median(), inplace= True)

df.isnull().sum()

############################
# Base Model 
############################

data = dff.copy()
cat_col = [col for col in cat_col if col not in ["Churn"]]
cat_col

def one_hot_encoder(dataframe, categorical_col, drop_first = False):
    dataframe = pd.get_dummies(dataframe, columns= categorical_col, drop_first= drop_first)
    return dataframe

data = one_hot_encoder(data, cat_col, drop_first=True)
data.head()


y = data["Churn"]
X = data.drop(["Churn","customerID"], axis=1)
y.head()
X.head()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 17)

catbosst_model = CatBoostClassifier(verbose=False, random_state= 12345).fit(X_train, y_train)
y_pred = catbosst_model.predict(X_test)


print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 4)}")
print(f"Precision:{round(precision_score(y_pred, y_test), 4)}")
print(f"F1:{round(f1_score(y_pred, y_test), 4)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 4)}")


###################################
# Outliers Analysis
###################################

def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3 = 0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 -quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>up_limit)|(dataframe[col_name]<low_limit)].any(axis = None):
        return True
    else:
        return False

def replace_with_threshold(dataframe, variable, q1=0.05, q3 = 0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=q1, q3=q3)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable] = up_limit

for col in num_col:
    print(col, check_outlier(dff, col))
    if check_outlier(dff,col):
        replace_with_threshold(dff, col)


##################################
# Feature Extraction
##################################
        

dff.loc[(dff["tenure"]>=0)&(dff["tenure"]<=12), "NEW_TENURE_YEAR"] = "0-1 Year"
dff.loc[(dff["tenure"]>12)&(dff["tenure"]<=24), "NEW_TENURE_YEAR"] = "1-2 Year"
dff.loc[(dff["tenure"]>24)&(dff["tenure"]<=36), "NEW_TENURE_YEAR"] = "2-3 Year"
dff.loc[(dff["tenure"]>36)&(dff["tenure"]<=48), "NEW_TENURE_YEAR"] = "3-4 Year"
dff.loc[(dff["tenure"]>48)&(dff["tenure"]<=60), "NEW_TENURE_YEAR"] = "4-5 Year"
dff.loc[(dff["tenure"]>60)&(dff["tenure"]<=72), "NEW_TENURE_YEAR"] = "5-6 Year"

dff["NEW_Engaged"] = dff["Contract"].apply(lambda x : 1 if x in ["One year", "Two year"] else 0)


dff["New_noProt"] = dff.apply(lambda x : 1 if (x["OnlineBackup"]!="Yes") or (x["DeviceProtection"]!= "Yes") or (x["TechSupport"]!= "Yes") else 0, axis=1)


dff["New_Young_Not_Engaged"]= dff.apply(lambda x : 1 if (x["NEW_Engaged"]==0)and (x["SeniorCitizen"]==0) else 0, axis=1)


dff["NEW_TotalServices"] = (dff[["PhoneService", "InternetService", "OnlineSecurity",
                                 "OnlineBackup", "DeviceProtection", "TechSupport",
                                 "StreamingTV", "StreamingMovies"]]=="Yes").sum(axis=1)


dff["NEW_FLAG_ANY_STREAMING"]= dff.apply(lambda x : 1 if (x["StreamingTV"]=="Yes") or (x["StreamingMovies"]=="Yes") else 0, axis=1)

dff["NEW_FLAG_AutoPayment"] = dff["PaymentMethod"].apply(lambda x : 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

dff["NEW_AVG_Charges"] = dff["TotalCharges"] / (dff["tenure"]+1)

dff["NEW_Inrease"] = dff["NEW_AVG_Charges"] / dff["MonthlyCharges"]

dff["NEW_AVG_Service_fee"] = dff["MonthlyCharges"] / (dff["NEW_TotalServices"]+1)

dff.shape

dff.columns

cat_col, num_col, cat_but_car = grab_col_names(dff)


#Label Encoding


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in dff.columns if dff[col].dtypes == "O" and dff[col].nunique()==2]

for col in binary_cols:
    dff = label_encoder(dff, col)


cat_cols = [col for col in cat_col if col not in binary_cols and col not in ["Churn","NEW_TotalServices"]]

def one_hot_encoder(dataframe, categorical_col, drop_first= False):
    dataframe= pd.get_dummies(dataframe, columns= categorical_col, drop_first=drop_first)
    return dataframe

dff = one_hot_encoder(dff, cat_cols, drop_first=True)

dff.head()


##################
# New Model
#################

y = dff["Churn"]
X = dff.drop(["Churn","customerID"], axis=1)


X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

catbost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)

y_pred = catbost_model.predict(X_test)

print(f"Accuracy:{round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall:{round(recall_score(y_pred, y_test), 2)}")
print(f"Precision:{round(precision_score(y_pred, y_test), 2)}")
print(f"F1:{round(f1_score(y_pred, y_test), 2)}")
print(f"Auc:{round(roc_auc_score(y_pred, y_test), 2)}")



def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    f1_df= pd.DataFrame(data)
    
    
    f1_df.sort_values(by=["feature_importance"], ascending= False, inplace= True)

    plt.figure(figsize=(15,10))

    sns.barplot(x=f1_df["feature_importance"], y = f1_df["feature_names"])
    plt.title(model_type + "FEATURE_IMPORTANCE")
    plt.xlabel(["FEATURE_IMPORTANCE"])
    plt.ylabel("FEATURE NAMES")
    plt.show()

plot_feature_importance(catbost_model.get_feature_importance(), X.columns, "CATBOOST")