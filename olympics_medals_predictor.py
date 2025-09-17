import pandas as pd

teams=pd.read_csv("teams.csv")

teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]
print(teams)

teams.select_dtypes(include=["number"]).corr()["medals"]

import seaborn as sns
sns.lmplot(x="athletes",y="medals",data=teams,fit_reg=True,ci=None)
sns.lmplot(x="age",y="medals",data=teams,fit_reg=True,ci=None)

teams.plot.hist(y="medals")
#search missing data
teams[teams.isnull().any(axis=1)]

#drop rows has missing values
teams =teams.dropna()
print(teams)

#spilt our data up take date of 2012,2016 to(test data set)
#rest to train ddata
train=teams[teams["year"]<2012].copy()
test=teams[teams["year"]>=2012].copy()

train.shape #(1609,7)
test.shape #(405,7)

#------------------train model-----------MULTI REGRESSION--------------
from sklearn.linear_model import LinearRegression

reg=LinearRegression()
predictors=["athletes","prev_medals"]
target="medals"
reg.fit(train[predictors],train["medals"])
LinearRegression()
predictions= reg.predict(test[predictors])
#print(predictions)

test["predictions"]=predictions
print(test)


#improve data so its not negative or not rounded
test.loc[test["predictions"]<0,"predictions"]=0
test["predictions"]=test["predictions"].round()
print(test)


#MEAN ABSOLUTE ERROR
from sklearn.metrics import mean_absolute_error
error=mean_absolute_error(test["medals"],test["predictions"])
print(error)



#check if this is good error or not
teams.describe()["medals"]
#since is less than std -> this is good error



#TEST
test[test["team"]=="USA"]
print(test)

errors = (test["medals"] - predictions).abs()
print(errors)
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio =  error_by_team / medals_by_team
print(error_ratio)


