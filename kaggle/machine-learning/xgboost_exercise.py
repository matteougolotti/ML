import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

data = pd.read_csv('train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

my_model = XGBRegressor()
my_model.fit(train_X, train_y, verbose=False)

predictions = my_model.predict(test_X)

print("Simple model Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Model with 1000 iterations Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))


my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)

print("Model with custom learning rate Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))