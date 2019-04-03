import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

iowa_file_path = 'train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

iowa_model = DecisionTreeRegressor(random_state = 1)

# Fit iowa_model with the training data.
iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

# Calculate Mean Absolute Error
val_mae = mean_absolute_error(val_y, val_predictions)
