from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd 
# ====================================================================

"""
The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree. 
It generally has much better predictive accuracy than a single decision tree and it works well with default parameters
"""

# save filepath to variable for easier access
melbourne_file_path = './01_basic_data_exploration/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# Extract a column to predict
y = melbourne_data.Price

# Extract features from dataframe (its like a subset)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]


# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print("Mean absolute error: ")
print(mean_absolute_error(val_y, melb_preds))