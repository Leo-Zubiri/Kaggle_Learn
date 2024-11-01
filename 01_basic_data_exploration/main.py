import pandas as pd 

# save filepath to variable for easier access
melbourne_file_path = './01_basic_data_exploration/melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# See the columns of the dataset
print(melbourne_data.columns)

# print a summary of the data in Melbourne data
melbourne_data.describe()
print(melbourne_data.describe())

# Drop rows with not available values  axis=0 rows  1 columns
print(melbourne_data.count)
melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data.count)

# Extract a column to predict
y = melbourne_data.Price

# Extract features from dataframe (its like a subset)
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())