import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data from CSV file
data = pd.read_csv("data.csv")

# Perform one-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=["Tool_Material"])

# Split the data into features (X) and target (y)
X = data_encoded.drop("Lifetime", axis=1)
y = data_encoded["Lifetime"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression
regression = LinearRegression()

# Train the regression using the training data
regression.fit(X_train, y_train)

# Make predictions on the test set
y_pred = regression.predict(X_test)

# Evaluate the regression performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Plot the actual vs. predicted lifetime values
plt.scatter(y_test, y_pred, color='blue')
plt.plot(y_test, y_test, color='red', linewidth=2)  # Plot the regression line
plt.xlabel("Actual Lifetime")
plt.ylabel("Predicted Lifetime")
plt.title("Actual vs. Predicted Lifetime")
plt.grid(True)
plt.show()


# Generate the feature importance plot
coefficients = pd.Series(regression.coef_, index=X_train.columns)
coefficients.plot(kind='bar')
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Feature Importance Plot")
plt.grid(True)
plt.show()

# Generate the distribution plot of lifetime
plt.hist(y, bins=15, edgecolor='black')
plt.xlabel("Lifetime")
plt.ylabel("Frequency")
plt.title("Distribution of Lifetime")
plt.grid(True)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the training data from CSV file (assuming you've already loaded the training data)
train_data = pd.read_csv("data.csv")

# Load the test data from CSV file
test_data = pd.read_csv("test.csv")

# Perform one-hot encoding for categorical variables in both training and test data
train_data_encoded = pd.get_dummies(train_data, columns=["Tool_Material"])
test_data_encoded = pd.get_dummies(test_data, columns=["Tool_Material"])

# Split the training data into features (X_train) and target (y_train)
X_train = train_data_encoded.drop("Lifetime", axis=1)
y_train = train_data_encoded["Lifetime"]

# Initialize the linear regression
regression = LinearRegression()

# Train the regression using the training data
regression.fit(X_train, y_train)

# Extract the features for the test data
X_test = test_data_encoded

# Predict the lifetime for the test data
predicted_lifetime = regression.predict(X_test)

# Print the predicted lifetime for each instance in the test data
for i, lifetime in enumerate(predicted_lifetime):
    print(f"Instance {i+1}: Predicted Lifetime = {lifetime}")




# Pickling Model for Deployment
import pickle
pickle.dump(regression,open('regmodel.pkl','wb'))


pickled_model=pickle.load(open('regmodel.pkl','rb'))
pickled_model.predict(X_test)





