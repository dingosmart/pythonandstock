# https://towardsdatascience.com/a-beginners-guide-to-linear-regression-in-python-with-scikit-learn-83a8f7ae2b4f

# https: // www.analyticsvidhya.com / blog / 2018 / 03 / introduction - k - neighbours - algorithm - clustering /

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

pd.set_option('display.max_rows', 2500)  # 最大行数
pd.set_option('display.max_columns', 2500)  # 最大列数
pd.set_option('display.width', 4000)  # 页面宽度

dataset = pd.read_csv('D:/test/Weather.csv', low_memory=False)
ds = dataset.describe()
print(ds)

dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

plt.figure(figsize=(15, 10))
plt.tight_layout()
seabornInstance.distplot(dataset['MaxTemp'])
plt.show()

# Our next step is to divide the data into “attributes” and “labels”.
# Attributes are the independent variables while labels are dependent variables whose values are to be predicted.
# In our dataset, we only have two columns. We want to predict the MaxTemp depending upon the MinTemp recorded.
# Therefore our attribute set will consist of the “MinTemp” column which is stored in the X variable,
# and the label will be the “MaxTemp” column which is stored in y variable.

X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)

# Next, we split 80% of the data to the training set while 20% of the data to test set using below code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)  # training the algorithm

# the linear regression model basically finds the best value for the intercept and slope, which results in a line
# that best fits the data. To see the value of the intercept and slope calculated by the linear regression algorithm
# for our dataset, execute the following code

# To retrieve the intercept:
print(regressor.intercept_)
# For retrieving the slope:
print(regressor.coef_)

# This means that for every one unit of change in Min temperature, the change in the Max temperature is about 0.92%.

# Now that we have trained our algorithm, it’s time to make some predictions.
# To do so, we will use our test data and see how accurately our algorithm predicts the percentage score.
# To make predictions on the test data, execute the following script

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

df1 = df.head(25)
df1.plot(kind='bar', figsize=(16, 10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# The final step is to evaluate the performance of the algorithm.
# This step is particularly important to compare how well different algorithms perform on a particular dataset.
# For regression algorithms, three evaluation metrics are commonly used:

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
