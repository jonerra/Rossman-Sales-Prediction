import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")
df = df[['DayOfWeek', 'Customers', 'Open', 'Promo', 'Sales']]

print(df.head())

xs = np.array(df['Customers'], dtype=np.float64)
ys = np.array(df['Sales'], dtype=np.float64)

slope, intercept, r_value, p_value, std_err = stats.linregress(xs,ys)
r_squared = r_value**2
print(r_squared)

regression_line = [(slope*x) + intercept for x in xs]

X = np.array(df[['DayOfWeek', 'Customers', 'Open', 'Promo']], dtype=np.float64)
X = preprocessing.scale(X)
X = X.reshape(-1, 4)
y = np.array(df['Sales'], dtype=np.float64)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))

with open('sales.txt', 'w') as f:
    for x in xs:
        y = (slope*x) + intercept
        f.write(str(y) + '\n')


# predict_x = 500
# predict_y = (slope*predict_x) + intercept
# print(predict_y)

# plt.scatter(xs, ys)
# plt.plot(xs, regression_line)
# plt.show()

