import pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("HousingData.csv")
df_X = df.drop(columns = "MEDV")
df_y = df["MEDV"]

train_X, temp_X, train_y, temp_y = train_test_split(df_X, df_y, test_size = 0.2, random_state=13)
val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, random_state=13)

train_X = train_X.fillna(train_X.median())
val_X = val_X.fillna(val_X.median())
test_X = test_X.fillna(test_X.median)

model = RandomForestRegressor(n_estimators=750, random_state=13)
model.fit(train_X, train_y), 
prediction = model.predict(val_X)
print(mean_absolute_error(val_y, prediction))
print(root_mean_squared_error(val_y, prediction))
r2 = r2_score(val_y, prediction)
print(r2)
n = df_X.shape[0]
p = df_X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(adj_r2)
coef1 = pd.Series(model.feature_importances_, df_X.columns).sort_values()
coef1.plot(kind='bar', title='Model Coefficients')
plt.show()