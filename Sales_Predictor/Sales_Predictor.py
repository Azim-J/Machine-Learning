import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("train.csv")

#Features to use: Outlet_Type, Outlet_Location_Type, Item_MRP, Target: Item_Outlet_Sales
df_Xy = df.loc[:, ["Outlet_Type", "Item_Identifier", "Outlet_Identifier", "Outlet_Establishment_Year", "Item_MRP", "Outlet_Location_Type", "Item_Visibility", "Item_Outlet_Sales"]]
df_Xy["Item_Category"] = df_Xy["Item_Identifier"].str[:2]
df_X = df_Xy.loc[:, ["Outlet_Type", "Item_Category", "Outlet_Identifier", "Outlet_Establishment_Year","Item_MRP", "Outlet_Location_Type", "Item_Visibility"]]
df_X["Outlet_Establishment_Year"] = df_X["Outlet_Establishment_Year"].apply(lambda x: 2013-x)
df_y = df_Xy.loc[:, "Item_Outlet_Sales"]

df_X = pd.get_dummies(df_X, columns=["Outlet_Type", "Outlet_Location_Type", "Outlet_Identifier", "Item_Category"])

train_X, temp_X, train_y, temp_y = train_test_split(df_X, df_y, random_state=13, test_size=0.2, shuffle=True)
val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, random_state=13, test_size=0.5, shuffle=True)

model = RandomForestRegressor(n_estimators=400, random_state=13, max_depth=100)
model.fit(train_X, train_y)

prediction = model.predict(val_X)
mae = mean_absolute_error(y_true=val_y, y_pred=prediction)
print("\n", "MAE:", mae)

k_folds = KFold(n_splits=5)
cv_score = cross_val_score(model, train_X, train_y, cv=k_folds, scoring="neg_mean_absolute_error")
print("Negative MAE out of 5 times:", cv_score, )
print("Mean MAE:", cv_score.mean(), "\n")
print("RSME:", root_mean_squared_error(val_y, prediction))
r2 = r2_score(val_y, prediction)
print(r2)
n = val_X.shape[0]
p = val_X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("Adjusted R²:", adj_r2)
coef3 = pd.Series(model.feature_importances_, df_X.columns).sort_values(ascending=False)
coef3.plot(kind='bar', title='Feature Importances')