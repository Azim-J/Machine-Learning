import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("student-mat.csv", sep=";")

X = df[["studytime", "school", "failures", "paid", "G1", "G2", "higher", "absences", "internet"]]
X = pd.get_dummies(X, columns=["higher", "school", "paid", "internet"])
y = df["G3"]
print(X)

model = RandomForestRegressor(random_state=13, )
train_X, check_X, train_y, check_y = train_test_split(X, y, shuffle = True, test_size=0.3, random_state=13)
val_X, test_X, val_y, test_y = train_test_split(check_X, check_y, test_size=0.5, random_state=13)
model.fit(train_X, train_y)
print(mean_absolute_error(y_true=val_y, y_pred=model.predict(val_X)))