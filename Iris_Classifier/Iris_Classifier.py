import pandas as pd
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("iris.data", header=None)
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df_X = df.iloc[:, :4]
df_y = df[:][["species"]]
df_y = pd.get_dummies(df_y, columns=["species"])
model = RandomForestClassifier(random_state=13, n_estimators=5)

train_X, temp_X, train_y, temp_y = train_test_split(df_X, df_y, test_size=0.2, shuffle=True, random_state=13)
val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, test_size=0.5, shuffle=True, random_state=13)
model.fit(train_X, train_y)
'''
print("Validation error: " + str(mean_absolute_error(y_true=val_y, y_pred=model.predict(val_X))))
print("Validation accuracy: " + str(accuracy_score(y_true=val_y, y_pred=model.predict(val_X))))
print("Test error: " + str(mean_absolute_error(y_true=test_y, y_pred=model.predict(test_X))))
print("Test accuracy: " + str(accuracy_score(y_true=test_y, y_pred=model.predict(test_X))))
print("Accuracy on 5 different splits: " + str(cross_val_score(model, df_X, df_y, cv=5)))
print("Mean accuracy on 5 different splits: " + str(cross_val_score(model, df_X, df_y, cv=5).mean()))
'''

user_data = pd.read_csv("iris_balanced_150_samples.csv")
user_data.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
user_X = user_data.iloc[:, :4]
pred_list = model.predict(user_X)
readable_list = []
for (i, pred) in enumerate(pred_list):
    if pred[0] == True:
        print(i+1, "Iris-setosa")
        readable_list.append("Iris-setosa")
    elif pred[1] == True:
        print(i+1, "Iris-versicolor")
        readable_list.append("Iris-versicolor")
    elif pred[2] == True:
        print(i+1, "Iris-virginica")
        readable_list.append("Iris-virginica")

setosa_acc = 50
for pred in readable_list[:50]:
    if pred != "Iris-setosa":
        setosa_acc -= 1

versicolor_acc = 50
for pred in readable_list[50:100]:
    if pred != "Iris-versicolor":
        versicolor_acc -= 1

virginica_acc = 50
for pred in readable_list[100:]:
    if pred != "Iris-virginica":
        virginica_acc -= 1

print("Setosa Accuracy: " + str(setosa_acc/50))
print("Versicolor Accuracy: " + str(versicolor_acc/50))
print("Virginica Accuracy: " + str(virginica_acc/50))

print("Total Accuracy: " + str((setosa_acc+versicolor_acc+virginica_acc)/150))