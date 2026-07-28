import pandas as pd
from math import log as ln
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

scaler = StandardScaler()

df = pd.read_csv("loan_data.csv")
df_both = df.loc[:, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History", "Loan_Status", "Loan_Amount_Term", "Education", "Self_Employed"]]
df_both = df_both.dropna()
df_X = df_both.loc[:, ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Credit_History", "Loan_Amount_Term", "Education", "Self_Employed"]]
df_X = pd.get_dummies(df_X, columns=["Education", "Self_Employed"])
df_X["LoanAmount"] = df_X["LoanAmount"].apply(lambda x: ln(x))
df_y = df_both.loc[:, "Loan_Status"]
df_y = df_y.map({"Y": 1, "N": 0})

train_X, temp_X, train_y, temp_y = train_test_split(df_X, df_y, shuffle = True, random_state=13, test_size=0.2)
val_X, test_X, val_y, test_y = train_test_split(temp_X, temp_y, shuffle = True, random_state=13, test_size=0.5)

train_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]] = scaler.fit_transform(train_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]])
val_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]] = scaler.transform(val_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]])
test_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]] = scaler.transform(test_X[["LoanAmount", "ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term"]])

model = LogisticRegression(max_iter=2000, random_state=13, C=0.1)
model.fit(train_X, train_y)
print(accuracy_score(y_true=val_y, y_pred=model.predict(val_X)))

print(confusion_matrix(val_y, model.predict(val_X)))