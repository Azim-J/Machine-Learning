import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv", sep=";")
#features: "studytime", "school", "failures", "paid", "G1", "G2", "higher", "absences", "internet"
print(df[["studytime", "school", "failures", "paid", "G1", "G2", "higher", "absences", "internet"]].describe(include="all"))
#for feature in ["studytime", "school", "failures", "paid", "G1", "G2", "higher", "absences", "internet"]:
#    sns.histplot(data=df, x=feature)
#    plt.show()