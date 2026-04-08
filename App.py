import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("real_student.csv")

# cleaning
df["Marks"] = df["Marks"].fillna(df["Marks"].mean())
df["Sleep"] = df["Sleep"].fillna(df["Sleep"].mean())

# input/output
X = df[["Hours", "Sleep"]]
y = df["Marks"]

# model
model = LinearRegression()
model.fit(X, y)

try:
    hours = float(input("Enter study hours: "))
    sleep = float(input("Enter sleep hours: "))

    if hours < 0 or sleep < 0:
        print("Invalid input! Values must be positive.")
    else:
        result = model.predict([[hours, sleep]])
        print(f"\n📊 Predicted Marks: {result[0]:.2f}")

        # graph
        plt.scatter(df["Hours"], df["Marks"], label="Actual Data")
        plt.scatter(hours, result[0], color="red", label="Your Prediction")

        plt.xlabel("Hours")
        plt.ylabel("Marks")
        plt.title("Study Hours vs Marks")
        plt.legend()

        plt.show()

except:
    print("Invalid input! Please enter numbers only.")
