import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/student_scores.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

X = df[["Hours"]]

y = df["Scores"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

model = LinearRegression()

print(model)

model.fit(X_train, y_train)

print("Model trained successfully!")

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nMAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

new_student = [[8]]

predicted_score = model.predict(new_student)

print("\nPredicted score for 8 study hours:")
print(predicted_score)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(results)

plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.title("Hours vs Scores")

plt.show()

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)