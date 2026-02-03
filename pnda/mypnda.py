import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
print("====== Prediction for House Property ======")

# Create dataset
data = {
    "size": [800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [2, 2, 3, 3, 4, 4],
    "Price": [25, 30, 40, 55, 65, 75]
}

df = pd.DataFrame(data)
print(df)

# Features & target
X = df[["size", "bedrooms"]]   # ✅ FIXED
y = df["Price"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model

y_pred=model.predict(X_test)
mae=mean_absolute_error(y_test,y_pred)

joblib.dump(model,"house_price_model.pkl")
print("Model error mae:",mae)

# User input
size = int(input("Enter house size (sqft): "))
rooms = int(input("Enter number of bedrooms: "))

# Prediction
predict = model.predict([[size, rooms]])
print(predict)
print("Predicted price is:", predict[0])


