
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(42)
n_samples = 800
ages = np.random.randint(16, 67, n_samples)
cart_status = np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7])
time_spent = np.random.randint(1, 30, n_samples)
items_in_cart = np.random.randint(0, 6, n_samples)
discount_applied = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
past_purchases = np.random.randint(0, 20, n_samples)

purchases = np.array([
    1 if (cart == 1 and (time > 10 or items > 1 or discount == 1) and np.random.rand() > 0.3) else 0
    for cart, time, items, discount in zip(cart_status, time_spent, items_in_cart, discount_applied)
])

df = pd.DataFrame({
    "Age": ages,
    "Added_to_Cart": cart_status,
    "Time_Spent": time_spent,
    "Items_in_Cart": items_in_cart,
    "Discount_Applied": discount_applied,
    "Past_Purchases": past_purchases,
    "Purchased": purchases
})

X = df[["Age", "Added_to_Cart", "Time_Spent", "Items_in_Cart", "Discount_Applied", "Past_Purchases"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\n===== Customer Purchase Prediction =====")
age = int(input("Enter Customer Age: "))
cart = int(input("Added to Cart? (1 = Yes, 0 = No): "))
time_spent = int(input("Time spent on website (minutes): "))
items_in_cart = int(input("Number of items in cart: "))
discount = int(input("Discount applied? (1 = Yes, 0 = No): "))
past = int(input("Number of past purchases: "))

features = [[age, cart, time_spent, items_in_cart, discount, past]]
prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0][1]

print("\n===== Prediction Result =====")
print(f"Predicted Class: {'Purchased' if prediction == 1 else 'Not Purchased'}")
print(f"Probability of Purchase: {probability:.2%}")
print("========================================")
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_pred, y_test))
