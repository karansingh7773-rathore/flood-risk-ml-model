import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
data = {
    "age": [18, 22, 25, 30, 34, 40, 45, 50, 55, 60, 65, 28, 33, 48, 52, 70],
    "monthly_charge": [299, 150, 200, 500, 450, 300, 350, 250, 400, 600, 550, 180, 320, 270, 390, 520],
    "churn": [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]  # 1 = Happy, 0 = Not Happy
}
df = pd.DataFrame(data)
x = df[['age','monthly_charge']]
y = df['churn']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)
model_SVC = SVC(kernel='linear')
model_SVC.fit(x_train, y_train)
userage = float(input("Enter age of User: "))
monthly_charge = int(input("Enter the monthly charge: "))
user_output = np.array([[userage, monthly_charge]])
predict = model_SVC.predict(user_output)
if predict[0] == 1:
    print("User is Happy with price")
else:
    print("User is not Happy")

