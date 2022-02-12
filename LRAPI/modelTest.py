from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle


data = pd.read_csv("ageinsurance.csv")
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

x_train, x_test, y_train, y_test = train_test_split(data[["age"]], data.has_insurance, test_size=0.25, random_state=0)

model = LogisticRegression()
model.fit(x_train,y_train)

prediction = model.predict(x_test)
accuracy = model.score(x_test,y_test)
print(accuracy)

pickle.dump(model, open("modelTest.pkl", "wb"))