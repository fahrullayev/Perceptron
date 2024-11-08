import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns


data = pd.read_csv('perceptron_dataset.csv')
X = data.drop('Target', axis=1)
y = data['Target']
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
perceptron = Perceptron(max_iter=1000, eta0=0.1)
perceptron.fit(X_train, y_train)


y_pred = perceptron.predict(X_test)
print(y_pred)
aniqlik = accuracy_score(y_test, y_pred)
print("Aniqlik:", aniqlik)


conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.ylabel("Haqiqiy")
plt.xlabel('Taxminiy')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Haqiqiy qiymatlar', marker='o')
plt.plot(y_pred, label='Bashorat qilingan qiymatlar', marker='x')
plt.xlabel('Namunalar')
plt.ylabel('Qiymatlar')
plt.title('Haqiqiy va Bashorat qilingan qiymatlar')
plt.legend()
plt.grid(True)
plt.show()
