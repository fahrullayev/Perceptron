import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox


root = tk.Tk()
root.title('Perceptron Dasturi')
root.geometry('400x200')


def yuklash_va_ishlash():
    try:
        
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

       
        data = pd.read_csv(filepath)
        X = data.drop('Target', axis=1)
        y = data['Target']
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
       
        perceptron = Perceptron(max_iter=1000, eta0=0.1)
        perceptron.fit(X_train, y_train)
        
       
        y_pred = perceptron.predict(X_test)
        aniqlik = accuracy_score(y_test, y_pred)
        messagebox.showinfo("Aniqlik", f"Aniqlik: {aniqlik:.4f}")
        
       
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 6))
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
        
    except Exception as e:
        messagebox.showerror("Xatolik", str(e))


yuklash_button = tk.Button(root, text="Ma'lumotlarni yuklash", command=yuklash_va_ishlash)
yuklash_button.pack(pady=20)


root.mainloop()
