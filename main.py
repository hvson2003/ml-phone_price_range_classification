import pandas as pd
import tkinter as tk
from tkinter import ttk

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

phone_data = pd.read_csv('phoneprice_dataset.csv')

dt_Train, dt_Test = train_test_split(phone_data, test_size=0.3, shuffle=True)

X_train = dt_Train.iloc[:, :20]
y_train = dt_Train.iloc[:, 20]
X_test = dt_Test.iloc[:, :20]
y_test = dt_Test.iloc[:, 20]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=10)
logistic_model.fit(X_train_scaled, y_train)
y_pred = logistic_model.predict(X_test_scaled)

root = tk.Tk()
root.title("Phone Price Prediction")

# Tạo các combobox và entry cho dữ liệu đầu vào
labels = ["Battery Power(mAh)", "Bluetooth", "Clock Speed (GHz)", "Dual Sim", "Front Camera (MP)", "4G",
          "Internal Memory (GB)", "Mobile Depth", "Mobile Weight (g)", "Number Cores", "PC", "Pixel Height",
          "Pixel Width", "Ram (GB)", "Screen Height (cm)", "Screen Weight (cm)", "Talk Time (hour)", "3G",
          "Touch Screen", "Wifi"]

combobox_list = []
default_values = [842, "No", 2.2, "No", 1, "No", 7, 0.6, 188, 2, 2, 20, 756, 2549, 9, 7, 19, "No", "No", "Yes",
                  "No"]

for i, (label, default_value) in enumerate(zip(labels, default_values)):
    ttk.Label(root, text=label).grid(row=i, column=0, padx=10, pady=5)

    if label in ["Bluetooth", "Dual Sim", "4G", "3G", "Touch Screen", "Wifi"]:
        # Sử dụng combobox cho các thuộc tính có lựa chọn "Yes" hoặc "No"
        combobox = ttk.Combobox(root, values=["Yes", "No"])
        combobox.grid(row=i, column=1, padx=10, pady=5)
        combobox.set(default_value)
        combobox_list.append(combobox)
    else:
        # Sử dụng entry cho các thuộc tính khác
        entry = ttk.Entry(root)
        entry.grid(row=i, column=1, padx=10, pady=5)
        entry.insert(0, str(default_value))
        combobox_list.append(entry)


def predict():
    # Lấy dữ liệu đầu vào từ GUI và chuyển đổi từ Yes/No sang 1/0 nếu cần thiết
    input_data = []
    for i, combobox in enumerate(combobox_list):
        value = combobox.get() if isinstance(combobox, ttk.Combobox) else combobox.get()
        if value == "Yes":
            input_data.append(1)
        elif value == "No":
            input_data.append(0)
        else:
            input_data.append(float(value))
    input_data_scaled = scaler.transform([input_data])  # Sử dụng scaler đã tồn tại

    # Dự đoán bằng mô hình đã huấn luyện
    prediction = logistic_model.predict(input_data_scaled)

    # Hiển thị kết quả dự đoán
    result_label.config(text=f'Predicted Price Range: {prediction[0]}')

    # Tính toán và hiển thị các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    accuracy_label.config(text=f'Accuracy: {accuracy}')
    precision_label.config(text=f'Precision: {precision}')
    recall_label.config(text=f'Recall: {recall}')
    f1_label.config(text=f'F1-score: {f1}')


predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=len(labels), column=0, columnspan=2, pady=10)

# Hiển thị kết quả
result_label = ttk.Label(root, text="Predicted Price Range:")
result_label.grid(row=len(labels) + 1, column=0, columnspan=2, pady=10)

# Hiển thị độ đo
accuracy_label = ttk.Label(root, text="Accuracy:")
accuracy_label.grid(row=2, column=3, columnspan=2, pady=5)

precision_label = ttk.Label(root, text="Precision:")
precision_label.grid(row=3, column=3, columnspan=2, pady=5)

recall_label = ttk.Label(root, text="Recall:")
recall_label.grid(row=+4, column=3, columnspan=2, pady=5)

f1_label = ttk.Label(root, text="F1-score:")
f1_label.grid(row=5, column=3, columnspan=2, pady=5)

# Chạy GUI
root.mainloop()
