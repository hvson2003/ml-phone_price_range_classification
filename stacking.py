import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Đọc dữ liệu
phone_data = pd.read_csv('phoneprice_dataset.csv')

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
dt_Train, dt_Test = train_test_split(phone_data, test_size=0.3, shuffle=True)

X_train = dt_Train.iloc[:, :20]
y_train = dt_Train.iloc[:, 20]
X_test = dt_Test.iloc[:, :20]
y_test = dt_Test.iloc[:, 20]

# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Mô hình Perceptron
perceptron_model = Perceptron(penalty='l1', alpha=0.001, max_iter=3000, class_weight='balanced')
perceptron_model.fit(X_train_scaled, y_train.values.ravel())
perceptron_preds = perceptron_model.predict(X_test_scaled)
print("Perceptron Accuracy:", accuracy_score(y_test, perceptron_preds))

# Mô hình Logistic Regression
logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=10)
logistic_model.fit(X_train_scaled, y_train.values.ravel())
logistic_preds = logistic_model.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_preds))

# Mô hình Neural Network (MLPClassifier)
nn_model = MLPClassifier(solver='lbfgs', max_iter=3000, alpha=0.001)
nn_model.fit(X_train_scaled, y_train)
nn_preds = nn_model.predict(X_test_scaled)
print("Neural Network Accuracy:", accuracy_score(y_test, nn_preds))

# Stacking
base_model_preds = np.column_stack((perceptron_preds, logistic_preds, nn_preds))

# Mô hình meta (Logistic Regression)
meta_model = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0)
meta_model.fit(base_model_preds, y_test)

# Dự đoán và in kết quả
stacking_preds = meta_model.predict(base_model_preds)

accuracy = accuracy_score(y_test, stacking_preds)
precision = precision_score(y_test, stacking_preds, average='macro')
recall = recall_score(y_test, stacking_preds, average='macro')
f1 = f1_score(y_test, stacking_preds, average='macro')

print()
print('Stacking(',f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}', ')')

