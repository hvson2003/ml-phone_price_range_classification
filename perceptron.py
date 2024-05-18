import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
phone_data = pd.read_csv('phoneprice_dataset.csv')

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm thử
dt_Train, dt_Test = train_test_split(phone_data, test_size=0.3, shuffle=True)

X_train = dt_Train.iloc[:, :20]
y_train = dt_Train.iloc[:, 20]
X_test = dt_Test.iloc[:, :20]
y_test = dt_Test.iloc[:, 20]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

perceptron_model = Perceptron(penalty='l1', alpha=0.001, max_iter=3000, class_weight='balanced')
perceptron_model.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron_model.predict(X_test_scaled)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred_perceptron)
precision = precision_score(y_test, y_pred_perceptron, average='macro')
recall = recall_score(y_test, y_pred_perceptron, average='macro')
f1 = f1_score(y_test, y_pred_perceptron, average='macro')

print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n')

