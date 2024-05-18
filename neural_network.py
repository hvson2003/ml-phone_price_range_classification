import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.neural_network import MLPClassifier
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

# Tiêu chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Xây dựng và huấn luyện mô hình
nn_model = MLPClassifier(solver='lbfgs', max_iter=3000, alpha=0.001)
nn_model.fit(X_train_scaled, y_train)
y_pred = nn_model.predict(X_test_scaled)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# In các độ đo đánh giá vào bảng điều khiển
print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-score: {f1}\n')
