import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score
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
grid_search_nn = MLPClassifier(solver='lbfgs', max_iter=3000, alpha=0.001)
grid_search_nn.fit(X_train_scaled, y_train)
best_nn_model = grid_search_nn
nn_preds = best_nn_model.predict(X_test_scaled)
print("Neural Network Accuracy:", accuracy_score(y_test, nn_preds))

# Mô hình học kết hợp (Stacking Classifier)
base_models = [('perceptron', perceptron_model), ('logistic', logistic_model), ('neural_network', best_nn_model)]
final_estimator = LogisticRegression(penalty='l2', solver='lbfgs', C=10)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train_scaled, y_train.values.ravel())
stacking_preds = stacking_model.predict(X_test_scaled)
print("Stacking Classifier Accuracy:", accuracy_score(y_test, stacking_preds))
