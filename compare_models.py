import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
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

# Mô hình Perceptron
perceptron_model = Perceptron(penalty='l1', alpha=0.001, max_iter=3000, class_weight='balanced')
perceptron_model.fit(X_train_scaled, y_train.values.ravel())
perceptron_preds = perceptron_model.predict(X_test_scaled)

# Mô hình Logistic Regression
logistic_model = LogisticRegression(penalty='l2', solver='lbfgs', C=10)
logistic_model.fit(X_train_scaled, y_train.values.ravel())
logistic_preds = logistic_model.predict(X_test_scaled)

# Mô hình Neural Network (MLPClassifier)
grid_search_nn = MLPClassifier(solver='lbfgs', max_iter=3000, alpha=0.001)
grid_search_nn.fit(X_train_scaled, y_train)
best_nn_model = grid_search_nn
nn_preds = best_nn_model.predict(X_test_scaled)

# Mô hình học kết hợp (Stacking Classifier)
base_models = [('perceptron', perceptron_model), ('logistic', logistic_model), ('neural_network', best_nn_model)]
final_estimator = LogisticRegression(penalty='l2', solver='lbfgs', C=10)
stacking_model = StackingClassifier(estimators=base_models, final_estimator=final_estimator)
stacking_model.fit(X_train_scaled, y_train.values.ravel())
stacking_preds = stacking_model.predict(X_test_scaled)

models = ["Perceptron", "Logistic Regression", "Neural Network", "Stacking Classifier"]
accuracies = [accuracy_score(y_test, perceptron_preds),
              accuracy_score(y_test, logistic_preds),
              accuracy_score(y_test, nn_preds),
              accuracy_score(y_test, stacking_preds)]
precisions = [precision_score(y_test, perceptron_preds, average='weighted'),
              precision_score(y_test, logistic_preds, average='weighted'),
              precision_score(y_test, nn_preds, average='weighted'),
              precision_score(y_test, stacking_preds, average='weighted')]
recalls = [recall_score(y_test, perceptron_preds, average='weighted'),
           recall_score(y_test, logistic_preds, average='weighted'),
           recall_score(y_test, nn_preds, average='weighted'),
           recall_score(y_test, stacking_preds, average='weighted')]
f1_scores = [f1_score(y_test, perceptron_preds, average='weighted'),
             f1_score(y_test, logistic_preds, average='weighted'),
             f1_score(y_test, nn_preds, average='weighted'),
             f1_score(y_test, stacking_preds, average='weighted')]

# Create a DataFrame to store and display the metrics
metrics_df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies,
    "Precision": precisions,
    "Recall": recalls,
    "F1-Score": f1_scores
})

print(metrics_df)
