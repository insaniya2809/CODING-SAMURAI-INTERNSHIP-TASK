
# Project 4 - Logistic Regression on Titanic


import matplotlib

matplotlib.use('TkAgg')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, ConfusionMatrixDisplay)
import joblib

import os
os.makedirs("model", exist_ok=True)

sns.set(style="whitegrid")

df = sns.load_dataset("titanic")
print("✅ Dataset loaded. Shape:", df.shape)
print(df.head())


print("\n✅ Info:")
print(df.info())

print("\n✅ Missing values:")
print(df.isnull().sum())

features = ['sex', 'age', 'pclass', 'fare', 'sibsp', 'parch', 'embarked']
target = 'survived'

df = df[features + [target]]
print("\n✅ Data with selected features shape:", df.shape)
print(df.head())

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n✅ Train/test split done. Train:", X_train.shape[0], "Test:", X_test.shape[0])


num_features = ['age', 'fare', 'sibsp', 'parch']
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),   
    ('scaler', StandardScaler())                     
])


cat_features = ['sex', 'pclass', 'embarked']

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])


clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])


clf_pipeline.fit(X_train, y_train)
print("\n✅ Baseline logistic regression trained.")


y_pred = clf_pipeline.predict(X_test)
y_proba = clf_pipeline.predict_proba(X_test)[:, 1]  

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rocauc = roc_auc_score(y_test, y_proba)

print("\n✅ Baseline Evaluation on Test Set:")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC AUC:   {rocauc:.3f}")

print("\nClassification report:\n", classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf_pipeline.named_steps['clf'].classes_)
disp.plot()
plt.title("Confusion Matrix")
plt.show(block=False)


fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {rocauc:.3f})')
plt.plot([0,1],[0,1],'--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show(block=False)


cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring='roc_auc')
print("\n✅ 5-Fold CV ROC AUC scores:", np.round(cv_scores,3))
print("Mean CV ROC AUC:", np.round(cv_scores.mean(),3))


param_grid = {
    'clf__C': [0.01, 0.1, 1.0, 10],
    'clf__penalty': ['l2'],
    
}

grid = GridSearchCV(clf_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
print("\n✅ GridSearch best params:", grid.best_params_)
print("Best CV ROC AUC:", grid.best_score_)

best_model = grid.best_estimator_


y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:,1]
print("\n✅ Best Model Test ROC AUC:", roc_auc_score(y_test, y_proba_best))
print("Best Model Test Accuracy:", accuracy_score(y_test, y_pred_best))
print("\nClassification report (best model):\n", classification_report(y_test, y_pred_best))


joblib.dump(best_model, "model/logistic_regression_model.joblib")
print("\n✅ Best model saved to model/logistic_regression_model.joblib")


plt.show(block=True)
input("Press ENTER to exit...")



