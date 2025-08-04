import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 1. Load data
df = pd.read_csv("/Users/uppiliy/Documents/final/Indian_Kids_Screen_Time.csv")

# 2. Feature lists
numeric_features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
categorical_features = ['Gender', 'Primary_Device', 'Health_Impacts', 'Urban_or_Rural']

# 3. Preprocessing pipelines
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 4. Split into train/val/test (60/20/20)
X = df[numeric_features + categorical_features]
y = df['Exceeded_Recommended_Limit']
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# 5. Compute imbalance ratio
ratio = (y_train == False).sum() / (y_train == True).sum()

# 6. Define pipelines (with simpler XGBoost)
models = {
    "Logistic Regression": Pipeline([
        ('preproc', preprocessor),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ('preproc', preprocessor),
        ('clf', DecisionTreeClassifier(class_weight='balanced', max_depth=6))
    ]),
    "Random Forest": Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=6))
    ]),
    "Support Vector Machine": Pipeline([
        ('preproc', preprocessor),
        ('clf', SVC(class_weight='balanced', probability=True))
    ]),
    "XGBoost": Pipeline([
        ('preproc', preprocessor),
        ('clf', xgb.XGBClassifier(
            scale_pos_weight=ratio,
            eval_metric='logloss',
            verbosity=0,
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        ))
    ])
}

# 7. Baseline validation performance
print("=== Validation Set Performance ===")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    print(f"--- {name} ---")
    print(classification_report(y_val, y_pred, target_names=['False','True']))

# Save the Logistic Regression model after training
logistic_model = models["Logistic Regression"]
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, "logistic_model_pipeline.pkl")
print("Logistic Regression model saved as 'logistic_model_pipeline.pkl'")

# 8. Learning Curve analysis for key models
def plot_learning_curve(pipe, title):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, val_mean, label='Validation score')
    plt.title(title)
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

for name in ["Random Forest", "XGBoost"]:
    plot_learning_curve(models[name], f"Learning Curve: {name}")

# 9. Final evaluation on test set
print("=== Final Test Set Performance ===")
for name, pipe in models.items():
    y_pred = pipe.predict(X_test)
    print(f"--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=['False','True']))
    cm = confusion_matrix(y_test, y_pred, labels=[False, True])
    disp = ConfusionMatrixDisplay(cm, display_labels=[False, True])
    disp.plot()
    plt.title(f"Confusion Matrix – {name}")
    plt.show()
