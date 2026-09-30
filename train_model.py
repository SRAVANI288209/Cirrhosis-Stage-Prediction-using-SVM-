from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = Path('data/cirrhosis.csv')
MODEL_PATH = Path('models/best_cirrhosis_model.pkl')
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if 'Stage' not in df.columns or 'ID' not in df.columns:
        raise ValueError('Dataset must contain both Stage and ID columns.')

    X = df.drop(columns=['Stage', 'ID'])
    y = df['Stage']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    best_estimators = {}
    prediction_labels_by_model = {}

    model_configs = [
        (
            'Logistic Regression',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', LogisticRegression(max_iter=5000, random_state=42))
            ]),
            {
                'model__C': [0.01, 0.1, 1, 10, 100],
                'model__solver': ['lbfgs', 'liblinear'],
                'model__class_weight': [None, 'balanced']
            }
        ),
        (
            'KNN',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', KNeighborsClassifier())
            ]),
            {
                'model__n_neighbors': [3, 5, 7, 9, 11, 15],
                'model__weights': ['uniform', 'distance'],
                'model__p': [1, 2]
            }
        ),
        (
            'Decision Tree',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', DecisionTreeClassifier(random_state=42))
            ]),
            {
                'model__max_depth': [None, 3, 5, 8, 10, 15],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__criterion': ['gini', 'entropy']
            }
        ),
        (
            'Random Forest',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))
            ]),
            {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [None, 8, 12, 16],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None]
            }
        ),
        (
            'AdaBoost',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', AdaBoostClassifier(random_state=42))
            ]),
            {
                'model__n_estimators': [50, 100, 200, 300],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
                'model__algorithm': ['SAMME', 'SAMME.R']
            }
        ),
        (
            'XGBoost',
            ImbPipeline([
                ('preprocessor', build_preprocessor(X)),
                ('smote', SMOTE(random_state=42)),
                ('model', XGBClassifier(
                    objective='multi:softmax',
                    num_class=4,
                    random_state=42,
                    eval_metric='mlogloss'
                ))
            ]),
            {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [3, 5, 7],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__subsample': [0.7, 0.9],
                'model__colsample_bytree': [0.7, 0.9]
            }
        )
    ]

    for name, pipe, params in model_configs:
        target_encoder = None
        y_fit = y_train
        if name == 'XGBoost':
            target_encoder = LabelEncoder()
            y_fit = target_encoder.fit_transform(y_train)

        grid = GridSearchCV(
            pipe,
            param_grid=params,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1
        )
        grid.fit(X_train, y_fit)
        best_estimator = grid.best_estimator_
        best_estimators[name] = best_estimator
        y_pred = best_estimator.predict(X_test)
        if target_encoder is not None:
            y_pred = target_encoder.inverse_transform(y_pred.astype(int))
            prediction_labels_by_model[name] = target_encoder.classes_.tolist()
        else:
            prediction_labels_by_model[name] = None
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Macro F1': macro_f1,
            'Best Params': grid.best_params_
        })
        print(f'\n=== {name} ===')
        print(f'Accuracy: {acc:.4f}')
        print(f'Macro F1: {macro_f1:.4f}')
        print(f'Best Params: {grid.best_params_}')

    comparison = pd.DataFrame(results).sort_values(by='Macro F1', ascending=False).reset_index(drop=True)
    print('\nFinal comparison:')
    print(comparison)

    best_row = comparison.iloc[0]
    best_model_name = best_row['Model']
    best_model = best_estimators[best_model_name]

    joblib.dump({
        'estimator': best_model,
        'prediction_labels': prediction_labels_by_model[best_model_name]
    }, MODEL_PATH)
    print(f'\nSaved best model to: {MODEL_PATH}')
    print(f'Best model selected: {best_model_name}')


if __name__ == '__main__':
    main()
