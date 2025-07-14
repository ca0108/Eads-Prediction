# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import joblib
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

inputfile = r"your file"

def main():
    plt.rcParams['font.family'] = 'Arial'
    df = pd.read_csv(inputfile, encoding='utf-8')
    df.drop(['System'], axis=1, inplace=True)

    features = np.array(df.drop(['target'], axis=1))
    feature_names = df.drop(['target'], axis=1).columns
    target = np.array(df['target'])

    training_features, testing_features, training_target, testing_target = \
        train_test_split(features, target, test_size=0.3, random_state=1)

    models = {
        #'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=0),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=5,random_state=0),
        'XGBoost': xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                                   objective='reg:squarederror', verbosity=0),
        'SVR': SVR(kernel='rbf', C=1.2),
        'Lasso': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=1.0),
        'AdaBoost': AdaBoostRegressor(n_estimators=100,random_state=0),

    }

    model_metrics = {
        'R2_Training': [],
        'R2_Testing': [],
        'RMSE_Testing': [],
        'Cross_Val_Mean': [],
        'MSE_Testing':[]
        #'Correlation_Training': [],
        #'Correlation_Testing': []
    }

    def adjusted_r2(r2_score, n_samples, n_features):

        return 1 - (1 - r2_score) * (n_samples - 1) / (n_samples - n_features - 1)

    model_names = []

    for model_name, model in models.items():
        print(f"\nEvaluating model: {model_name}")
        model.fit(training_features, training_target)

        n_train, p_train = training_features.shape
        n_test, p_test = testing_features.shape

        training_predictions = model.predict(training_features)
        testing_predictions = model.predict(testing_features)

        mse_training = mean_squared_error(training_target, training_predictions)
        mse_testing = mean_squared_error(testing_target, testing_predictions)
        rmse_testing = np.sqrt(mse_testing)

        r2_training = r2_score(training_target, training_predictions)
        r2_testing = r2_score(testing_target, testing_predictions)

        crossScore = cross_val_score(model, training_features, training_target, cv=5)

        model_names.append(model_name)
        model_metrics['R2_Training'].append(r2_training)
        model_metrics['R2_Testing'].append(r2_testing)
        model_metrics['RMSE_Testing'].append(rmse_testing)
        model_metrics['Cross_Val_Mean'].append(crossScore.mean())
        model_metrics['MSE_Testing'].append(mse_testing)

        print(f"score (R^2) - Training: {r2_training:.4f}, Testing: {r2_testing:.4f}")
        #print(f"相关系数 - Training: {corr_training:.4f}, Testing: {corr_testing:.4f}") #新增相关系数
        print(f"test RMSE: {rmse_testing:.4f}")
        print(f"CV score: {crossScore.mean():.4f}")
        print(f"test MSE: {mse_testing:.4f}")

        training_data = pd.DataFrame({
            'True Training Values': training_target,
            'Predicted Training Values': training_predictions
        })
        testing_data = pd.DataFrame({
            'True Testing Values': testing_target,
            'Predicted Testing Values': testing_predictions
        })

        training_data.to_csv(f"./{model_name}_name.csv", index=False)
        testing_data.to_csv(f"./{model_name}_name1.csv", index=False)

        plt.figure(figsize=(6, 6))
        plt.scatter(training_target, training_predictions, color='r', alpha=0.5, label='Training Set')
        plt.scatter(testing_target, testing_predictions, color='g', alpha=0.5, label='Testing Set')

        min_val = min(min(training_target), min(testing_target))
        max_val = max(max(training_target), max(testing_target))
        plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', lw=2, alpha=0.5)

        plt.legend(loc='best')
        plt.xlabel('True Values', fontsize=14, weight='bold')
        plt.ylabel('Predicted Values', fontsize=14, weight='bold')
        plt.tick_params(axis='both', which='major', labelsize=14, width=3)
        ax = plt.gca()
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        plt.title(f'{model_name} Prediction vs True', fontsize=16)
        plt.show()

    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.2

    plt.bar(x - width*1.5, model_metrics['R2_Training'], width, label='R² Training', color='skyblue')
    plt.bar(x - width/2, model_metrics['R2_Testing'], width, label='R² Testing', color='lightgreen')
    plt.bar(x + width/2, model_metrics['RMSE_Testing'], width, label='RMSE Testing', color='salmon')
    plt.bar(x + width*1.5, model_metrics['Cross_Val_Mean'], width, label='Cross Val Mean', color='purple')

    plt.xlabel('Models', fontsize=12, weight='bold')
    plt.ylabel('Score', fontsize=12, weight='bold')
    plt.title('Model Performance Comparison', fontsize=14, weight='bold')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()