import pandas as pd
import numpy as np
import argparse
import os
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import OneHotEncoder
import pickle
import optuna

MODEL_PATH = "model/catboost_model.cbm"
OPTUNA_PARAMS_PATH = "model/optuna_params.pkl"

class My_Classifier_Model:
    def __init__(self):
        self.model = CatBoostClassifier()

        if os.path.exists(MODEL_PATH):
            self.model.load_model(MODEL_PATH)
            print(f"Модель загружена из {MODEL_PATH}")
        else:
            print("Модель не найдена! Нужно обучить её.")

        if os.path.exists("model/ohe.pkl"):
            with open("model/ohe.pkl", "rb") as f:
                self.ohe = pickle.load(f)
            print("OneHotEncoder загружен.")
        else:
            print("OneHotEncoder не найден! Нужно обучить модель.")

        if os.path.exists(OPTUNA_PARAMS_PATH):
            with open(OPTUNA_PARAMS_PATH, "rb") as f:
                self.best_params = pickle.load(f)
            print("Лучшие параметры загружены.")
        else:
            self.best_params = None

    def preprocess(self, df, training=True):
        exclude_cols = ["PassengerId", "Cabin", "Name", "Age"]
        for col in df.columns:
            if col in exclude_cols:
                continue
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        df["Age"] = df["Age"].fillna(df["Age"].mean())
        df["Cabin"] = df["Cabin"].fillna("None")
        df["Name"] = df["Name"].fillna("None")

        cost_col = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        df["Cost"] = df[cost_col].sum(axis=1)
        df["haveCost"] = df["Cost"] > 0

        cabin = df["Cabin"].str.split("/", expand=True)
        cabin.columns = ["Deck", "Num", "Side"]
        df["Deck"] = cabin["Deck"]
        df["Num"] = cabin["Num"].fillna("None")
        df["Side"] = cabin["Side"].fillna("None")

        cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'haveCost', 'Deck', 'Side']
        num_cols = ['Age', 'Cost', 'Num']

        ohe = OneHotEncoder(sparse_output=False)

        if training:
            ohe.fit(df[cat_cols])
            self.ohe = ohe
        else:
            ohe = self.ohe

        cat_cols_transform = ohe.transform(df[cat_cols])

        numeric = df[num_cols].values

        X = np.concatenate([numeric, cat_cols_transform], axis=1)

        X_columns = num_cols + list(np.concatenate(ohe.categories_))

        if training:
            y = df["Transported"].values
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42)
            return X_train, X_valid, y_train, y_valid, X_columns
        else:
            return X, X_columns

    def objective(self, trial):
        params = {
            "iterations": trial.suggest_int("iterations", 200, 1000, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.1, 10.0),
            # Случайное подмешивание данных
            "random_strength": trial.suggest_float("random_strength", 0.1, 10.0),
            "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 10),
            # Количество категорий для One-Hot Encoding
            "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
            # Метод обновления весов
            "depth": trial.suggest_int("depth", 4, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "Logloss"
        }

        X_train, X_valid, y_train, y_valid, _ = self.preprocess(self.df, training=True)

        model = CatBoostClassifier(**params, cat_features=None, verbose=False)
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, verbose=False)

        preds = model.predict(X_valid)
        accuracy = accuracy_score(y_valid, preds)
        return accuracy

    def train(self, dataset_path):
        self.df = pd.read_csv(dataset_path)

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=20)

        self.best_params = study.best_params
        with open(OPTUNA_PARAMS_PATH, "wb") as f:
            pickle.dump(self.best_params, f)
        print(f"Оптимизация завершена! Лучшие параметры: {self.best_params}")

        X_train, X_valid, y_train, y_valid, _ = self.preprocess(self.df, training=True)
        self.model = CatBoostClassifier(**self.best_params, cat_features=None)
        self.model.fit(X_train, y_train, eval_set=(X_valid, y_valid), early_stopping_rounds=50, verbose=False)

        os.makedirs("model", exist_ok=True)
        self.model.save_model(MODEL_PATH)

        with open("model/ohe.pkl", "wb") as f:
            pickle.dump(self.ohe, f)
        print(f"Модель обучена и сохранена в {MODEL_PATH}")

    def predict(self, dataset_path):
        df = pd.read_csv(dataset_path)
        X, X_columns = self.preprocess(df, training=False)

        predictions = self.model.predict(X)

        result_df = pd.DataFrame({"PassengerId": df.PassengerId, "Transported": predictions})
        result_df.to_csv("data/results.csv", index=False)

        print("Предсказания сохранены в '../data/results.csv'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Titanic Model CLI")
    parser.add_argument("command", choices=["train", "predict"], help="Команда: train или predict")
    parser.add_argument("--dataset", required=True, help="Путь к CSV-файлу с данными")

    args = parser.parse_args()
    classifier = My_Classifier_Model()

    if args.command == "train":
        classifier.train(args.dataset)
    elif args.command == "predict":
        classifier.predict(args.dataset)