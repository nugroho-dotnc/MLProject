import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    mlflow.autolog()
    
    # Load dataset
    df = pd.read_csv('wine_preprocessing.csv')

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="Github_Actions_Run"):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_test)
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("Training pada MLProject sukses.")

if __name__ == "__main__":
    main()