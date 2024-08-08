import pandas as pd
import os
from dotenv import load_dotenv
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

passenger_id = "PassengerId"
survived_ind = "Survived"
passenger_cls = "Pclass"
p_age = "Age"
sib_spouse_cnt = "SibSp"
par_child_cnt = "Parch"
fare_amt = "Fare"
cabin_id = "Cabin"
embark_port = "Embarked"


def read_training_data() -> pd.DataFrame:
    fpath = os.path.join(os.environ["dataset_dir"], "titanic", "train.csv")
    df = pd.read_csv(fpath)
    return df


def read_test_data() -> pd.DataFrame:
    fpath = os.path.join(os.environ["dataset_dir"], "titanic", "test.csv")
    df = pd.read_csv(fpath)
    return df


def eda(df: pd.DataFrame) -> None:
    print("Sample data")
    print(df.head())
    print()

    print("Describe")
    print(df.describe())
    print()

    print("Null Amount")
    print(df.isna().sum() / len(df))
    print()

    return df


def transform(X: pd.DataFrame) -> pd.DataFrame:
    features = [passenger_cls, sib_spouse_cnt, par_child_cnt]
    return X[features]


def train(X, y) -> DecisionTreeClassifier:
    mdl = DecisionTreeClassifier()
    mdl.fit(X, y)
    y_pred = mdl.predict(X)

    print("Training Results: ")
    print("Survived (Predicted): ", y_pred.sum())
    print("Survived (Actual): ", y.sum())
    print("Accuracy : ", accuracy_score(y, y_pred))
    print("Precision: ", precision_score(y, y_pred))
    print("Recall   : ", recall_score(y, y_pred))
    print("F1 Score : ", f1_score(y, y_pred))
    print()

    cmatrix = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cmatrix, cmap="YlGn", annot=True, vmin=0, fmt=".0f")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Train Confusion")
    plt.show()
    return mdl


def test(mdl: DecisionTreeClassifier, X, y):
    y_pred = mdl.predict(X)
    print("Test Results: ")
    print("Survived (Predicted): ", y_pred.sum())
    print("Survived (Actual): ", y.sum())
    print("Accuracy : ", accuracy_score(y, y_pred))
    print("Precision: ", precision_score(y, y_pred))
    print("Recall   : ", recall_score(y, y_pred))
    print("F1 Score : ", f1_score(y, y_pred))
    print()

    cmatrix = confusion_matrix(y, y_pred)
    ax = sns.heatmap(cmatrix, cmap="YlGn", annot=True, vmin=0, fmt=".0f")
    ax.set(xlabel="Predicted", ylabel="Actual", title="Test Confusion")
    plt.show()
    return y_pred


if __name__ == "__main__":
    load_dotenv(override=True)
    df = read_training_data()
    df = eda(df)

    X = df.drop(columns=[survived_ind])
    y = df[survived_ind]
    X = transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mdl = train(X_train, y_train)
    test(mdl, X_test, y_test)

    plot_tree(mdl)
    plt.show()
