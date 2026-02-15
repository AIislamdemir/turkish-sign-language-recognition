import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt


def main():
    # ---- Paths ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "landmarks.csv")
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    # ---- Load data ----
    df = pd.read_csv(csv_path)

    # Basic sanity prints (good for screenshots)
    print("[INFO] CSV loaded.")
    print("[INFO] Columns:", list(df.columns)[:8], "...", list(df.columns)[-2:])
    print("[INFO] Total samples:", len(df))
    print("[INFO] Class counts:\n", df["label"].value_counts())

    # ---- Split X / y ----
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values
    y = df["label"].values

    # ---- Train/Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Test size : {len(X_test)}")

    # ---- Model: KNN (with scaling) ----
    # Scaling is important because KNN uses distances.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[RESULT] Test Accuracy: {acc:.4f}")

    # Confusion matrix + report
    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels)

    print("\n[RESULT] Classification Report:\n", report)
    print("[RESULT] Confusion Matrix:\n", cm)

    # ---- Save confusion matrix figure ----
    fig_path = os.path.join(outputs_dir, "confusion_matrix_knn.png")
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix (KNN)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[INFO] Confusion matrix saved to: {fig_path}")


if __name__ == "__main__":
    main()
