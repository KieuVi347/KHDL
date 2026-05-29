import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from config import (
    OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    N_SPLITS
)

from model_factory import get_models


def save_confusion_matrix(y_true, y_pred, labels, model_name, file_prefix):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 14))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        values_format="d",
        colorbar=True
    )

    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    safe_model_name = model_name.replace(" ", "_").replace("/", "_")
    file_path = OUTPUT_DIR / f"{file_prefix}_{safe_model_name}.png"

    plt.savefig(file_path, dpi=300)
    plt.close()

    return file_path


def evaluate_holdout(X, y, labels):
    print("\n==============================")
    print("ĐÁNH GIÁ HOLD-OUT")
    print("==============================")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Số mẫu train: {len(X_train)}")
    print(f"Số mẫu test : {len(X_test)}")

    models = get_models()
    results = []

    for model_name, model in models.items():
        print("\n------------------------------")
        print(f"Đang huấn luyện: {model_name}")
        print("------------------------------")

        start_time = time.time()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        end_time = time.time()
        training_time = end_time - start_time

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Training Time (s)": training_time
        })

        print(f"Accuracy      : {accuracy:.4f}")
        print(f"Precision     : {precision:.4f}")
        print(f"Recall        : {recall:.4f}")
        print(f"F1-score      : {f1:.4f}")
        print(f"Training Time : {training_time:.2f} giây")

        report = classification_report(
            y_test,
            y_pred,
            labels=labels,
            zero_division=0
        )

        print("\nClassification Report:")
        print(report)

        safe_model_name = model_name.replace(" ", "_")
        report_path = OUTPUT_DIR / f"holdout_classification_report_{safe_model_name}.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        cm_path = save_confusion_matrix(
            y_test,
            y_pred,
            labels,
            model_name,
            "holdout_confusion_matrix"
        )

        print(f"Đã lưu ma trận nhầm lẫn: {cm_path}")

    results_df = pd.DataFrame(results)
    result_path = OUTPUT_DIR / "holdout_results.csv"
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print("\nBẢNG KẾT QUẢ HOLD-OUT:")
    print(results_df)
    print(f"\nĐã lưu: {result_path}")

    return results_df


def evaluate_kfold(X, y):
    print("\n==============================")
    print("ĐÁNH GIÁ K-FOLD CROSS VALIDATION")
    print("==============================")

    unique_labels, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()

    actual_splits = min(N_SPLITS, min_class_count)

    if actual_splits < 2:
        print("Không đủ dữ liệu mỗi lớp để chạy K-fold.")
        return pd.DataFrame()

    print(f"Số fold sử dụng: {actual_splits}")

    kfold = StratifiedKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = get_models()

    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted"
    }

    results = []

    for model_name, model in models.items():
        print("\n------------------------------")
        print(f"Đang chạy K-fold: {model_name}")
        print("------------------------------")

        start_time = time.time()

        cv_result = cross_validate(
            model,
            X,
            y,
            cv=kfold,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False
        )

        end_time = time.time()
        total_time = end_time - start_time

        accuracy_scores = cv_result["test_accuracy"]
        precision_scores = cv_result["test_precision_weighted"]
        recall_scores = cv_result["test_recall_weighted"]
        f1_scores = cv_result["test_f1_weighted"]

        results.append({
            "Model": model_name,

            "Mean Accuracy": accuracy_scores.mean(),
            "Std Accuracy": accuracy_scores.std(),

            "Mean Precision": precision_scores.mean(),
            "Std Precision": precision_scores.std(),

            "Mean Recall": recall_scores.mean(),
            "Std Recall": recall_scores.std(),

            "Mean F1-score": f1_scores.mean(),
            "Std F1-score": f1_scores.std(),

            "Total Time (s)": total_time
        })

        print("Accuracy từng fold :", accuracy_scores)
        print("Precision từng fold:", precision_scores)
        print("Recall từng fold   :", recall_scores)
        print("F1-score từng fold :", f1_scores)

        print(f"Mean Accuracy : {accuracy_scores.mean():.4f}")
        print(f"Mean F1-score : {f1_scores.mean():.4f}")
        print(f"Total Time    : {total_time:.2f} giây")

    results_df = pd.DataFrame(results)
    result_path = OUTPUT_DIR / "kfold_results.csv"
    results_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print("\nBẢNG KẾT QUẢ K-FOLD:")
    print(results_df)
    print(f"\nĐã lưu: {result_path}")

    return results_df


def plot_holdout_comparison(holdout_df):
    if holdout_df.empty:
        return

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(holdout_df["Model"], holdout_df[metric])
        plt.title(f"So sánh {metric} giữa các mô hình - Hold-out")
        plt.xlabel("Mô hình")
        plt.ylabel(metric)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()

        file_path = OUTPUT_DIR / f"holdout_comparison_{metric.replace('-', '_')}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"Đã lưu biểu đồ: {file_path}")


def plot_kfold_comparison(kfold_df):
    if kfold_df.empty:
        return

    metrics = [
        ("Mean Accuracy", "K-fold Mean Accuracy"),
        ("Mean Precision", "K-fold Mean Precision"),
        ("Mean Recall", "K-fold Mean Recall"),
        ("Mean F1-score", "K-fold Mean F1-score")
    ]

    for metric_col, title in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(kfold_df["Model"], kfold_df[metric_col])
        plt.title(f"So sánh {title} giữa các mô hình")
        plt.xlabel("Mô hình")
        plt.ylabel(metric_col)
        plt.xticks(rotation=30, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()

        safe_name = metric_col.replace(" ", "_").replace("-", "_")
        file_path = OUTPUT_DIR / f"kfold_comparison_{safe_name}.png"
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"Đã lưu biểu đồ: {file_path}")