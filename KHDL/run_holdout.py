import numpy as np

from data_loader import load_all_data, save_dataset_summary
from evaluation_utils import evaluate_holdout, plot_holdout_comparison


def main():
    X, y = load_all_data()

    print("\n==============================")
    print("THÔNG TIN DỮ LIỆU")
    print("==============================")
    print(f"Tổng số mẫu      : {X.shape[0]}")
    print(f"Số đặc trưng/mẫu : {X.shape[1]}")
    print(f"Số lớp           : {len(np.unique(y))}")
    print(f"Các lớp          : {np.unique(y)}")

    labels = sorted(np.unique(y))

    save_dataset_summary(y)

    holdout_df = evaluate_holdout(X, y, labels)

    plot_holdout_comparison(holdout_df)

    if not holdout_df.empty:
        best_model = holdout_df.sort_values(
            by="F1-score",
            ascending=False
        ).iloc[0]

        print("\nMô hình tốt nhất theo Hold-out F1-score:")
        print(best_model)


if __name__ == "__main__":
    main()