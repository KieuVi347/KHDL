import numpy as np

from config import OUTPUT_DIR
from data_loader import load_all_data, save_dataset_summary
from evaluation_utils import (
    evaluate_holdout,
    evaluate_kfold,
    plot_holdout_comparison,
    plot_kfold_comparison
)


def main():
    print("==============================================")
    print("ĐÁNH GIÁ CÁC MÔ HÌNH PHÂN LỚP SIGN LANGUAGE")
    print("==============================================")

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
    kfold_df = evaluate_kfold(X, y)

    plot_holdout_comparison(holdout_df)
    plot_kfold_comparison(kfold_df)

    print("\n==============================================")
    print("HOÀN THÀNH ĐÁNH GIÁ")
    print("==============================================")
    print(f"Tất cả kết quả lưu trong thư mục: {OUTPUT_DIR}")

    if not holdout_df.empty:
        best_holdout = holdout_df.sort_values(
            by="F1-score",
            ascending=False
        ).iloc[0]

        print("\nMô hình tốt nhất theo Hold-out F1-score:")
        print(best_holdout)

    if not kfold_df.empty:
        best_kfold = kfold_df.sort_values(
            by="Mean F1-score",
            ascending=False
        ).iloc[0]

        print("\nMô hình tốt nhất theo K-fold Mean F1-score:")
        print(best_kfold)


if __name__ == "__main__":
    main()