import cv2
import numpy as np
import pandas as pd

from config import (
    TRAIN_DIR,
    TEST_DIR,
    OUTPUT_DIR,
    IMG_SIZE,
    MAX_IMAGES_PER_CLASS
)


def preprocess_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0

    # Chuyển ảnh 2D thành vector 1D
    features = img.flatten()

    return features


def load_dataset_from_folder(data_dir):
    X = []
    y = []

    if not data_dir.exists():
        print(f"Không tìm thấy thư mục: {data_dir}")
        return np.array([]), np.array([])

    class_folders = sorted([
        folder for folder in data_dir.iterdir()
        if folder.is_dir()
    ])

    for class_folder in class_folders:
        label = class_folder.name

        image_files = [
            file for file in class_folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ]

        if MAX_IMAGES_PER_CLASS is not None:
            image_files = image_files[:MAX_IMAGES_PER_CLASS]

        count = 0

        for image_path in image_files:
            features = preprocess_image(image_path)

            if features is None:
                continue

            X.append(features)
            y.append(label)
            count += 1

        print(f"Đã load lớp {label}: {count} ảnh")

    return np.array(X), np.array(y)


def load_all_data():
    print("\n==============================")
    print("ĐANG LOAD DATASET")
    print("==============================")

    X_train, y_train = load_dataset_from_folder(TRAIN_DIR)
    X_test, y_test = load_dataset_from_folder(TEST_DIR)

    if len(X_train) == 0 and len(X_test) == 0:
        raise ValueError(
            "Không tìm thấy dữ liệu ảnh. Kiểm tra lại dataSet/trainingData và dataSet/testingData."
        )

    if len(X_train) > 0 and len(X_test) > 0:
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
    elif len(X_train) > 0:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test

    return X, y


def save_dataset_summary(y):
    labels, counts = np.unique(y, return_counts=True)

    df = pd.DataFrame({
        "Class": labels,
        "Number of Images": counts
    })

    file_path = OUTPUT_DIR / "dataset_summary.csv"
    df.to_csv(file_path, index=False, encoding="utf-8-sig")

    print("\nTHỐNG KÊ DATASET:")
    print(df)
    print(f"\nĐã lưu: {file_path}")

    return df