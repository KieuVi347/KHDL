from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_DIR / "dataSet" / "trainingData"
TEST_DIR = PROJECT_DIR / "dataSet" / "testingData"

OUTPUT_DIR = PROJECT_DIR / "model_evaluation_results"
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 64

# Nếu máy yếu hoặc dữ liệu quá nhiều, đổi None thành 100 hoặc 300
MAX_IMAGES_PER_CLASS = None

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5