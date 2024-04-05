import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LOGGING = config.get("LOGGING", {})
TRAINING_ARGS = config.get("TRAINING_ARGS", {})
DATA_ARGS = config.get("DATA_ARGS", {})

logging_level: str = LOGGING.get("LEVEL", "INFO")
logging_format: str = LOGGING.get(
    "FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

model_name: str = TRAINING_ARGS.get(
    "MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment"
)
learning_rate: float = TRAINING_ARGS.get("LEARNING_RATE", 2.0e-5)
max_length: int = TRAINING_ARGS.get("MAX_LENGTH", 128)
batch_size: int = TRAINING_ARGS.get("BATCH_SIZE", 1)
epochs: int = TRAINING_ARGS.get("EPOCHS", 1)
full_dataset: bool = TRAINING_ARGS.get("FULL_DATASET", False)
test_run: bool = TRAINING_ARGS.get("TEST_RUN", True)
output_dir: str = TRAINING_ARGS.get("OUTPUT_DIR", "models/")
submission_name: str = TRAINING_ARGS.get("SUBMISSION_NAME", "submission.csv")

input_train_data: str = DATA_ARGS.get("INPUT_TRAIN_DATA", "data/goodreads_train.csv")
input_test_data: str = DATA_ARGS.get("INPUT_TEST_DATA", "data/goodreads_test.csv")
output_train_data: str = DATA_ARGS.get(
    "OUTPUT_TRAIN_DATA", "data/processed_goodreads_train.csv"
)
output_test_data: str = DATA_ARGS.get(
    "OUTPUT_TEST_DATA", "data/processed_goodreads_test.csv"
)
