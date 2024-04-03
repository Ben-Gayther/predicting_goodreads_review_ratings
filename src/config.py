import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

LOGGING = config.get("LOGGING", {})
TRAINING_ARGS = config.get("TRAINING_ARGS", {})
DATA_ARGS = config.get("DATA_ARGS", {})

logging_level = LOGGING.get("LEVEL", "INFO")
logging_format = LOGGING.get(
    "FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

model_name = TRAINING_ARGS.get(
    "MODEL_NAME", "nlptown/bert-base-multilingual-uncased-sentiment"
)
learning_rate = TRAINING_ARGS.get("LEARNING_RATE", 2.0e-5)
max_length = TRAINING_ARGS.get("MAX_LENGTH", 128)
batch_size = TRAINING_ARGS.get("BATCH_SIZE", 1)
epochs = TRAINING_ARGS.get("EPOCHS", 1)
full_dataset = TRAINING_ARGS.get("FULL_DATASET", False)
test_run = TRAINING_ARGS.get("TEST_RUN", True)
output_dir = TRAINING_ARGS.get("OUTPUT_DIR", "models/")
submission_name = TRAINING_ARGS.get("SUBMISSION_NAME", "submission.csv")

input_train_data = DATA_ARGS.get("INPUT_TRAIN_DATA", "data/goodreads_train.csv")
input_test_data = DATA_ARGS.get("INPUT_TEST_DATA", "data/goodreads_test.csv")
output_train_data = DATA_ARGS.get(
    "OUTPUT_TRAIN_DATA", "data/processed_goodreads_train.csv"
)
output_test_data = DATA_ARGS.get(
    "OUTPUT_TEST_DATA", "data/processed_goodreads_test.csv"
)
