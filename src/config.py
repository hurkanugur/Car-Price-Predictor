# -------------------------
# Paths
# -------------------------
MODEL_PATH = "../model/car_price_model.pth"
STATISTICS_PATH = "../model/statistics.pkl"
FEATURE_TRANSFORMER_PATH = "../model/feature_transformer.pkl"
DATASET_CSV_PATH = "../data/car_price_dataset.csv"

# -------------------------
# Training hyperparameters
# -------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-6
BATCH_SIZE = 128
NUM_EPOCHS = 100
VAL_INTERVAL = 1

# -------------------------
# Data split ratios
# -------------------------
SPLIT_DATASET = True
SPLIT_RANDOMIZATION_SEED = 42
TRAIN_SPLIT_RATIO = 0.7
VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15
