#model
LEARNING_RATE = 0.001
EPOCHS = 60
ALPHA = 0.3
EMBEDDING_DIM = 512

# ds
PER_SUBJECT_SAMPLES = 4

# env
ENV = "KAGGLE"
DATA_PATH = "/home/mahmoud-sayed/Desktop/Datasets/VGGFace2" if ENV == "LOCAL" else "/kaggle/input/vggface2"
DEVICE = "cpu" if ENV == "LOCAL" else "cuda"
OUTPUT_DIR = "/home/mahmoud-sayed/Desktop/Code/Python/Face Verification 2" if ENV == "LOCAL" else "/kaggle/working"

# logger
DEBUG = True
CHECKPOINT = True
LOG = True

#train
BATCH_SIZE = 8
SAVE_EVERY = 4
PATIENCE = 4
LR_REDUCTION_FACTOR = 0.1
MODE = 'min'
MIN_LR = 0.000001