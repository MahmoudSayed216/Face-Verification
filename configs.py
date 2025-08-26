#model
LEARNING_RATE = 0.0001
EPOCHS = 120
ALPHA = 0.3
EMBEDDING_DIM = 128

# ds
PER_SUBJECT_SAMPLES = 4

# env
ENV = "KAGGLE"
DATA_PATH = "/home/mahmoud-sayed/Desktop/Datasets/VGGFace2" if ENV == "LOCAL" else "/kaggle/input/vggface2"
DEVICE = "cpu" if ENV == "LOCAL" else "cuda"
OUTPUT_DIR = "/home/mahmoud-sayed/Desktop/Code/Python/Face Verification 2" if ENV == "LOCAL" else "/kaggle/working"

# logs
DEBUG = True
CHECKPOINT = True
LOG = True
PRINT_EVERY = 20

#train
BATCH_SIZE = 12
SAVE_EVERY = 4
PATIENCE = 8
LR_REDUCTION_FACTOR = 0.5
MODE = 'min'
MIN_LR = 0.00001