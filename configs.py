#model
LEARNING_RATE = 0.0001
EPOCHS = 30
ALPHA = 0.2
EMBEDDING_DIM = 512

# ds
PER_SUBJECT_SAMPLES = 4

# env
ENV = "LOCAL"
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