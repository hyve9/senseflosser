import tensorflow as tf
from tensorflow.config import list_physical_devices, experimental

print(tf.__version__)

# Configure TensorFlow to use GPU
gpus = list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU found: {gpu}")
    try:
        # Set memory growth to prevent TensorFlow from using all GPU memory
        for gpu in gpus:
            experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
