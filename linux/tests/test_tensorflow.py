import tensorflow as tf
import datetime
import os

LOG_FILE = "log.txt"

# Function to log messages
def log(level, message):
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}\n")

# Check if GPU is available through TensorFlow
if tf.config.list_physical_devices('GPU'):
    gpus = tf.config.list_physical_devices('GPU')
    log("INFO", f"GPU is available. Found {len(gpus)} GPU device(s).")
    log("INFO", f"Installed version of TensorFlow: {tf.__version__}")
        
    # Test basic GPU operation
    with tf.device('/GPU:0'):
        try:
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
            log("INFO", "Successfully ran a basic GPU operation.")
        except Exception as e:
            log("ERROR", f"Failed to run operations on GPU: {str(e)}")
else:
    log("ERROR", "GPU is not available for TensorFlow.")
    log("ERROR", "Make sure you have installed the GPU version of TensorFlow")
    log("ERROR", "You may need to restart your system if you haven't done so after installation.")
