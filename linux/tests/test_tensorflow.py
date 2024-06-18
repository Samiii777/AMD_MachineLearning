import tensorflow as tf

# Check if GPU is available through TensorFlow
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
    print("Installed version of TensorFlow-GPU:")
    print(tf.version.VERSION)
else:
    print("ERROR: GPU is not available. It is advised to restart your system if you haven't done so yet and try again after the restart.")
    print("Also make sure you have added the user to the render and video group.")
