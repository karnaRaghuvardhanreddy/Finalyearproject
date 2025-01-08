import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import pathlib

# Step 1: Load the data
data_dir_train = pathlib.Path("data/Train/")
data_dir_test = pathlib.Path("data/Test/")

# Prepare test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_test,
    batch_size=32,
    image_size=(180, 180),
    label_mode='categorical'
)

# Get class names for output layers
image_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir_train, 
    batch_size=32, 
    image_size=(180, 180), 
    label_mode='categorical'
)
class_names = image_dataset.class_names


# Step 2: Define model creation functions
def create_model_cnn():
    model = Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])
    return model


def create_model_vgg16():
    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    vgg_base.trainable = False
    model = Sequential([
        vgg_base,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(len(class_names), activation='softmax')
    ])
    return model


def create_model_efficientnet():
    eff_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
    eff_base.trainable = False
    model = Sequential([
        eff_base,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(len(class_names), activation='softmax')
    ])
    return model


# Step 3: Federated learning setup
def model_fn_cnn():
    return create_model_cnn()


def model_fn_vgg16():
    return create_model_vgg16()


def model_fn_efficientnet():
    return create_model_efficientnet()


# Federated averaging process
iterative_process_cnn = tff.learning.build_federated_averaging_process(
    model_fn=model_fn_cnn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
)

iterative_process_vgg16 = tff.learning.build_federated_averaging_process(
    model_fn=model_fn_vgg16,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
)

iterative_process_efficientnet = tff.learning.build_federated_averaging_process(
    model_fn=model_fn_efficientnet,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.001)
)

# Initialize federated state
state_cnn = iterative_process_cnn.initialize()
state_vgg16 = iterative_process_vgg16.initialize()
state_efficientnet = iterative_process_efficientnet.initialize()

# Federated training (mock data for demonstration purposes)
# In real-world scenarios, provide federated data for training
client_data = [list(image_dataset.as_numpy_iterator())] * 3  # Mock data from clients
for round_num in range(1, 5):  # Number of training rounds
    state_cnn, _ = iterative_process_cnn.next(state_cnn, client_data)
    state_vgg16, _ = iterative_process_vgg16.next(state_vgg16, client_data)
    state_efficientnet, _ = iterative_process_efficientnet.next(state_efficientnet, client_data)


# Step 4: Model evaluation on centralized test data
def evaluate_federated_model(model_fn, weights, test_data):
    model = model_fn()
    model.set_weights(weights)
    return model.evaluate(test_data, verbose=1)


# Extract final weights
cnn_weights = iterative_process_cnn.get_model_weights(state_cnn).trainable
vgg16_weights = iterative_process_vgg16.get_model_weights(state_vgg16).trainable
efficientnet_weights = iterative_process_efficientnet.get_model_weights(state_efficientnet).trainable

# Evaluate models
cnn_eval = evaluate_federated_model(create_model_cnn, cnn_weights, test_ds)
vgg16_eval = evaluate_federated_model(create_model_vgg16, vgg16_weights, test_ds)
efficientnet_eval = evaluate_federated_model(create_model_efficientnet, efficientnet_weights, test_ds)

# Step 5: Display results
print("Testing Results:")
print(f"CNN - Loss: {cnn_eval[0]:.4f}, Accuracy: {cnn_eval[1]:.4f}")
print(f"VGG16 - Loss: {vgg16_eval[0]:.4f}, Accuracy: {vgg16_eval[1]:.4f}")
print(f"EfficientNet - Loss: {efficientnet_eval[0]:.4f}, Accuracy: {efficientnet_eval[1]:.4f}")
