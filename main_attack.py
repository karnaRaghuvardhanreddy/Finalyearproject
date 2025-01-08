# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import VGG16, EfficientNetB0
# import numpy as np
# import random

# # --- Simulated Federated Learning Setup ---

# def create_cnn_model(input_shape, num_classes):
#     model = models.Sequential([
#         layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

# def create_vgg16_model(input_shape, num_classes):
#     base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#     base_model.trainable = False
#     model = models.Sequential([
#         base_model,
#         layers.Flatten(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

# def create_efficientnet_model(input_shape, num_classes):
#     base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
#     base_model.trainable = False
#     model = models.Sequential([
#         base_model,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

# # Simulate data poisoning
# def poison_data(data, labels, num_poisoned, target_class):
#     """Inject malicious data by modifying labels to a target class."""
#     poisoned_data = data.copy()
#     poisoned_labels = labels.copy()
#     indices = random.sample(range(len(data)), num_poisoned)
#     for idx in indices:
#         poisoned_labels[idx] = target_class
#     return poisoned_data, poisoned_labels

# # Federated Learning Simulation
# def federated_learning_simulation(models, clients_data, clients_labels, num_rounds, defense_strategy):
#     global_model = models[0]
#     for round_num in range(num_rounds):
#         print(f"Round {round_num+1}/{num_rounds}")

#         local_weights = []
#         for client_idx, (client_data, client_labels) in enumerate(zip(clients_data, clients_labels)):
#             model = models[client_idx]
#             model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#             model.fit(client_data, client_labels, epochs=1, verbose=0)
#             local_weights.append(model.get_weights())

#         # Aggregate weights using the defense strategy
#         aggregated_weights = defense_strategy(local_weights)
#         global_model.set_weights(aggregated_weights)

#     return global_model

# def trimmed_mean_aggregation(weights, trim_fraction=0.1):
#     """Apply trimmed mean to mitigate poisoned updates."""
#     trimmed_weights = []
#     for layer_weights in zip(*weights):
#         stacked = np.stack(layer_weights)
#         trimmed = np.mean(np.sort(stacked, axis=0)[
#             int(len(stacked) * trim_fraction):-int(len(stacked) * trim_fraction)], axis=0)
#         trimmed_weights.append(trimmed)
#     return trimmed_weights

# # Example workflow
# input_shape = (180, 180, 3)
# num_classes = 3
# num_clients = 5
# num_rounds = 10

# # Generate models for each client
# models = [
#     create_cnn_model(input_shape, num_classes),
#     create_vgg16_model(input_shape, num_classes),
#     create_efficientnet_model(input_shape, num_classes),
#     create_cnn_model(input_shape, num_classes),
#     create_vgg16_model(input_shape, num_classes)
# ]

# # Simulate data for clients (normally you'd use real data here)
# clients_data = [np.random.rand(100, *input_shape) for _ in range(num_clients)]
# clients_labels = [
#     tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)
#     for _ in range(num_clients)
# ]

# # Poison one client's data
# poisoned_client_idx = random.randint(0, num_clients - 1)
# clients_data[poisoned_client_idx], clients_labels[poisoned_client_idx] = poison_data(
#     clients_data[poisoned_client_idx], clients_labels[poisoned_client_idx], num_poisoned=20, target_class=0
# )

# # Run federated learning simulation with defense
# final_model = federated_learning_simulation(models, clients_data, clients_labels, num_rounds, trimmed_mean_aggregation)

# print("Federated learning complete. Final model trained.")


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, EfficientNetB0
import numpy as np
import random

# --- Simulated Federated Learning Setup ---

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_vgg16_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_efficientnet_model(input_shape, num_classes):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Simulate data poisoning
def poison_data(data, labels, num_poisoned, target_class):
    """Inject malicious data by modifying labels to a target class."""
    poisoned_data = data.copy()
    poisoned_labels = labels.copy()
    indices = random.sample(range(len(data)), num_poisoned)
    for idx in indices:
        poisoned_labels[idx] = target_class
    return poisoned_data, poisoned_labels

# Federated Learning Simulation
def federated_learning_simulation(models, clients_data, clients_labels, num_rounds, defense_strategy):
    global_model = models[0]
    for round_num in range(num_rounds):
        print(f"Round {round_num+1}/{num_rounds}")

        local_weights = []
        for client_idx, (client_data, client_labels) in enumerate(zip(clients_data, clients_labels)):
            model = models[client_idx]
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(client_data, client_labels, epochs=1, verbose=0)
            local_weights.append(model.get_weights())

        # Aggregate weights using the defense strategy
        aggregated_weights = defense_strategy(local_weights)
        global_model.set_weights(aggregated_weights)

    return global_model

def trimmed_mean_aggregation(weights, trim_fraction=0.1):
    """Apply trimmed mean to mitigate poisoned updates."""
    trimmed_weights = []
    for layer_weights in zip(*weights):
        stacked = np.stack(layer_weights)
        trimmed = np.mean(np.sort(stacked, axis=0)[
            int(len(stacked) * trim_fraction):-int(len(stacked) * trim_fraction)], axis=0)
        trimmed_weights.append(trimmed)
    return trimmed_weights

# Example workflow
input_shape = (180, 180, 3)
num_classes = 3
num_clients = 5
num_rounds = 10

# Use the CNN model for all clients to avoid weight shape mismatches
models = [create_cnn_model(input_shape, num_classes)] * num_clients

# Simulate data for clients (normally you'd use real data here)
clients_data = [np.random.rand(100, *input_shape) for _ in range(num_clients)]
clients_labels = [
    tf.keras.utils.to_categorical(np.random.randint(0, num_classes, 100), num_classes)
    for _ in range(num_clients)
]

# Poison one client's data
poisoned_client_idx = random.randint(0, num_clients - 1)
clients_data[poisoned_client_idx], clients_labels[poisoned_client_idx] = poison_data(
    clients_data[poisoned_client_idx], clients_labels[poisoned_client_idx], num_poisoned=20, target_class=0
)

# Run federated learning simulation with defense
final_model = federated_learning_simulation(models, clients_data, clients_labels, num_rounds, trimmed_mean_aggregation)

print("Federated learning complete. Final model trained.")
