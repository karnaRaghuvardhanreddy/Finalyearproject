from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np

# Load the EfficientNetB0 model pre-trained on ImageNet
efficient_net_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(180, 180, 3))

# Freeze the base model
efficient_net_base.trainable = False

# Build the model
model = Sequential([
    efficient_net_base,  # Add the EfficientNetB0 base
    Flatten(),  # Flatten the feature maps
    Dense(128, activation='relu'),  # Dense layer
    Dropout(0.25),  # Dropout for regularization
    Dense(len(class_names), activation='softmax')  # Output layer for classification
])

# Display the model architecture
model.summary()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
epochs = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# Save the model
model.save('skin_cancer_efficientnet_model.h5')
print("Model saved as 'skin_cancer_efficientnet_model.h5'")

# Plot training curves
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)

# Plot Model Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot Model Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Test the model with a single image
test_image_path = os.path.join(data_dir_test, class_names[1], '*')
test_image_files = glob(test_image_path)
test_image = load_img(test_image_files[-1], target_size=(180, 180))
plt.imshow(test_image)
plt.grid(False)

test_image_array = img_to_array(test_image) / 255.0
test_image_array = np.expand_dims(test_image_array, axis=0)

# Predict the class
pred = model.predict(test_image_array)
pred_class = class_names[np.argmax(pred)]

print(f"Actual Class: {class_names[1]}")
print(f"Predicted Class: {pred_class}")
