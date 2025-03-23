from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Dataset path and categories
data_dir = r'C:\Users\raksh\Downloads\DATASET'
categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']

# Preprocessing the images
def preprocess_images(data_dir, categories, img_size=128):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)
    data = np.array(data) / 255.0  # Normalize the images to range [0, 1]
    labels = to_categorical(labels, num_classes=len(categories))  # One-hot encoding
    return data, labels

# Load and preprocess the dataset
data, labels = preprocess_images(data_dir, categories)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer with softmax activation
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile the model
input_shape = (128, 128, 3)  # Input shape for RGB images of size 128x128
num_classes = len(categories)  # Number of categories in the dataset
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the trained model
model.save('cnn_model.keras')
print("CNN Model saved as 'cnn_model.keras'")

# Predict the labels for the test set
y_pred = model.predict(X_test)

# Convert predictions from one-hot encoding to class labels
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=categories))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Evaluate the model on the test set for loss and accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Calculate precision, recall, and F1 score using the classification report
report = classification_report(y_true, y_pred_classes, target_names=categories, output_dict=True)

precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

