import os
import cv2  
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Dataset path
data_dir = r'C:\Users\raksh\Downloads\DATASET'
categories = ['acne', 'dark spots', 'normal skin', 'puffy eyes', 'wrinkles']
img_size = 128  # Image size for resizing

# Preprocessing the images
def preprocess_images(data_dir, categories, img_size=128):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path):
            print(f"Warning: Directory '{path}' does not exist. Skipping.")
            continue
        label = categories.index(category)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_size, img_size))
                    data.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error processing image '{img_path}': {e}")
    data = np.array(data) / 255.0  
    labels = to_categorical(labels)  
    return data, labels

# Load and preprocess the dataset
data, labels = preprocess_images(data_dir, categories, img_size)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create DenseNet model
def create_densenet_model(input_shape, num_classes):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Initialize the DenseNet model
input_shape = (128, 128, 3)   
num_classes = len(categories)  
densenet_model = create_densenet_model(input_shape, num_classes)

# Train the DenseNet model
print("Training DenseNet model...")
history = densenet_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the DenseNet model
densenet_model.save('densenet_model.keras')
print("DenseNet model saved as 'densenet_model.keras'")

# Predict the labels for the test set
y_pred = densenet_model.predict(X_test)

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
loss, accuracy = densenet_model.evaluate(X_test, y_test, verbose=0)
print(f"\nModel Loss: {loss:.4f}")
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Calculate precision, recall, and F1 score using the classification report
report = classification_report(y_true, y_pred_classes, target_names=categories, output_dict=True)
precision = report['accuracy']
recall = report['accuracy']
f1_score = report['accuracy']

print(f"\nPrecision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}") 
