import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QDialog, QVBoxLayout, QTextBrowser, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PIL import Image, ImageQt
import numpy as np
import tensorflow as tf
from keras.models import load_model
import csv

# Load your pre-trained model here
# model = tf.keras.models.load_model('/Users/donkoshy/Desktop/EEE4022S/Basic ML/EfficientNetB0-525-(224 X 224)- 98.97.h5')
custmodel= tf.keras.models.load_model('/Users/donkoshy/Desktop/EEE4022S/Basic ML/customInception.h5', custom_objects={'F1_score':'F1_score'})
effmodel= tf.keras.models.load_model('/Users/donkoshy/Desktop/EEE4022S/Basic ML/efficientNet.h5')
vggmodel= tf.keras.models.load_model('/Users/donkoshy/Desktop/EEE4022S/Basic ML/vgg16.keras')
# Specify the path to your CSV file containing class labels
csv_file_path = '/Users/donkoshy/Desktop/EEE4022S/Basic ML/dataset/subdirectory_names.csv'

# Initialize an empty list to store the class labels
class_labels = []

# Read the species names from the CSV file
with open(csv_file_path, mode='r') as file:
    reader = csv.reader(file)
    #next(reader)  # Skip the header row if present
    for row in reader:
        # Assuming the species names are in the first column of the CSV
        class_labels.append(row[0])  # Change the index if needed

class ImageClassificationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Species Classification")
        self.setGeometry(100, 100, 600, 400)
        
        # Outline box for model selection section
        self.model_selection_box = QLabel(self)
        self.model_selection_box.setGeometry(40, 280, 525, 90)
        self.model_selection_box.setStyleSheet("border: 2px solid black;")

        # Outline box for browse button section
        self.browse_box = QLabel(self)
        self.browse_box.setGeometry(40, 30, 255, 230)
        self.browse_box.setStyleSheet("border: 2px solid black;")

        # Outline box for result section
        self.result_box = QLabel(self)
        self.result_box.setGeometry(310, 30, 255, 230)
        self.result_box.setStyleSheet("border: 2px solid black;")
        
                # Initialize predicted_label and confidence_label attributes (moved inside the result box)
        self.output = QLabel("Output", self.result_box)
        self.output.setGeometry(95, 7, 55, 20)
        
        self.preview_label = QLabel("Image Preview:", self.browse_box)  # Label for displaying image preview
        self.preview_label.setGeometry(25, 90, 200, 130)  # Adjust the geometry as needed
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.predicted_label = QLabel("", self)
        self.predicted_label.setGeometry(320, 70, 400, 200)

        self.browse_button = QPushButton("Browse", self.browse_box)
        self.browse_button.setGeometry(25, 25, 200, 50)
        self.browse_button.clicked.connect(self.classify_image)

        self.confidence_label = QLabel("", self)
        self.confidence_label.setGeometry(200, 250, 400, 50)
        
        self.model_label = QLabel("Selected Model: Custom Model", self)  # Default model label
        self.model_label.setGeometry(200, 270, 400, 50)
        
        # Add model selection buttons
        self.custmodel_button = QPushButton("Custom Model", self)
        self.custmodel_button.setGeometry(60, 310, 150, 50)
        self.custmodel_button.clicked.connect(self.select_custmodel)

        self.effmodel_button = QPushButton("EfficientNet Model", self)
        self.effmodel_button.setGeometry(220, 310, 150, 50)
        self.effmodel_button.clicked.connect(self.select_effmodel)

        self.vggmodel_button = QPushButton("VGG Model", self)
        self.vggmodel_button.setGeometry(380, 310, 150, 50)
        self.vggmodel_button.clicked.connect(self.select_vggmodel)

        # Initialize selected model
        self.selected_model = custmodel

    def select_custmodel(self):
        self.selected_model = custmodel
        self.model_label.setText("Selected Model: Custom Model")
        print("Custom Model selected.")

    def select_effmodel(self):
        self.selected_model = effmodel
        self.model_label.setText("Selected Model: EfficientNet Model")
        print("EfficientNet Model selected.")

    def select_vggmodel(self):
        self.selected_model = vggmodel
        self.model_label.setText("Selected Model: VGG-16 Model")
        print("VGG Model selected.")

    def classify_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )

        if file_path:
            image = Image.open(file_path)
            image = image.resize((200, 130))  # Resize the image for display
            
            pixmap = QPixmap.fromImage(ImageQt.ImageQt(image))
            self.preview_label.setPixmap(pixmap)
            
            resized_image = image.resize((224, 224))
            image_array = np.array(resized_image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

            predictions = self.selected_model.predict(image_array)
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]  # Get indices of top 5 predictions
            top_5_classes = [class_labels[i] for i in top_5_indices]
            top_5_probs = predictions[0][top_5_indices]

            # Get the actual species label (the label with the highest confidence)
            actual_class_index = np.argmax(predictions, axis=1)
            actual_class_name = class_labels[actual_class_index[0]]

            # Display the top 5 predictions and probabilities along with the actual species label
            result_text = f"Actual Species: {actual_class_name}\n\nTop 5 Predictions:\n"
            for class_name, prob in zip(top_5_classes, top_5_probs):
                result_text += f"{class_name}: {prob * 100:.2f}%\n"
            self.predicted_label.setText(result_text.strip())
            # image.show()

    def show_class_labels(self):
        class_labels_dialog = ClassLabelsDialog(self)
        class_labels_dialog.exec()
        


class ClassLabelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("All Class Labels")
        self.setGeometry(200, 200, 400, 400)

        layout = QVBoxLayout()

        class_labels_text = QTextBrowser()
        class_labels_text.setPlainText("\n".join(class_labels))  # Display the class labels
        layout.addWidget(class_labels_text)

        self.setLayout(layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassificationApp()
    window.show()
    sys.exit(app.exec())
    



