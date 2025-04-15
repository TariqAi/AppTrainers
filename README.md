![AppTrainers LOGO](https://github.com/user-attachments/assets/31909115-723d-4e2b-8e29-af0b8cd8bcee)
# AppTrainers

Add the files here...

## Set-ExecutionPolicy RemoteSigned

## param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'degree': [2, 3, 4, 5]
                }


## https://forms.gle/Mx7Vt64zApwFmKm97

## Resume : https://novoresume.com/?noRedirect=true

## Form to upload your CV: https://forms.gle/h4mKpnjGmaHptnoKA

➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖➖

# 18-02-2025

requirements.txt

pip install -r requirements.txt

pip freeze > requirements.txt

## To push our project
# git remote add origin https://github.com/user/repo_name.git
# git branch -M main
# git push -u origin main


# 20-02-2025

## https://www.kaggle.com/code/tanmay111999/clustering-pca-k-means-dbscan-hierarchical/notebook

# 15-03-2025

## Descibtion: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis

# 06-04-2025

## https://drive.google.com/drive/folders/1bRPpKIMsqjdd0VsMtV58EwkzqhEG3TgU?usp=drive_link

# 08-04-2025
mohammed jermy kaggle project:
## https://www.kaggle.com/code/mohammadaljermy/imdb-sentiment-analysis

# 13-04-2025
video: https://drive.google.com/drive/folders/1kZ0QVwlwMERyTyi5c72GeqKgr8qAUx2o

Notes:
1) Use YOLO v11
2) tracking the persons


# 🔴 Car license plates
## https://drive.google.com/file/d/1QaekEG4uRI5DyovlgAdrrSFg5028OaUO/view?usp=sharing

1) Use YOLO v11
2) Use Region of Intrest ( ROI ) --> https://polygonzone.roboflow.com/
3) Extract the number using (Tesseract OCR or PaddleOCR or Aspose.OCR)
4) Save it the in txt file (Number_plate, Time)

HINT or STEPS:
- First: Detect the license plate region using object detection.
- Second: Crop the plate region and apply OCR to extract text.




🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰🔰

import cv2
import matplotlib.pyplot as plt
from IPython.display import display, Image
import numpy as np

img_path = "detected_plates/plate_20250415_144839_821598.jpg"

def preprocess_image(img):
    """Enhanced pipeline for OCR on license plates."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction (Median Blur or Gaussian Blur)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological Closing to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return closed

# Load the image
img = cv2.imread(img_path)

if img is None:
    print("❌ Failed to load image. Check the path.")
else:
    # Process the image
    processed_img = preprocess_image(img)
    
    # Convert BGR to RGB for matplotlib (since OpenCV uses BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display original and processed images side by side
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title("Processed Image (OCR-ready)")
    plt.axis('off')
    
    plt.show()





