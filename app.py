import cv2
import numpy as np
import joblib
import gradio as gr

from skimage.feature import hog, local_binary_pattern

# Load trained model and scaler
svm = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

class_names = [
    "Bacterial Pneumonia",
    "Corona Virus Disease",
    "Normal",
    "Tuberculosis",
    "Viral Pneumonia"
]

IMG_SIZE = 128  # MUST match training

def extract_hog(img):
    # img must be 2D grayscale
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=False
    )
    return features

def extract_lbp(img):
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist



def predict_image(image):
    # Convert uploaded image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # Feature extraction
    hog_feat = extract_hog(gray)
    lbp_feat = extract_lbp(gray)

    combined = np.hstack([hog_feat, lbp_feat])
    combined_scaled = scaler.transform([combined])

    # Predict class index
    disease = svm.predict(combined_scaled)[0]

    # Confidence score (SVM decision function)
    decision_scores = svm.decision_function(combined_scaled)

    if decision_scores.ndim > 1:
        confidence = np.max(decision_scores)
    else:
        confidence = abs(decision_scores)

    return (
        f"Detected Lung Condition: {disease}\n"
        f"Confidence Score: {confidence:.2f}"
    )

interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Lung X-ray Image"),
    outputs=gr.Textbox(label="Prediction Result"),
    title="Lung Disease Classification from X-ray",
    description=(
        "Upload a lung X-ray image. "
        "The system will classify it into one of the following categories:\n"
        "Bacterial Pneumonia, Corona Virus Disease, Normal, Tuberculosis, Viral Pneumonia."
    )
)

interface.launch()



