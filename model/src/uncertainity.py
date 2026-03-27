import tensorflow as tf
import numpy as np
import cv2

# Define the class mapping so the frontend gets actual names, not numbers
CLASS_MAP = {
    0: "Actinic keratoses (akiec)",
    1: "Basal cell carcinoma (bcc)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Dermatofibroma (df)",
    4: "Melanoma (mel)",
    5: "Melanocytic nevi (nv)",
    6: "Vascular lesions (vasc)"
}

def remove_hair_from_bytes(image_bytes):
    """Applies the OpenCV DullRazor algorithm to raw image bytes from the API."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(gray_scale, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    inpainted_image = cv2.inpaint(image, mask, 1, cv2.INPAINT_TELEA)
    
    # Resize to match your model's expected input
    resized_image = cv2.resize(inpainted_image, (128, 128))
    # Normalize pixel values
    final_image = resized_image / 255.0 
    # Expand dimensions for the batch size (1, 128, 128, 3)
    return np.expand_dims(final_image, axis=0)

def predict_with_uncertainty(model, image_bytes, num_passes=30):
    """
    To be called by FastAPI. Takes raw image upload, runs preprocessing, 
    and executes MC Dropout to return the prediction and variance.
    """
    try:
        # 1. Preprocess the uploaded image
        tensor = remove_hair_from_bytes(image_bytes)
        
        # 2. Run Monte Carlo Dropout
        predictions = []
        for _ in range(num_passes):
            # training=True forces the Dropout layers to stay active
            pred = model(tensor, training=True)
            predictions.append(pred.numpy())
            
        predictions = np.array(predictions) # Shape: (30, 1, 7)
        
        # 3. Calculate metrics
        mean_prediction = np.mean(predictions, axis=0)[0] 
        variance = np.var(predictions, axis=0)[0]
        uncertainty_score = float(np.mean(variance))
        
        predicted_class_idx = int(np.argmax(mean_prediction))
        confidence = float(mean_prediction[predicted_class_idx])
        
        # 4. Determine if the model is too uncertain
        # If variance is highly erratic, flag it for human review
        needs_doctor_review = bool(uncertainty_score > 0.02)
        
        return {
            "predicted_class": CLASS_MAP[predicted_class_idx],
            "confidence_score": round(confidence * 100, 2),
            "uncertainty_variance": round(uncertainty_score, 5),
            "needs_doctor_review": needs_doctor_review
        }
        
    except Exception as e:
        return {"error": str(e)}