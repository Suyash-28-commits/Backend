import gradio as gr
import joblib
import numpy as np
import google.generativeai as genai
import logging
import os

# -------------------- Setup Logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Global Variables --------------------
model = None
genai_configured = False
MODEL_NAME = "gemini-1.5-flash"
MODEL_PATH = "temp_1_model.pkl"

# -------------------- Load ML Model --------------------
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info("ML model loaded successfully")
    else:
        logger.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading ML model: {str(e)}")

# -------------------- Configure Gemini --------------------
try:
    # Try both possible environment variable names
    GEMINI_API_KEY = os.environ.get("VITE_GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_configured = True
        logger.info("Gemini AI configured successfully")
    else:
        logger.warning("No Gemini API key found in environment variables")
except Exception as e:
    logger.error(f"Error configuring Gemini AI: {str(e)}")

# -------------------- Prediction Function for Gradio --------------------
def predict_and_explain(latitude, longitude, pres, psal):
    """
    Function that takes input values, predicts temperature,
    and generates an explanation using Gemini AI.
    """
    logger.info(f"Received prediction request with: LATITUDE={latitude}, LONGITUDE={longitude}, PRES={pres}, PSAL={psal}")

    # Validate inputs
    if not all(isinstance(val, (int, float)) for val in [latitude, longitude, pres, psal]):
        return "Error: All inputs must be numbers.", "Please check your input values."
        
    if model is None:
        return "Error: ML model not available. Please check the logs.", "Model loading failed."

    try:
        features = np.array([latitude, longitude, pres, psal]).reshape(1, -1)
        predicted_temp = float(model.predict(features)[0])
        genai_text = "Temperature prediction completed."

        # -------------------- Gemini Explanation --------------------
        if genai_configured:
            try:
                explanation_prompt = f"""
You are an intelligent ocean assistant.
Data:
- LATITUDE: {latitude}
- LONGITUDE: {longitude} 
- PRES (pressure): {pres}
- PSAL (salinity): {psal}
- Predicted TEMP: {predicted_temp:.2f} °C
Explain in clear, full sentences what these ocean conditions indicate. Provide insights on the temperature and ocean environment based on the provided data.
"""
                model_instance = genai.GenerativeModel(MODEL_NAME)
                response = model_instance.generate_content(explanation_prompt)
                genai_text = response.text.strip()
            except Exception as e:
                logger.error(f"Gemini explanation failed: {str(e)}")
                genai_text = "Temperature prediction completed. Explanation service temporarily unavailable."
        
        return (
            f"Predicted Temperature: **{predicted_temp:.2f}°C**",
            f"**Explanation:**\n{genai_text}"
        )

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return "An internal error occurred during prediction.", f"Error: {str(e)}"

# -------------------- Gradio Interface --------------------

# Define the input components for Gradio
inputs = [
    gr.Number(label="LATITUDE"),
    gr.Number(label="LONGITUDE"),
    gr.Number(label="PRES (Pressure)"),
    gr.Number(label="PSAL (Salinity)")
]

# Define the output components for Gradio
outputs = [
    gr.Markdown(label="Predicted Temperature"),
    gr.Markdown(label="Gemini AI Explanation")
]

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_and_explain,
    inputs=inputs,
    outputs=outputs,
    title="ARGO Ocean Temperature Prediction",
    description="Enter oceanographic data to predict temperature and get a Gemini AI-powered explanation."
)

# Launch the Gradio application
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
