import gradio as gr
import requests

# Set your FastAPI endpoint (update the host and port if different)
API_URL = "http://127.0.0.1/predict"

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Prepare the data payload for the POST request
    payload = {
        "SepalLengthCm": sepal_length,
        "SepalWidthCm": sepal_width,
        "PetalLengthCm": petal_length,
        "PetalWidthCm": petal_width
    }

    # Send the POST request to the FastAPI server
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_species", "No prediction returned")
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio Interface
interface = gr.Interface(
    fn=predict_iris,
    inputs=["number", "number", "number", "number"],
    outputs="text",
    live=True,
    title="Iris Species Identifier",
    description="Enter the four measurements to predict the Iris species."
)

if __name__ == "__main__":
    interface.launch()
