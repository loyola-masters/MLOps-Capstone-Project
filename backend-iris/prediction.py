import pandas as pd
import joblib
import huggingface_hub
from huggingface_hub import hf_hub_download

token = 'YOUR_TOKEN'

# Using model from Hugging Face Hub: https://huggingface.co/brjapon/iris-dt
# Accompanying dataset is hosted in Hugging Face under 'brjapon/iris'
model_path = hf_hub_download(repo_id="brjapon/iris-dt",
                             filename="iris_dt.joblib",
                             repo_type="model",
                             token=token)

model = joblib.load(model_path)


# Example prediction (random values below)
sample_input = pd.DataFrame([{
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}])

prediction = model.predict(sample_input)

# Convert the prediction to the string label
if prediction == 0:
    prediction_species = 'iris-setosa'
elif prediction == 1:
    prediction_species = 'Iris-versicolor'
elif prediction == 2:
    prediction_species = 'Iris-virginica'
else:
    prediction_species = "Invalid prediction"

print(prediction, prediction_species)  # e.g., [0] which might correspond to 'setosa'
