import pandas as pd
import joblib
import huggingface_hub
from huggingface_hub import hf_hub_download

huggingface_hub.login(token = 'YOUR_TOKEN')

# Using model from Hugging Face Hub: https://huggingface.co/brjapon/iris-dt
# Accompanying dataset is hosted in Hugging Face under 'brjapon/iris'
model_path = hf_hub_download(repo_id="brjapon/iris-dt",
                             filename="iris_dt.joblib",
                             repo_type="model")

model = joblib.load(model_path)

# Example prediction (random values below)
sepal_length = 5.1
sepal_width = 3.5
petal_length = 1.4
petal_width = 0.2

def predict_iris(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    # Convert the input parameters into a Dataframe, as expected by the model
    input = pd.DataFrame([{
    "SepalLengthCm": SepalLengthCm,
    "SepalWidthCm": SepalWidthCm,
    "PetalLengthCm": PetalLengthCm,
    "PetalWidthCm": PetalWidthCm
    }])
    prediction = model.predict(input)
    
    # Convert the prediction to the string label
    if prediction == 0:
        return 'iris-setosa'
    elif prediction == 1:
        return 'Iris-versicolor'
    elif prediction == 2:
        return 'Iris-virginica'
    else:
        return "Invalid prediction"

prediction = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
print(prediction)  # e.g., [0] which might correspond to 'setosa'

