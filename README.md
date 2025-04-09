# Deploying a Machine Learning Application
We will use the Iris dataset and model that we developed in the previous session, documented in this Github repository:
- https://github.com/loyola-masters/HuggingFaceHub-Quick-Start

## Running Iris backend app
The backend will serve predictions via a API REST developed with FastAPI library.

Follow these steps to get the application running:

1. Create a new environment and install dependencies:
```bash
conda create -n MLOps-Capstone-Project python=3.9
pip install -r requirements.txt
```

2. (Optional) Script `train_model.py` makes all the work locally, producing the model file `iris_model.joblib`. You can use this script to remember how the model was trained with the `RandomForestClassifier` of Sklearn
```bash
cd backend-iris
python train_model.py
```

3. Use the script `prediction.py` to confirm that predictions are done properly using the model hosted in Hugging Face (`brjapon/iris-dt`). This model was a `DecisionTreeClassifier`, whose accuracy is very similar to the one above.

```bash
cd backend-iris
python prediction.py
```

4. Run the backend with FastAPI
```bash
uvicorn main:app --host 0.0.0.0 --port 80
```
Find the Swagger documentation of the API at `http://127.0.0.1/docs`

Backend FastAPI works as expected

## Running frontend app Iris (Gradio)
Run the gradio app:
```bash
python .\frontend-iris\gradio_app.py
```
This block of the script is the responsible of getting the prediction from the API:
```python
   # Send the POST request to the FastAPI server
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("predicted_species", "No prediction returned")
    except Exception as e:
        return f"Error: {str(e)}"
```


## ANNEX: Using Docker

### Iris backend app: running with Docker

1. **Build** the Docker image:
   ```bash
   docker build -t backend-iris .
   ```
2. **Run** a container from that image:
   ```bash
   docker run -d -p 8001:80 --name fastapi-iris-app fastapi-iris
   ```
3. Test at [http://127.0.0.1:8001](http://127.0.0.1:8001).

Use the above commands on your system where Docker is installed and configured.

(This is done by Docker) Run the FastAPI server (e.g., `uvicorn main:app --reload`) and then send a POST request to `/predict` with the Iris flower measurements to get a prediction.

#### How to test the Iris prediction endpoint

1. Run the server:
   ```bash
   uvicorn main:app --reload
   ```
2. Send a POST request to `http://127.0.0.1:8001/predict` with JSON body, for example:
   ```json
   {
     "SepalLengthCm": 5.1,
     "SepalWidthCm": 3.5,
     "PetalLengthCm": 1.4,
     "PetalWidthCm": 0.2
   }
   ```
   You can test this via [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs) in your browser or with a tool like `curl` or Postman. The response will look like:
   ```json
   {
     "predicted_species": "iris-setosa"
   }
   ```

   **Using Powershell**
```powershell
   Invoke-RestMethod `
  -Uri http://127.0.0.1:8001/predict `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Iris Gradio app: running with Docker
TO DO
- This is your task
