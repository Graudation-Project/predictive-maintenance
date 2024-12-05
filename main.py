import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from fastapi import FastAPI,File,UploadFile
import pandas as pd
from models.modeling.modeling import Modeling
import uvicorn
from pydantic import BaseModel,Field
import h2o
from datetime import datetime,date,timedelta

app = FastAPI()
class InputData(BaseModel):
    OperatingHours : float = Field(
        0,
        description="The Total Working Hours Of The Machine After One Month",
        ge=0
        )
    Temperature: float = Field(
        0,
        description="The Daily Average Temperature Of the Machine",
        ge=0
        )
    VibrationLevel: float = Field(
        0,
        description="The Average Vibration Level Of The Machine",
        ge=0
        )
    LoadPercentage: float = Field(
        0,
        description="The Total Working Hours Of The Machine",
        ge=0
        )
    LastMaintanenceDate: date = Field(
        ...,
        description="The start date for calculating the next maintenance date in YYYY-MM-DD format"
    )

@app.post("/Extract")
async def extract_data(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    # Load Data
    model = Modeling(data)
    # Train and Save The Model
    model.model_train()
    return {"Data Loaded Successfully..."}

@app.post("/Predict")
async def predict(request:InputData):
    h2o.init()
    model_path = "/media/moaaz/Work/Graduation Team/PM/project/models/trained_models/StackedEnsemble_BestOfFamily_6_AutoML_1_20241204_213546"
    try:
        loaded_model = h2o.load_model(model_path)

        data = {
            'Operating Hours': [request.OperatingHours],
            'Temperature (C)': [request.Temperature],
            'Vibration Level (mm/s)': [request.VibrationLevel],
            'Load Percentage (%)': [request.LoadPercentage]
        }
        new_data = pd.DataFrame(data)
        new_data_h2o = h2o.H2OFrame(new_data)

        predictions = loaded_model.predict(new_data_h2o)
        
        prediction_value = predictions.as_data_frame().iloc[0, 0]  
        reference_date = datetime.today()  # Use today's date as the start date
        predicted_date = reference_date + timedelta(hours=round(prediction_value))
        formatted_date = predicted_date.strftime("%Y/%m/%d")
        return {"Next Maintenance Date": formatted_date}

    except Exception as e:
        print(f"Error loading model or making prediction: {e}")
        return {"Error": str(e)}
    
    

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)