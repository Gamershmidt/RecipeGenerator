import sys
import os
import torch
from pydantic import BaseModel
sys.path.append(os.path.abspath(os.path.join(os.pardir, os.pardir)))
sys.path.append("/path/to/your/project")

from code.models.model import RNNModel, make_pred

# app.py
from fastapi import FastAPI


model = RNNModel(num_layers=3,bidir=True, seq="lstm")
model.load_state_dict(torch.load("models/model_lstm.h5", map_location=torch.device('cpu')), strict=False)

#
app = FastAPI()
class InputData(BaseModel):
    input_data: str

@app.post("/predict")
def predict(data: InputData):
    return {"prediction": f"{make_pred(model, data.input_data)}"}
# @app.post("/predict")
# def predict(data: InputData):
#     return {"prediction": f"{data.input_data}"}
