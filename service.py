# Tip 1: Use CustomRunnable to apply a custom_logic() on input_data before prediction.
# Tip 2: Try not to change the default variable names!

from typing import Any

import bentoml
from bentoml.io import *
from pydantic import BaseModel


MODEL_NAME = "YOUR_MODEL_NAME:latest"
SERVICE_NAME = "YOUR_SERVICE_NAME"


class InputSchema(BaseModel):
   # Specify your input schema (e.g. how a JSON payload needs to be structured)
   # E.g. Iris dataset expected input JSON structure
   # sepal_len: float
   # sepal_width: float
   # petal_len: float
   # petal_width: float


class CustomRunnable(bentoml.Runnable):
   SUPPORTED_RESOURCES = ("cpu",)  # Add "nvidia.com/gpu" support if needed!
   SUPPORTS_CPU_MULTI_THREADING = True

   def __init__(self):
       # self.model = your custom model load...

   @bentoml.Runnable.method(batchable=False)
   def custom_logic(self, input_data):
       """Summary

       Returns:
           TYPE: Description
       """
       input_data = # Custom data processing...

       return self.model.predict(input_data)


custom_runner = bentoml.Runner(CustomRunnable)
svc = bentoml.Service(SERVICE_NAME, runners=[custom_runner])