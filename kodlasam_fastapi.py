import numpy as np
import pandas as pd
import pickle
# from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
import uvicorn



# Modeli Yükleme
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
kodlasam_app = FastAPI()

# kodlasam_app.add_middleware(CORSMiddleware, allow_origins=["*"])

HOST = '0.0.0.0'
PORT = 8863

@kodlasam_app.get("/house_prediction")
def house_prediction(longitude, latitude, housing_median_age,
                     total_rooms, total_bedrooms, population, households,
                     median_income, ocean_proximity):
    
    print(longitude)
    
    input_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income],
    'ocean_proximity': [str(ocean_proximity)]
        })
    
    prediction = model.predict(input_data)
   
    
    result = {'prediction': prediction[0]}
    
    return result

if __name__ == '__main__':
    uvicorn.run("kodlasam_fastapi:kodlasam_app", host=HOST, port=PORT, reload=True)