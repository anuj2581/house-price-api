from fastapi import FastAPI
from pydantic import BaseModel,Field

from sklearn.linear_model import LinearRegression
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import pandas as pd


# create fast API

app=FastAPI(title="House price predictin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
        # data = {
        #     "size": [800, 1000, 1200, 1500, 1800, 2000],
        #     "bedrooms": [2, 2, 3, 3, 4, 4],
        #     "Price": [25, 30, 40, 55, 65, 75]
        # }

        # df = pd.DataFrame(data)
        # X = df[["size","bedrooms"]]
        # y = df["Price"]

        # # tarin and test
        # x_train,x_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

        # # Train Model

        # model=LinearRegression()
        # # model.fit(X,y)
        # model.fit(x_train,y_train)
        # # Evalute Model
        # y_pred= model.predict(x_test)
        # mae= mean_absolute_error(y_test,y_pred)

# request scema
model=joblib.load("house_price_model.pkl")
class HouseInput(BaseModel):
    size:int = Field(...,gt=100,description="House size must be greater than 100 sqft.")
    bedrooms:int=Field(... ,lt=10,gt=0 ,description="no of bedroom should be greater than 0 and less than 10")
    
# API Endpoint

@app.post("/predict")
def predict_price(request_model:HouseInput):
    prediction  = model.predict([[request_model.size,request_model.bedrooms]])
    return {
        "predict_price": round(prediction[0],2)
        #"model_mae_lakh" :round(mae,2)
    }


