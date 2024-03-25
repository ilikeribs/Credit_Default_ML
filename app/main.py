import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import json
from pydantic import BaseModel

class loan_dec(BaseModel):

    ORGANIZATION_TYPE: str
    EXT_SOURCE_2: float
    EXT_SOURCE_1: float
    EXT_SOURCE_3: float
    DAYS_EMPLOYED: float
    DAYS_REGISTRATION: float
    AMT_ANNUITY: int
    AMT_CREDIT: float
    DAYS_ID_PUBLISH: float
    DAYS_LAST_PHONE_CHANGE: float


app = FastAPI()

default_classifier =joblib.load("app/simple_model.sav")


class loanJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, loan_dec):
            return obj.model_dump()
        return super().default(obj)

@app.get('/')
def index():
    return {'message': "Welcome to HomeCredit loan default risk evaluation tool"}

@app.post('/evaluate', response_class=JSONResponse)
def predict_loan_status(payload: loan_dec):
    try:
        df = pd.DataFrame([payload.model_dump()])
        pred = default_classifier.predict_proba(df)[:,1] 
        result = {"Loan default probability": "{:.2%}".format(pred.tolist()[0])}

        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)