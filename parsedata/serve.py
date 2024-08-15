from fastapi import FastAPI
import uvicorn
from parsedata.preprocess_tc import preprocess_tc
from parsedata.preprocess_ner import preprocess_ner
from parsedata.dataset import MLDataset
from parsedata.models import *
from parsedata.functions import *
from parsedata.argsconfig import Args


args = Args().get_parser()
app = FastAPI()
@app.get("/")
def read_root(requestdata: str, encoded: bool = False):
    if encoded == True:
        requestdata = requestdata.decode('utf-8')
    
    return {"Hello": "World"}


def serve(MyDeploymentDialog):
    uvicorn.run(app, host="0.0.0.0", port=2066)


def UpdateArgs(MyDeploymentDialog):
    args.port = MyDeploymentDialog.lineEdit_Port.text()
    args.model_dir = MyDeploymentDialog.lineEdit_TrainedModel.text()