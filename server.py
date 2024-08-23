from fastapi import FastAPI
from parsedata.test import parsetext
from parsedata.argsconfig import Args
import uvicorn
import json
import torch
import os
from pydantic import BaseModel


class Requestdata(BaseModel):
    data: str


args = Args().get_parser()
args.checkpoint_path = r'checkpoints\testre'
app = FastAPI()
@app.post("/")
def read_root(serialized: bool, requestdata: Requestdata):
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'bestmodel.pt'))
    args.task_type = checkpoint['task_type']
    args.task_type_detail = checkpoint['task_type_detail']
    data = requestdata.data
    if serialized:
        data = json.loads(data)
    if type(data) == str:
        input = [data]
    elif type(data) == list:
        input = data
    results = parsetext(input, args)
    return results


@app.head("/")
def read():
    return True


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2066)  # 交互式文档：http://10.17.107.43:2066/docs

