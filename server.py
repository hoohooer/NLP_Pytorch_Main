from fastapi import FastAPI
from parsedata.test import parsetext
from parsedata.argsconfig import Args
import uvicorn
import json
from pydantic import BaseModel


class Requestdata(BaseModel):
    data: str


args = Args().get_parser()
args.checkpoint_path = r'checkpoints\testner'
app = FastAPI()
@app.post("/")
def read_root(serialized: bool, requestdata: Requestdata):
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
    uvicorn.run(app, host="0.0.0.0", port=2066)