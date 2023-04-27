from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI()

@app.get("/ping")
async def ping():
    return "Server is alive!!"

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...)
):

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")