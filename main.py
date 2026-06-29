from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "HxH lore agent is alive"}