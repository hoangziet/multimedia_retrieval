from fastapi import FastAPI 
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api import endpoints 
from configs import configs
import os

app = FastAPI()
app.include_router(endpoints.router)

@app.get("/")
def root():
    return FileResponse("frontend.html")

keyframes_dir = os.path.abspath(configs.KEYFRAMES_DIR)
app.mount("/keyframes", StaticFiles(directory=keyframes_dir), name="keyframes")

video_dir = os.path.abspath(configs.VIDEO_DIR)
app.mount("/video", StaticFiles(directory=video_dir), name="video")