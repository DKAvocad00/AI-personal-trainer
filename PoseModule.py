from starlette.responses import FileResponse
from pathlib import Path
from src.utils.utils import *
from src.utils.pose_model import PoseDetector
from src.exercises.exercicse_handler import ExercisePoses
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/data", StaticFiles(directory="data"), name="data")


class Config:
    input_video_path = "data/input/input.mp4"
    output_video_path = "data/output/output.mp4"
    allowed_video_types = {"video/mp4", "video/quicktime", "video/x-msvideo"}


class VideoProcessor:
    def __init__(self, input_video_path: Path = Config.input_video_path,
                 output_video_path: Path = Config.output_video_path):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.exercise_name = "automatic"

    async def process_video(self, mode="automatic") -> None:
        pose_key_point_frames = []
        self.exercise_name = mode
        cap = cv2.VideoCapture(self.input_video_path)
        detector = PoseDetector()
        preprocess_instance = PreprocessVideo()
        detect_model, idx_2_category = load_model(detector.model)

        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                              (1280, 720))

        while cap.isOpened():
            success, img = cap.read()

            if not success:
                break

            preprocess_instance.calculate_fps()
            img = cv2.resize(img, (1280, 720))
            results = detector.findPose(img)
            landmark_list = detector.findLandmarks()

            if mode == "automatic":
                pose_key_point_frames.append(landmark_list.tolist())
                if len(pose_key_point_frames) == 5:
                    self.exercise_name = pose_detect(detect_model, idx_2_category, pose_key_point_frames)
                    del pose_key_point_frames[0]

            ExercisePoses(self.exercise_name, img, detector, preprocess_instance, results)

            preprocess_instance.draw_info(img, self.exercise_name)

            out.write(img)

        out.release()
        cap.release()

    def get_processed_video(self) -> str:
        return self.output_video_path


video_processor = VideoProcessor()


@app.post("/uploadfile/")
async def process_video(file: UploadFile = File(...), mode: str = Form("automatic")):
    if file.content_type not in Config.allowed_video_types:
        raise HTTPException(status_code=400, detail="Invalid file type, must be a video")

    if os.path.exists(video_processor.output_video_path):
        os.remove(video_processor.output_video_path)

    with open(video_processor.input_video_path, 'wb') as video_file:
        video_file.write(file.file.read())

    await video_processor.process_video(mode)

    return JSONResponse(content={"processed_video_path": video_processor.get_processed_video(), "mode": mode})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    processed_video_path = video_processor.get_processed_video()
    return templates.TemplateResponse("video_page.html",
                                      {"request": request, "processed_video_path": processed_video_path})


@app.get("/download_processed_video/")
async def download_processed_video():
    return FileResponse(video_processor.get_processed_video(), media_type="video/mp4")
