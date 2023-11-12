
from starlette.responses import StreamingResponse, FileResponse

from src.utils.utils import *
from src.utils.pose_model import PoseDetector
from src.exercises.exercicse_handler import ExercisePoses
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from fastapi.responses import JSONResponse

import cv2

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class VideoProcessor:
    def __init__(self):
        self.input_video_path = "data/input/input.mp4"
        self.output_video_path = "data/output/output.mp4"

    async def process_video(self) -> None:
        pose_key_point_frames = []
        exercise_name = "detection"
        cap = cv2.VideoCapture(self.input_video_path)
        detector = PoseDetector()
        preprocess_instance = PreprocessVideo()
        detect_model, idx_2_category = load_model(detector.model)

        output_width = 1280
        output_height = 720
        output_fps = 30.0

        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps,
                              (output_width, output_height))

        while cap.isOpened():
            success, img = cap.read()

            if not success:
                break

            preprocess_instance.calculate_fps()
            img = cv2.resize(img, (1280, 720))
            results = detector.findPose(img)
            landmark_list = detector.findLandmarks()
            pose_key_point_frames.append(landmark_list.tolist())

            if len(pose_key_point_frames) == 5:
                exercise_name = pose_detect(detect_model, idx_2_category, pose_key_point_frames)
                del pose_key_point_frames[0]

            ExercisePoses(exercise_name, img, detector, preprocess_instance, results)

            preprocess_instance.draw_info(img, exercise_name)

            out.write(img)

        out.release()
        cap.release()

    def get_processed_video(self) -> str:
        return self.output_video_path


video_processor = VideoProcessor()


@app.post("/uploadfile/")
async def process_video(file: UploadFile = File(...)):
    allowed_video_types = {"video/mp4", "video/quicktime", "video/x-msvideo"}

    if file.content_type not in allowed_video_types:
        raise HTTPException(status_code=400, detail="Invalid file type, must be a video")

    if os.path.exists(video_processor.output_video_path):
        os.remove(video_processor.output_video_path)

    with open(video_processor.input_video_path, 'wb') as video_file:
        video_file.write(file.file.read())

    await video_processor.process_video()

    processed_video_path = video_processor.get_processed_video()

    return JSONResponse(content={"processed_video_path": processed_video_path})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    processed_video_path = video_processor.get_processed_video()
    return templates.TemplateResponse("video_page.html",
                                      {"request": request, "processed_video_path": processed_video_path})


@app.get("/download_processed_video/")
async def download_processed_video():
    return FileResponse(video_processor.get_processed_video(), media_type="video/mp4")


@app.get("/stream_processed_video/")
async def stream_processed_video():
    file_path = video_processor.get_processed_video()
    return StreamingResponse(open(file_path, "rb"), media_type="video/mp4")
