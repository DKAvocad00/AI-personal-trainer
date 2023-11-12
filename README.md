# AI-Personal-Trainer

## Overview

Welcome to the Video AI Personal Trainer repository! This project is designed to process uploaded videos, detect poses, and generate an output video with additional information about detected exercises. Below is an overview of the project structure, functionalities, and key components.

## Project Structure

The repository is organized into several components:

- **Folder**: `src`
  - **Files**:
    - `utils/utils.py`: Contains utility functions.
    - `utils/pose_model.py`: Implements the PoseDetector class.
    - `exercises/exercise_handler.py`: Manages exercise poses.

- **Folder**: `templates`
  - **HTML Template**: `video_page.html`

- **Main Script**: `PoseModule.py`

## Functionality

The main functionality of the application involves processing uploaded videos, detecting poses, and enhancing the output video with exercise information. The key components include:

- **VideoProcessor Class**: Defined in `PoseModule.py`, this class processes the input video using a PoseDetector, identifies exercise poses, and generates an output video with added information.

- **FastAPI Endpoints**:
  - `/uploadfile/`: Accepts video uploads and initiates video processing.
  - `/`: Displays the main page with the processed video.
  - `/download_processed_video/`: Allows users to download the processed video.
  - `/stream_processed_video/`: Provides a real-time streaming option for the processed video.

## How to Run

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repository.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd your-repository
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI application:**

    ```bash
    uvicorn main:app --reload
    ```

    Open your web browser and go to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the application.

## Usage

1. **Upload a video file:**
   - Click the "Choose File" button on the main page.
   - Select a video file.

2. **Process the video:**
   - Click the "Process Video" button to initiate video processing.

3. **Download the processed video:**
   - Once processing is complete, click the "Download Processed Video" link.

4. **Stream the processed video:**
   - To view the processed video in real-time, click the "Stream Processed Video" link.

## Result

The processed videos are stored in the `data/output/` directory. You can access the processed video through the web interface or directly from the file system.


1. **Video 1 (pushups_output.mp4):**
   - Preview:
     ![Video 1 Preview](data/output/pushups_output.gif)

2. **Video 2 (squats_output.mp4):**
   - Preview:
     ![Video 2 Preview](data/output/squats_output.gif)

