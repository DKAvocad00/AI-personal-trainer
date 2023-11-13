import cv2
import numpy as np
import time
import os
import torch
from data.for_detect.train import LSTM
import json


class PreprocessVideo:
    def __init__(self) -> None:
        self.reaching = False
        self.reaching_last = False
        self.state_keep = False
        self.counter = 0
        self.prev_time = time.time()
        self.fps = 0
        self.reaching_angle = False

    def repetition_counter(self, current_angle, start_angle, end_angle):

        if current_angle < end_angle:
            self.reaching = True
            if current_angle < end_angle - 15:
                self.reaching_angle = True
        if current_angle > start_angle:
            self.reaching = False

        if self.reaching != self.reaching_last:
            self.reaching_last = self.reaching
            if self.reaching:
                self.state_keep = True
            if not self.reaching and self.state_keep:
                self.counter += 1
                self.reaching_angle = False
                self.state_keep = False

        return self.counter

    def calculate_fps(self):
        current_time = time.time()
        elapsed_time = current_time - self.prev_time
        self.fps = 1 / elapsed_time if elapsed_time > 0 else 0
        self.prev_time = current_time

    def draw_info(self, img, exercise_name):
        img_y, img_x, _ = img.shape

        cv2.rectangle(img, (30, 25), (350, 220), (254, 254, 235), cv2.FILLED)
        cv2.rectangle(img, (30, 25), (350, 220), (253, 248, 134), 2, cv2.LINE_AA)

        text_size = cv2.getTextSize(exercise_name, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = 40
        text_y = int(((30, 25)[1] + (280, 75)[1] + text_size[1]) / 2)
        cv2.putText(img, f"Exercise: {exercise_name}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 170, 24),
                    2)

        count_x = 40
        count_y = text_y + text_size[1] + 10
        cv2.putText(img, f"Count: {self.counter}", (count_x, count_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 24), 2)

        fps_x = count_x
        fps_y = count_y + text_size[1] + 10
        cv2.putText(img, f"FPS: {self.fps:.2f}", (fps_x, fps_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 170, 24), 2)

        info_x = fps_x
        info_y = fps_y + text_size[1] + 10

        if self.reaching_angle:
            cv2.putText(img, f"INFO: MAKE HIRE", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, f"INFO: Everything good", (info_x, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),
                        2)


def load_model(model):
    with open(os.path.join("./data/for_detect/checkpoint", 'idx_2_category.json'), 'r') as f:
        idx_2_category = json.load(f)
    detect_model = LSTM(17 * 2, 8, 2, 3, model.device)
    model_path = os.path.join("./data/for_detect/checkpoint", 'best_model.pt')
    model_weight = torch.load(model_path)
    detect_model.load_state_dict(model_weight)

    return detect_model, idx_2_category


def pose_detect(model, idx_2_category, pose_key_point_frames):
    input_data = torch.tensor(pose_key_point_frames)
    input_data = input_data.reshape(5, 17 * 2)
    x_mean, x_std = torch.mean(input_data), torch.std(input_data)
    input_data = (input_data - x_mean) / x_std
    input_data = input_data.unsqueeze(dim=0)
    input_data = input_data.to(model.device)
    rst_detector = model(input_data)
    idx = rst_detector.argmax().cpu().item()
    exersice_type = idx_2_category[str(idx)]
    return exersice_type


def draw_line(img, detector, points):
    for p1, p2, p3 in points:
        key_points = [detector.lmList[p1].cpu().numpy(),
                      detector.lmList[p2].cpu().numpy(),
                      detector.lmList[p3].cpu().numpy()]
        key_points = np.array(key_points).reshape((-1, 1, 2)).astype(int)

        x1, y1 = key_points[0][0]
        x2, y2 = key_points[1][0]
        x3, y3 = key_points[2][0]

        cv2.line(img, (x1, y1), (x2, y2), (253, 248, 134), 2)
        cv2.line(img, (x3, y3), (x2, y2), (253, 248, 134), 2)
        cv2.circle(img, (x1, y1), 6, (254, 254, 235), cv2.FILLED)
        cv2.circle(img, (x2, y2), 6, (254, 254, 235), cv2.FILLED)
        cv2.circle(img, (x3, y3), 6, (254, 254, 235), cv2.FILLED)

    return img
