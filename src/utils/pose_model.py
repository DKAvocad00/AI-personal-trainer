import math
from ultralytics import YOLO


class PoseDetector():
    def __init__(self):
        self.lmList = None
        self.results = None
        self.model = YOLO('yolov8s-pose.pt')

    def findPose(self, img):
        self.results = self.model(img)

        return self.results

    def findLandmarks(self):
        if self.results[0].keypoints:
            self.lmList = self.results[0].keypoints.data[0, :, 0:2]
        return self.lmList

    def calculate_angle(self, key_points, left_points_idx, right_points_idx):
        left_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in left_points_idx]
        right_points = [[key_points.data[0][i][0], key_points.data[0][i][1]] for i in right_points_idx]
        line1_left = [
            left_points[1][0].item(), left_points[1][1].item(),
            left_points[0][0].item(), left_points[0][1].item()
        ]
        line2_left = [
            left_points[1][0].item(), left_points[1][1].item(),
            left_points[2][0].item(), left_points[2][1].item()
        ]
        angle_left = self._calculate_angle(line1_left, line2_left)
        line1_right = [
            right_points[1][0].item(), right_points[1][1].item(),
            right_points[0][0].item(), right_points[0][1].item()
        ]
        line2_right = [
            right_points[1][0].item(), right_points[1][1].item(),
            right_points[2][0].item(), right_points[2][1].item()
        ]
        angle_right = self._calculate_angle(line1_right, line2_right)
        angle = (angle_left + angle_right) / 2
        return angle

    def _calculate_angle(self, line1, line2):

        slope1 = math.atan2(line1[3] - line1[1], line1[2] - line1[0])
        slope2 = math.atan2(line2[3] - line2[1], line2[2] - line2[0])

        angle1 = math.degrees(slope1)
        angle2 = math.degrees(slope2)
        angle_diff = abs(angle1 - angle2)

        if angle_diff > 180:
            angle_diff = 360 - angle_diff

        return angle_diff
