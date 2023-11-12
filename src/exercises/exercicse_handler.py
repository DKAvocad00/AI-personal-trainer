from src.utils.utils import draw_line


class ExercisePoses:
    def __init__(self, exercise_name, img, detector, preprocess_instance, results):
        self.exercise_name = exercise_name
        self.img = img
        self.detector = detector
        self.preprocess_instance = preprocess_instance
        self.results = results

        self.sport_list = {
            'situp': {
                'left_points_idx': [6, 12, 14],
                'right_points_idx': [5, 11, 13],
                'maintaining': 70,
                'relaxing': 110,
                'concerned_skeletons_idx': [(6, 12, 14), (5, 11, 13), (5, 7, 9), (6, 8, 10), (12, 14, 16), (11, 13, 15)]
            },
            'pushup': {
                'left_points_idx': [6, 8, 10],
                'right_points_idx': [5, 7, 9],
                'maintaining': 140,
                'relaxing': 120,
                'concerned_skeletons_idx': [(6, 12, 14), (5, 11, 13), (5, 7, 9), (6, 8, 10), (12, 14, 16), (11, 13, 15)]
            },
            'squat': {
                'left_points_idx': [11, 13, 15],
                'right_points_idx': [12, 14, 16],
                'maintaining': 110,
                'relaxing': 160,
                'concerned_skeletons_idx': [(11, 13, 15), (12, 14, 16), (5, 7, 9), (6, 8, 10)]
            }
        }

        self()

    def __call__(self):
        if self.exercise_name in self.sport_list:
            self.exercise()
        else:
            print(f"Exercise '{self.exercise_name}' not found in the class.")

    def exercise(self):

        angle = self.detector.calculate_angle(self.results[0].keypoints,
                                              self.sport_list[self.exercise_name]['left_points_idx'],
                                              self.sport_list[self.exercise_name]['right_points_idx'])

        draw_line(self.img, self.detector, self.sport_list[self.exercise_name]['concerned_skeletons_idx'])

        self.preprocess_instance.counter = self.preprocess_instance.repetition_counter(angle, self.sport_list[
            self.exercise_name]['relaxing'], self.sport_list[self.exercise_name]['maintaining'])
