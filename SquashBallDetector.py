import cv2
import numpy as np
import os

class SquashBallDetector:

    video_path = '' # ściezka do pliku video
    detected_objects_params = {} #parametry wykrywanych obiektow
    squash_ball_color = ''
    
    def __init__(self, video_path, detected_objects_params, squash_ball_color):
        self.video_path = video_path
        self.detected_objects_params = detected_objects_params
        self.squash_ball_color = squash_ball_color
        
    def proccess_video_file(self):
        capture = cv2.VideoCapture(self.video_path)

        while True:
            _, frame = capture.read() #pojedyncza klatka filmu
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # transformacja kolorow z BGR na HSV
            green_color_mask = self.add_color_mask(hsv_frame) # maska dla zielonego koloru

            detected_balls = self.ball_detection(green_color_mask) #detekcja pilki
            ball_numbers = 0
            if detected_balls is not None:
                for ball in detected_balls:
                    coordinates = int(ball[0][0]),  int(ball[0][1])
                    self.add_circle_to_ball(frame, coordinates)
                    ball_numbers = ball_numbers + 1
            
            if ball_numbers > 1:
                print(ball_numbers)
            
            self.display_video_with_ball(frame)

    def add_color_mask(self, hsv_frame):
        return cv2.inRange(hsv_frame, np.array([35, 52, 72]), np.array([85, 255, 255])) #maska dla zakresu koloru w skali HSV (Hue, Saturation, Value)

    def ball_detection(self, green_color_mask):
        detected_balls = cv2.goodFeaturesToTrack(green_color_mask, **self.detected_objects_params)
        return detected_balls

    def add_circle_to_ball(self, frame, coordinates):
        cv2.circle(frame, coordinates, 8,  (255, 0 , 0) , 2)
    
    def display_video_with_ball(self, frame):
        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            pass

video_path = os.path.join(os.path.abspath(os.getcwd()), 'sample' , '10.mp4')
feature_params = dict(maxCorners = 10, qualityLevel = .01 , minDistance = 200, blockSize = 25)

ball_tracker = SquashBallDetector(video_path, feature_params, 'green')
ball_tracker.proccess_video_file()