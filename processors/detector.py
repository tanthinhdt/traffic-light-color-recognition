import os
import cv2
import torch
import numpy as np
import moviepy.editor as mp
from tqdm import tqdm
from processors.processor import Processor


class Detector(Processor):
    def __init__(self) -> None:
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5n6")
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.light_colors = {
            "red": (0, 0, 255),
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
        }

        # Define color ranges for red, yellow, and green
        self.lower_red = np.array([0, 70, 50])
        self.upper_red = np.array([10, 255, 255])

        self.lower_yellow = np.array([21, 39, 64])
        self.upper_yellow = np.array([40, 255, 255])

        self.lower_green = np.array([40, 100, 100])
        self.upper_green = np.array([80, 255, 255])

    def process(self, modality: str, source: str = None, output_path: str = None) -> None:
        """
        Process function.
        :param modality:        Modality of the input data.
        :param source:          Path to video or image file or directory.
        :param output_path:     Path to output directory.
        """
        if modality == "realtime":
            self.detect_realtime()
        elif modality == "video":
            self.detect_video(source, output_path)
        elif modality == "image":
            self.detect_image(source, output_path)
        else:
            raise ValueError("Invalid modality")

    def detect_realtime(self) -> None:
        """
        Detect traffic lights in realtime.
        """
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Traffic Light Detector", self.detect_image_array(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        cv2.destroyAllWindows()

    def detect_video(self, video_path: str, output_path: str) -> None:
        """
        Detect traffic lights in a video and save the output video.
        :param video_path:      Path to video file.
        :param output_path:     Path to output video file.
        """
        video = mp.VideoFileClip(video_path)
        duration = video.duration
        fps = video.fps
        video.close()

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        writer = cv2.VideoWriter(
            output_path,
            self.fourcc,
            fps,
            (frame_width, frame_height),
        )

        progress_bar = tqdm(
            self.get_frames(cap),
            desc="Process frames",
            unit="frame",
            total=int(duration * fps),
            leave=False,
        )
        for frame in progress_bar:
            writer.write(self.detect_image_array(frame))

        cap.release()
        writer.release()

    def get_frames(self, cap: cv2.VideoCapture) -> np.ndarray:
        """
        Generator function to get frames from a video.
        :param cap:     Video capture object.
        """
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                yield frame

    def detect_image(self, image_path: str, output_path: str) -> None:
        """
        Detect traffic lights in an image and save the output image.
        :param image_path:      Path to image file.
        """
        image = cv2.imread(image_path)
        cv2.imwrite(output_path, self.detect_image_array(image))

    def detect_image_array(self, image: np.ndarray) -> np.ndarray:
        """
        Detect traffic lights in an image and return the image with bounding boxes.
        :param image:   Image array.
        :return:        Image array with bounding boxes.
        """
        results = self.model(image).xyxy[0].tolist()
        if len(results) == 0:
            return image
        traffic_lights = []
        for result in results:
            x1, y1, x2, y2, _, label = map(int, result)
            if label == 9:
                traffic_lights.append([x1, y1, x2, y2])
        if len(traffic_lights) == 0:
            return image
        return self.draw_bounding_boxes(image, traffic_lights)

    def draw_bounding_boxes(self, image: np.ndarray, bounding_boxes: list) -> np.ndarray:
        """
        Draw bounding boxes on an image.
        :param image:           Image array.
        :param bounding_boxes:  List of bounding boxes.
        """
        # Draw the bounding box
        for bounding_box in bounding_boxes:
            x1, y1, x2, y2 = bounding_box

            # Determine the color of the traffic light
            color = self.detect_color(image[y1:y2, x1:x2, :])

            # Draw the label and bounding box on the image
            image = cv2.rectangle(
                img=image,
                pt1=(x1, y1), pt2=(x2, y2),
                color=self.light_colors[color],
                thickness=5,
            )
            image = cv2.putText(
                img=image,
                text=color,
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=self.light_colors[color],
                thickness=3,
            )
        return image

    def detect_color(self, image: np.ndarray) -> str:
        """
        Detect the color of a traffic light.
        :param image:   Image array.
        :return:        Color of the traffic light.
        """
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for each color
        red_mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)
        yellow_mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)
        green_mask = cv2.inRange(hsv_image, self.lower_green, self.upper_green)

        # Calculate the percentage of each color in the image
        total_pixels = image.shape[0] * image.shape[1]
        red_pixels = np.sum(red_mask > 0)
        yellow_pixels = np.sum(yellow_mask > 0)
        green_pixels = np.sum(green_mask > 0)

        red_percentage = (red_pixels / total_pixels) * 100
        yellow_percentage = (yellow_pixels / total_pixels) * 100
        green_percentage = (green_pixels / total_pixels) * 100

        # Determine the color based on the highest percentage
        if red_percentage > yellow_percentage and red_percentage > green_percentage:
            return "red"
        elif yellow_percentage > red_percentage and yellow_percentage > green_percentage:
            return "yellow"
        return "green"
