# Personal-Passion-Projects
A passion project combining baseball, AI, and data analytics. Using Python, I analyze player and team performance, visualize statistics, and explore machine learning to identify trends and make predictions. Built to strengthen my skills in data science, software development, and sports analytics while solving real-world baseball problems.
# BaseballTrack ⚾

## Overview

BaseballTrack is a computer vision project that detects and tracks a baseball using machine learning and computer vision techniques. The goal of the project is to analyze baseball footage in real time or from recorded videos, serving as a foundation for features such as pitch tracking, strike zone analysis, and AI-assisted umpiring.

## Features

* Detects and tracks a baseball in video footage.
* Supports webcam or prerecorded video input.
* Uses deep learning object detection models.
* Draws bounding boxes and tracking information on each frame.
* Designed to be expanded with AI umpire and pitch analysis features.

## Technologies Used

* Python
* OpenCV
* TensorFlow
* YOLO (Ultralytics)
* NumPy

## Project Structure

* `mock1.py` – Main program used to run the baseball tracker.
* `tracker.py` – Ball tracking logic.
* `train.py` – Model training script.
* `VIDEO.py` / `webc.py` – Video and webcam utilities.
* `data.yaml` – Dataset configuration.
* `yolov8n.pt` / `yolov5su.pt` – Pretrained YOLO models.
* `output.tfrecord`, `train.record`, `valid.record` – Training data files.

## Future Improvements

* AI-assisted ball/strike calling.
* Strike zone visualization.
* Pitch trajectory analysis.
* Pitch speed estimation.
* Support for multiple camera angles.
* Improved detection accuracy using custom-trained models.

## Dataset

This project uses a custom baseball dataset for training and evaluation. Additional configuration files and dataset information are included in the repository.

## How to Run

1. Clone the repository.
2. Install the required dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Run the main application:

   ```
   python mock1.py
   ```

## Status

This project is actively being developed as a learning project focused on computer vision, machine learning, and sports analytics. New features and improvements are continuously being added.
