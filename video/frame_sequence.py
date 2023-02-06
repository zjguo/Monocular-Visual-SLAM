import cv2 as cv
import os

class FrameSequence:

    def __init__(self, frames_generator):
        self.frames = frames_generator

    @classmethod
    def from_frames_directory(cls, directory):
        cls.frames = generator_from_folder(directory)
        return cls

    @classmethod
    def from_mp4(cls, mp4_file):
        video_capture = cv.VideoCapture(mp4_file)
        cls.frames = generator_from_video_capture(video_capture)
        return cls

def generator_from_video_capture(video_capture):
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            yield None
        yield frame

def generator_from_folder(directory):
    _, _, files = next(os.walk(directory))
    files = [x for x in files if x.endswith(('.png','.jpg'))]
    files.sort()
    for i in files:
        yield cv.imread(os.path.join(directory, i))
    yield None

