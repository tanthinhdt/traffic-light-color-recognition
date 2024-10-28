from pytube import YouTube
from processors.processor import Processor


class Downloader(Processor):
    def process(self, url: str, output_path: str) -> None:
        """
        Download a video from a YouTube URL.
        :param url:             YouTube URL.
        :param output_path:     Path to output video file.
        """
        YouTube(url).streams.get_highest_resolution().download(output_path)
