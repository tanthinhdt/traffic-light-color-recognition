import os
import sys

sys.path.append(os.getcwd())

import glob
import argparse
from tqdm import tqdm
from processors.detector import Detector


def parse_args():
    """
    Get arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True, help="Path to video or image file or directory"
    )
    parser.add_argument(
        "--modality", type=str, required=True, help="Modality of the input data"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output directory"
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Main function.
    :param args: Arguments from command line.
    """
    detector = Detector()

    source_paths = []
    if args.modality == "image":
        if os.path.isdir(args.source):
            source_paths = glob.glob(os.path.join(args.source, "*.jpg"))
            source_paths.extend(glob.glob(os.path.join(args.source, "*.png")))
            source_paths.extend(glob.glob(os.path.join(args.source, "*.jpeg")))
            if len(source_paths) == 0:
                raise ValueError("No video file(s) found")
        elif args.source.split(".")[-1] in ["jpg", "png", "jpeg"] and os.path.isfile(args.source):
            source_paths = [args.source]
        else:
            raise ValueError("No video file(s) found")
    elif args.modality == "video":
        if os.path.isdir(args.source):
            source_paths = glob.glob(os.path.join(args.source, "*.mp4"))
            source_paths.extend(glob.glob(os.path.join(args.source, "*.avi")))
            if len(source_paths) == 0:
                raise ValueError("No video file(s) found")
        elif args.source.split(".")[-1] in ["mp4", "avi"] and os.path.isfile(args.source):
            source_paths = [args.source]
        else:
            raise ValueError("No video file(s) found")

    for source_path in tqdm(source_paths, total=len(source_paths)):
        output_path = os.path.join(args.output, os.path.basename(source_path))
        detector.process(
            source=source_path,
            modality=args.modality,
            output_path=output_path,
        )


if __name__ == "__main__":
    main(parse_args())
    print("Done!")
