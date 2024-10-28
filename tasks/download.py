import os
import sys
import argparse
import validators
sys.path.append(os.getcwd())


def parse_args():
    """
    Get arguments from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str, required=True,
        help="Url or path to text file containing urls"
    )
    parser.add_argument(
        "--output",
        type=str, required=True,
        help="Path to output directory"
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    """
    Main function.
    :param args: Arguments from command line.
    """
    if validators.url(args.url):
        from processors.downloader import Downloader
        downloader = Downloader()
        downloader.process(args.url, args.output)
    elif os.path.isfile(args.url):
        with open(args.url) as f:
            urls = f.readlines()
        urls = [url.strip() for url in urls]
        for url in urls:
            if not validators.url(url):
                raise ValueError(f"Invalid url: {url}")
        from processors.downloader import Downloader
        downloader = Downloader()
        for url in urls:
            downloader.process(url, args.output)
    else:
        raise ValueError("Invalid url or path to text file containing urls")


if __name__ == "__main__":
    main(parse_args())
