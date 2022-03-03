from argparse import ArgumentParser, Namespace
from pathlib import Path

from geoscreens.consts import DETECTIONS_PATH, LATEST_DETECTION_MODEL_NAME, SEG_PATH
from geoscreens.video_seg import compute_segments, format_ui_to_gamestates_map, ui_to_gamestates_map


def main(args: Namespace):
    format_ui_to_gamestates_map(ui_to_gamestates_map)
    segments = compute_segments(args, args.model, multi_threaded=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_id", type=str)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=SEG_PATH,
        help="Where to save the segmentation outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LATEST_DETECTION_MODEL_NAME,
        help="Segmentation will be done using detection outputs from this model.",
    )
    parser.add_argument(
        "--dets_path",
        type=Path,
        default=DETECTIONS_PATH,
        help="Where to save the segmentation outputs.",
    )
    parser.add_argument(
        "--force",
        dest="force",
        action="store_true",
        help="Force re-compute & overwrite of segmentation, even if seg files already exist.",
    )
    # parser.add_argument("--ls_url", type=str, default="http://localhost:6008")
    args = parser.parse_args()
    main(args)
