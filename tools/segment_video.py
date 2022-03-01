from argparse import ArgumentParser, Namespace
from pathlib import Path

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
        default=Path("/shared/gbiamby/geo/segment/seg"),
        help="Where to save the segmentation outputs.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=Path(
            "gsmoreanch02_012--geoscreens_012-model_faster_rcnn-bb_resnest50_fpn-2b72cbf305"
        ),
        help="Path of model checkpoint to use for predictions.",
    )
    parser.add_argument(
        "--dets_path",
        type=Path,
        default=Path("/shared/gbiamby/geo/segment/detections"),
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
