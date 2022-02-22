from argparse import ArgumentParser, Namespace

from geoscreens.video_seg import compute_segments, format_ui_to_gamestates_map, ui_to_gamestates_map


def main(args: Namespace):
    format_ui_to_gamestates_map(ui_to_gamestates_map)
    model = "gs_011_extra_augs--geoscreens_011-model_faster_rcnn-bb_resnest50_fpn-3f36fb97fa"
    segments = compute_segments(model)
    # compare_to_ground_truth(segments, seg_gt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_id", type=str)
    # parser.add_argument("--ls_url", type=str, default="http://localhost:6008")
    args = parser.parse_args()
    main(args)
