from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VIDEO_PATH = Path("/shared/g-luo/geoguessr/videos").resolve()
EXTRACTED_FRAMES_PATH = Path("/shared/gbiamby/geo/video_frames")
DETECTIONS_PATH = Path("/shared/gbiamby/geo/segment/detections")
SEG_PATH = Path("/shared/gbiamby/geo/segment/seg")
LATEST_DETECTION_MODEL_NAME = (
    "gs_urls_gamma0.2_03a_014--geoscreens_014-model_faster_rcnn-bb_resnest50_fpn-fb53301bd3"
)
LATEST_DETECTION_MODEL_PATH = Path(f"/shared/gbiamby/geo/models/{LATEST_DETECTION_MODEL_NAME}")
FRAMES_METADATA_PATH = EXTRACTED_FRAMES_PATH / "frame_meta_003.json"
DATASET_VERSION = "014"
DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / f"geoscreens_{DATASET_VERSION}"
    / f"geoscreens_{DATASET_VERSION}.json"
)
