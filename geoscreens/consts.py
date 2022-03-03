from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
VIDEO_PATH = Path("/shared/g-luo/geoguessr/videos").resolve()
EXTRACTED_FRAMES_PATH = Path("/shared/gbiamby/geo/video_frames")
DETECTIONS_PATH = Path("/shared/gbiamby/geo/segment/detections")
SEG_PATH = Path("/shared/gbiamby/geo/segment/seg")
LATEST_DETECTION_MODEL_NAME = (
    "gsmoreanch02_012--geoscreens_012-model_faster_rcnn-bb_resnest50_fpn-2b72cbf305"
)
LATEST_DETECTION_MODEL_PATH = Path(f"/shared/gbiamby/geo/models/{LATEST_DETECTION_MODEL_NAME}")
FRAMES_METADATA_PATH = EXTRACTED_FRAMES_PATH / "frame_meta_002.json"
