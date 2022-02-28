import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union, cast

from label_studio_sdk import Client, Project
from requests import Response
from tqdm.contrib.bells import tqdm


def get_labelstudio_export_from_api(
    project: Project, export_type: str, download_all_tasks: str = "true"
) -> Union[List[Dict], Response]:
    response = project.make_request(
        method="GET",
        url=f"/api/projects/{project.id}/export?exportType={export_type}&download_all_tasks={download_all_tasks}",
        timeout=500,
    )
    if response.headers["Content-Type"] == "application/zip":
        return response
    else:
        export = response.json()
        return export
