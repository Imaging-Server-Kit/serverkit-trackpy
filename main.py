"""
Algorithm server definition.
Documentation: https://github.com/Imaging-Server-Kit/cookiecutter-serverkit
"""

from typing import List, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit

import trackpy as tp
import pandas as pd


class Parameters(BaseModel):
    """Defines the algorithm parameters"""

    points: str = Field(
        title="Points",
        description="The points to track.",
        json_schema_extra={"widget_type": "points"},
    )
    search_range: int = Field(
        default=30,
        title="Search range",
        description="Search range in pixels.",
        ge=1,
        le=100,
        json_schema_extra={"widget_type": "int"},
    )
    memory: int = Field(
        default=3,
        title="Memory",
        description="Maximum number of skipped frames for a single track.",
        ge=1,
        le=10,
        json_schema_extra={"widget_type": "int"},
    )

    @field_validator("points", mode="after")
    def decode_points_array(cls, v) -> np.ndarray:
        points_array = serverkit.decode_contents(v)
        return points_array


class TrackpyServer(serverkit.AlgorithmServer):
    def __init__(
        self,
        algorithm_name: str = "trackpy",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        points: np.ndarray,
        search_range: int,
        memory: int,
        **kwargs,
    ) -> List[tuple]:
        """Runs the algorithm."""
        df = pd.DataFrame(
            {
                "frame": points[:, 0],
                "y": points[:, 1],
                "x": points[:, 2],
            }
        )

        linkage_df = tp.link(df, search_range=search_range, memory=memory)

        tracks = linkage_df[["particle", "frame", "y", "x"]].values.astype(float)

        return [(tracks, {"name": "Tracks"}, "tracks")]

    def load_sample_images(self) -> List["np.ndarray"]:
        """Loads one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = TrackpyServer()
app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
