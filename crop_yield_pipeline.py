"""
Crop Yield Prediction Under Changing Climate Conditions
Convenience wrapper to run pipeline from project root.
"""
from pathlib import Path
import runpy

pipeline_path = Path(__file__).resolve().parent / "notebooks" / "crop_yield_pipeline.py"
runpy.run_path(str(pipeline_path), run_name="__main__")
