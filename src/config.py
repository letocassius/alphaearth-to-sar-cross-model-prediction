"""Project configuration for the AlphaEarth-to-SAR reconstruction pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"
REPORTS_DIR = ROOT_DIR / "reports"

# The first 3 bands of the full-stack GeoTIFF should be Sentinel-1 SAR targets.
SAR_BANDS = ["S1_VV", "S1_VH", "S1_VV_div_VH"]

# The next 64 bands should be AlphaEarth embeddings.
EMBEDDING_BANDS = [f"A{i:02d}" for i in range(64)]

# Default expected input names. You can override these in run_pipeline.py CLI args.
DEFAULT_FULL_STACK_PATH = DATA_RAW_DIR / "sentinel1_alphaearth_small_stack_sf_downtown_golden_gate_2024.tif"
DEFAULT_SAR_PATH = DATA_RAW_DIR / "sentinel1_small_vv_vh_sf_downtown_golden_gate_2024.tif"

DEFAULT_SAMPLE_PROBABILITY = 0.002
DEFAULT_SAMPLING_STRATEGY = "random"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_CHUNK_SIZE = 512
RANDOM_STATE = 42
