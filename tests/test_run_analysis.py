import os
from pathlib import Path
import pytest

# Skip the test if required libraries are missing
pytest.importorskip("pandas")
pytest.importorskip("numpy")
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("windrose")

import pandas as pd
import matplotlib
matplotlib.use("Agg")

from valmet_analisis import run_analysis


def test_run_analysis_creates_zip(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=3, freq="H").strftime("%d-%m-%Y %H:%M"),
        "ws": [1.0, 2.0, 3.0],
        "wd": [10, 20, 30],
        "temp_k": [273.15, 274.15, 275.15],
    })
    csv_file = tmp_path / "data.csv"
    df.to_csv(csv_file, index=False)

    output_dir = tmp_path / "out"
    output_dir.mkdir()

    zip_path, _ = run_analysis(str(csv_file), str(output_dir), palette_name="viridis", project_title="test")

    assert zip_path is not None
    assert Path(zip_path).exists()
