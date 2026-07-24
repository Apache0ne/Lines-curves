import lzma
from pathlib import Path

import numpy as np

from lines_curves.synthetic import CurvePointBank, discover_point_files


def test_curveml_manifest_prefers_point_sets_and_reads_xz(tmp_path: Path):
    metadata = tmp_path / "details_reproducibility.csv"
    metadata.write_text("\n".join(f"{i},{i + 1}" for i in range(12)), encoding="utf-8")

    point_set = tmp_path / "family" / "point_set_clean.csv.xz"
    point_set.parent.mkdir()
    with lzma.open(point_set, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(f"{i},{i * i},0" for i in range(12)))

    assert discover_point_files(tmp_path) == [point_set]
    (tmp_path / "point_manifest.txt").write_text(
        f"{point_set.relative_to(tmp_path)}\n", encoding="utf-8"
    )
    bank = CurvePointBank(tmp_path)
    points = bank.sample(np.random.default_rng(0))
    assert points is not None
    assert points.shape == (12, 2)
    assert np.allclose(points[3], [3.0, 9.0])
