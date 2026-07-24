from pathlib import Path

import cv2
import numpy as np

from scripts.prepare_data import prepare_biped


def test_prepare_biped_official_layout(tmp_path: Path):
    root = tmp_path / "BIPED"
    image_dir = root / "edges" / "imgs" / "train" / "rgbr" / "real"
    edge_dir = root / "edges" / "edge_maps" / "train" / "rgbr" / "real"
    image_dir.mkdir(parents=True)
    edge_dir.mkdir(parents=True)
    image = np.zeros((64, 96, 3), np.uint8)
    cv2.ellipse(image, (48, 32), (25, 12), 0, 0, 300, (255, 255, 255), 2)
    edge = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(image_dir / "sample.jpg"), image)
    cv2.imwrite(str(edge_dir / "sample.png"), edge)

    output = tmp_path / "normalized"
    assert prepare_biped(root, output) == 1
    assert len(list((output / "train" / "images").glob("*.png"))) == 1
    assert len(list((output / "train" / "edges").glob("*.png"))) == 1
    assert len(list((output / "train" / "curves").glob("*.png"))) == 1
