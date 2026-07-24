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


def test_nested_biped_zip_extraction(tmp_path: Path):
    import zipfile

    from scripts.colab_setup import resolve_biped_root

    package = tmp_path / "kaggle"
    package.mkdir()
    archive_path = package / "BIPEDv2.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("edges/imgs/train/rgbr/real/sample.jpg", b"image")
        archive.writestr("edges/edge_maps/train/rgbr/real/sample.png", b"edge")
    extracted = resolve_biped_root(package, tmp_path / "extracted")
    assert (extracted / "edges/imgs/train/rgbr/real/sample.jpg").exists()
    assert (extracted / "edges/edge_maps/train/rgbr/real/sample.png").exists()
