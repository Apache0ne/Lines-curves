import cv2
import numpy as np

from lines_curves.curve_labels import derive_curve_mask


def test_curve_labels_reject_straight_lines_and_keep_ellipses():
    straight = np.zeros((256, 256), np.uint8)
    cv2.line(straight, (20, 40), (230, 40), 255, 2)
    straight_curve = derive_curve_mask(straight)
    assert np.count_nonzero(straight_curve) == 0

    ellipse = np.zeros((256, 256), np.uint8)
    cv2.ellipse(ellipse, (128, 128), (80, 50), 0, 0, 360, 255, 2)
    ellipse_curve = derive_curve_mask(ellipse)
    assert np.count_nonzero(ellipse_curve) > 500
    assert np.all(
        (ellipse_curve > 0)
        <= (cv2.dilate(ellipse, np.ones((3, 3), np.uint8)) > 0)
    )
