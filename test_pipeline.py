import cv2
import numpy as np
import pytest
from predict import predict

def test_predict_mock_output(tmp_path):
    pre_path = str(tmp_path / "pre.png")
    post_path = str(tmp_path / "post.png")
    dummy_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.imwrite(pre_path, dummy_img)
    cv2.imwrite(post_path, dummy_img)

    result = predict(pre_path, post_path, use_mock=True)

    assert result["overlay_image"].shape == (512, 512, 3)
    assert result["mask"].shape == (512, 512)
    assert result["total_buildings"] == 4
    assert set(result["counts"].keys()) == {"no_damage", "minor", "major", "destroyed"}

def test_invalid_image_path():
    with pytest.raises(ValueError):
        predict("fake_before.png", "fake_after.png", use_mock=True)