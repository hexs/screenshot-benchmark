from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import ImageGrab


def take_screenshot(
        region: Optional[Tuple[int, int, int, int]] = None,
        color: str = "RGB",
) -> np.ndarray:
    img = np.array(ImageGrab.grab(region))
    return img.copy() if color == "RGB" else img[:, :, ::-1].copy()


if __name__ == "__main__":
    for col in ("RGB", "BGR"):
        img = take_screenshot(color=col)
        cv2.imshow(col, img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
