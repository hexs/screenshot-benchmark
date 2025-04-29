from typing import Tuple, Optional
import hexss

hexss.check_packages('pillow', 'mss', 'opencv-python', auto_install=True)

import numpy as np
import cv2
from PIL import ImageGrab
import mss


def take_screenshot(
        region: Optional[Tuple[int, int, int, int]] = None,
        color: str = "RGB",
        backend: str = "PIL"
) -> np.ndarray:
    """
    Capture a screenshot.

    Args:
        region: (left, top, width, height). If None, captures full screen.
        color: 'RGB' or 'BGR' output channels.
        backend: 'PIL' or 'MSS' — which library to use.

    Returns:
        H×W×3 NumPy array in requested channel order.
    """
    backend = backend.upper()
    color = color.upper()
    if backend not in ("PIL", "MSS"):
        raise ValueError("backend must be 'PIL' or 'MSS'")
    if color not in ("RGB", "BGR"):
        raise ValueError("color must be 'RGB' or 'BGR'")

    # Build bounding box
    if region:
        l, t, w, h = region
        if backend == "PIL":
            bbox = (l, t, l + w, t + h)
        else:
            bbox = {"left": l, "top": t, "width": w, "height": h}
    else:
        bbox = None

    # Grab frame
    if backend == "PIL":
        img = np.array(ImageGrab.grab(bbox))  # RGB
        if color == "BGR":
            img = img[:, :, ::-1]
    else:
        with mss.mss() as sct:
            shot = sct.grab(bbox or sct.monitors[0])
        img = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)  # BGR
        if color == "RGB":
            img = img[:, :, ::-1]

    return img.copy()


# Example usage
if __name__ == "__main__":
    rgb = take_screenshot(color="RGB")
    cv2.imshow("RGB PIL", rgb)
    cv2.waitKey(0)

    bgr = take_screenshot(color="BGR")
    cv2.imshow("BGR PIL", bgr)
    cv2.waitKey(0)

    rgb = take_screenshot(color="RGB", backend="MSS")
    cv2.imshow("RGB MSS", rgb)
    cv2.waitKey(0)

    bgr = take_screenshot(color="BGR", backend="MSS")
    cv2.imshow("BGR MSS", bgr)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
