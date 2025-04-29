from datetime import datetime, timedelta
import hexss

hexss.check_packages('pillow', 'numpy', 'opencv-python', 'mss', auto_install=True)

from PIL import ImageGrab
import numpy as np
import cv2
import mss


# Common base class for grabbers
class ScreenGrabber:
    def __init__(self, region):
        if isinstance(region, dict):
            self.left = region['left']
            self.top = region['top']
            self.width = region['width']
            self.height = region['height']
        else:
            self.left, self.top, self.width, self.height = region

    def grab(self):
        raise NotImplementedError


class MSSGrabber(ScreenGrabber):
    def grab(self):
        with mss.mss() as sct:
            box = {'left': self.left, 'top': self.top, 'width': self.width, 'height': self.height}
            sct_img = sct.grab(box)
        img = np.array(sct_img)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


class PillowGrabber(ScreenGrabber):
    def grab(self):
        bbox = (self.left, self.top, self.left + self.width, self.top + self.height)
        pil_img = ImageGrab.grab(bbox)
        arr = np.array(pil_img)
        return arr[:, :, ::-1].copy()


def benchmark(grabber, duration=10):
    frame_count = 0
    total_time = 0.0
    end_time = datetime.now() + timedelta(seconds=duration)

    while datetime.now() < end_time:
        t0 = datetime.now()
        frame = grabber.grab()
        cv2.imshow(f"{grabber.__class__.__name__}", cv2.resize(frame, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        dt = (datetime.now() - t0).total_seconds()
        total_time += dt
        frame_count += 1

    fps = frame_count / total_time if total_time > 0 else 0
    print(f"[{grabber.__class__.__name__}] Duration: {duration}s, "
          f"Frames: {frame_count}, Avg FPS: {fps:.2f}")
    return fps


def main():
    region = {'left': 0, 'top': 0, 'width': 1920, 'height': 1080}

    print("Starting MSS benchmark...")
    mss_fps = benchmark(MSSGrabber(region))

    print("Starting Pillow benchmark...")
    pil_fps = benchmark(PillowGrabber(region))

    print("\n=== Results ===")
    print(f"MSS FPS    : {mss_fps:.2f}")
    print(f"Pillow FPS : {pil_fps:.2f}")


if __name__ == "__main__":
    main()

# === Results === (30/4/35 i5-13600k)
# MSS FPS    : 28.18
# Pillow FPS : 17.63
