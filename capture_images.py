import pyzed.sl as sl
import cv2
import os

# Thư mục lưu ảnh
LEFT_DIR = "calib_images/left"
RIGHT_DIR = "calib_images/right"
os.makedirs(LEFT_DIR, exist_ok=True)
os.makedirs(RIGHT_DIR, exist_ok=True)

# Khởi tạo ZED
zed = sl.Camera()
init_params = sl.InitParameters()
init_params.depth_mode = sl.DEPTH_MODE.QUALITY
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
init_params.coordinate_units = sl.UNIT.METER

if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Cannot open ZED camera")
    exit()

image_left = sl.Mat()
image_right = sl.Mat()
count = 0

print("Press 'c' to capture stereo image, 'q' to quit.")

# Kích thước màn hình
screen_width, screen_height = 1920, 1080
# Tỷ lệ thu nhỏ thêm (ví dụ 0.6 = 60% so với màn hình)
shrink_ratio = 0.6

while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # Lấy ảnh LEFT và RIGHT
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        zed.retrieve_image(image_right, sl.VIEW.RIGHT)

        # Lấy frame BGR nguyên bản
        frame_left = image_left.get_data()   # 1280x720
        frame_right = image_right.get_data() # 1280x720

        # Ghép ảnh **dọc** (top-bottom)
        combined = cv2.vconcat([frame_left, frame_right])  # Left trên, Right dưới

        # Resize chung 1 lần
        scale_w = screen_width / combined.shape[1]
        scale_h = screen_height / combined.shape[0]
        scale_factor = min(scale_w, scale_h, 1.0)

        # Áp dụng shrink_ratio
        scale_factor *= shrink_ratio

        if scale_factor < 1.0:
            combined = cv2.resize(combined, (0, 0), fx=scale_factor, fy=scale_factor)

        # Hiển thị
        cv2.imshow("Stereo Capture (Left above | Right below)", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            fname_l = os.path.join(LEFT_DIR, f"img_{count:03d}.png")
            fname_r = os.path.join(RIGHT_DIR, f"img_{count:03d}.png")
            cv2.imwrite(fname_l, frame_left)
            cv2.imwrite(fname_r, frame_right)
            print(f"Saved {fname_l} and {fname_r}")
            count += 1
        elif key == ord('q'):
            break

cv2.destroyAllWindows()
zed.close()
print("Stereo capture finished.")
