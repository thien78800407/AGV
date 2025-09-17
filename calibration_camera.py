import pyzed.sl as sl
import cv2
import numpy as np

# Khởi tạo ZED
zed = sl.Camera()
init_params = sl.InitParameters()
zed.open(init_params)

# Lấy calibration parameters
calib = zed.get_camera_information().camera_configuration.calibration_parameters

camera_matrix = np.array([[calib.left_cam.fx, 0, calib.left_cam.cx],
                          [0, calib.left_cam.fy, calib.left_cam.cy],
                          [0, 0, 1]])

dist_coeffs = np.array(calib.left_cam.disto)  # [k1, k2, p1, p2, k3]

# Lưu ra file YAML theo chuẩn OpenCV
fs = cv2.FileStorage("Stereo_camera_Zed2/zed2_intrinsic.yaml", cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix", camera_matrix)
fs.write("dist_coeffs", dist_coeffs)
fs.release()

zed.close()
print("Saved calibration to zed2_intrinsic.yaml")
