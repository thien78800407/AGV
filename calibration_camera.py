import cv2
import numpy as np
import glob
import os

# --- Thông số checkerboard ---
CHECKERBOARD = (9, 6)  # số inner corners width x height
SQUARE_SIZE = 0.025     # 25mm

# Thư mục chứa ảnh stereo (Left và Right)
LEFT_DIR = "calib_images/left"
RIGHT_DIR = "calib_images/right"

# Object points chuẩn
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0],0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints = []      # 3d points trong world
imgpoints_left = [] # 2d points trên ảnh left
imgpoints_right = []# 2d points trên ảnh right

# Lấy danh sách ảnh sorted để pair
images_left = sorted(glob.glob(os.path.join(LEFT_DIR, "*.png")))
images_right = sorted(glob.glob(os.path.join(RIGHT_DIR, "*.png")))

if len(images_left) != len(images_right):
    print("Number of left and right images do not match")
    exit()

for fname_l, fname_r in zip(images_left, images_right):
    img_l = cv2.imread(fname_l)
    img_r = cv2.imread(fname_r)
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

    ret_l, corners_l = cv2.findChessboardCorners(gray_l, CHECKERBOARD, None)
    ret_r, corners_r = cv2.findChessboardCorners(gray_r, CHECKERBOARD, None)

    if ret_l and ret_r:
        objpoints.append(objp)
        imgpoints_left.append(corners_l)
        imgpoints_right.append(corners_r)

        # Vẽ corners kiểm tra
        cv2.drawChessboardCorners(img_l, CHECKERBOARD, corners_l, ret_l)
        cv2.drawChessboardCorners(img_r, CHECKERBOARD, corners_r, ret_r)
        cv2.imshow("Left", img_l)
        cv2.imshow("Right", img_r)
        cv2.waitKey(100)

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("Not enough valid images for stereo calibration")
    exit()

# --- Calibrate từng camera trước ---
ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
    objpoints, imgpoints_left, gray_l.shape[::-1], None, None)
ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
    objpoints, imgpoints_right, gray_r.shape[::-1], None, None)

# --- Stereo calibration ---
flags = cv2.CALIB_FIX_INTRINSIC
ret_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_left, imgpoints_right,
    mtx_l, dist_l, mtx_r, dist_r,
    gray_l.shape[::-1], criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5),
    flags=flags
)

print("Stereo Calibration done!")
print("R (rotation between cameras):\n", R)
print("T (translation between cameras):\n", T)

# --- Lưu YAML ---
os.makedirs("Stereo_camera_Zed2", exist_ok=True)
file_path = os.path.join("Stereo_camera_Zed2", "zed2_stereo_intrinsic.yaml")
fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_WRITE)
fs.write("camera_matrix_left", mtx_l)
fs.write("dist_coeffs_left", dist_l)
fs.write("camera_matrix_right", mtx_r)
fs.write("dist_coeffs_right", dist_r)
fs.write("R", R)
fs.write("T", T)
fs.release()
print(f"Saved stereo calibration to {file_path}")
