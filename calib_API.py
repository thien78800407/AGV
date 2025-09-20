import pyzed.sl as sl
import cv2
import numpy as np

def save_stereo_calib(calib, filename="zed2_stereo_intrinsic.yaml"):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)

    # Left camera intrinsics
    mtx_l = np.array([[calib.left_cam.fx, 0, calib.left_cam.cx],
                      [0, calib.left_cam.fy, calib.left_cam.cy],
                      [0, 0, 1]], dtype=np.float64)
    dist_l = np.array(calib.left_cam.disto, dtype=np.float64).reshape(1, -1)

    # Right camera intrinsics
    mtx_r = np.array([[calib.right_cam.fx, 0, calib.right_cam.cx],
                      [0, calib.right_cam.fy, calib.right_cam.cy],
                      [0, 0, 1]], dtype=np.float64)
    dist_r = np.array(calib.right_cam.disto, dtype=np.float64).reshape(1, -1)

    # Stereo transform (Left → Right)
    T = np.array(calib.stereo_transform.get_translation().get(), dtype=np.float64).reshape(3, 1)
    R = np.array(calib.stereo_transform.get_rotation_matrix().r, dtype=np.float64).reshape(3, 3)  # ✅ rotation matrix

    # Save YAML
    fs.write("camera_matrix_left", mtx_l)
    fs.write("dist_coeffs_left", dist_l)
    fs.write("camera_matrix_right", mtx_r)
    fs.write("dist_coeffs_right", dist_r)
    fs.write("R", R)
    fs.write("T", T)

    fs.release()
    print(f"Stereo calibration saved to {filename}")

def main():
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE   # chỉ cần calib, không cần depth
    init_params.camera_resolution = sl.RESOLUTION.HD720  # calibration phụ thuộc vào resolution
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init_params.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Cannot open ZED camera")
        return

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters

    # Save to file
    save_stereo_calib(calib, "zed2_stereo_intrinsic.yaml")

    zed.close()

if __name__ == "__main__":
    main()
