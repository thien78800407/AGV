import cv2
import cv2.aruco as aruco
import numpy as np
import pyzed.sl as sl
import math

# === Load calibration từ file YAML (OpenCV) ===
def load_calibration_data_opencv(calib_file="zed2_intrinsic.yaml"):
    fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open calibration file: {calib_file}")

    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs

# === Chuyển vector quay thành yaw ===
def get_yaw(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = np.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw

# === Tìm ArUco marker ===
def find_aruco_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(aruco_dict, aruco_params)
        bboxs, ids, _ = detector.detectMarkers(gray)
    else:
        bboxs, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        aruco.drawDetectedMarkers(img, bboxs, ids)
    return bboxs, ids

# === Ước lượng pose ArUco + depth ===
def estimate_pose(bbox, ids, img, mtx, dist, depth_map, depth_scale=100.0, marker_size=0.173):
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(bbox, marker_size, mtx, dist)
    if rvecs is not None and tvecs is not None:
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 0.07)
            yaw = get_yaw(rvec)

            # Dùng SDK depth: lấy điểm trung tâm marker
            corners = bbox[i][0].astype(int)
            cx, cy = np.mean(corners[:, 0]), np.mean(corners[:, 1])
            cx, cy = int(cx), int(cy)

            depth_value = depth_map.get_value(cx, cy)[1]  # đơn vị METERS
            if not np.isnan(depth_value) and not np.isinf(depth_value):
                depth_cm = depth_value * depth_scale
            else:
                depth_cm = -1

            # Pose từ solvePnP (OpenCV)
            x, y, z = tvec[0] * 100  # từ m sang cm
            cv2.putText(img, f"ID={ids[i][0]} X={x:.1f} Y={y:.1f} Z={z:.1f} Yaw={yaw:.1f}deg Depth={depth_cm:.1f}cm",
                        (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            print(f"[Aruco {ids[i][0]}] X={x:.1f} Y={y:.1f} Z={z:.1f} (cm), Yaw={yaw:.1f} deg, Depth={depth_cm:.1f} cm")

# === Main ===
def main():
    # Load calibration của LEFT
    camera_matrix, dist_coeffs = load_calibration_data_opencv("Stereo_camera_Zed2/zed2_intrinsic.yaml")

    # Init ZED2
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.depth_minimum_distance = 0.2  # meters
    init_params.depth_maximum_distance = 10.0 # meters
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Cannot open ZED2 camera")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    image_left = sl.Mat()
    depth_map = sl.Mat()
    sensors_data = sl.SensorsData()
    zed_pose = sl.Pose()

    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Ảnh LEFT
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            frame = image_left.get_data()

            # Depth
            zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)

            # Detect ArUco
            bboxs, ids = find_aruco_markers(frame)
            if ids is not None:
                estimate_pose(bboxs, ids, frame, camera_matrix, dist_coeffs, depth_map)

            # IMU
            zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT)
            imu_data = sensors_data.get_imu_data()

            orientation = imu_data.get_pose().get_orientation().get()  # quaternion
            print(f"[IMU] Orientation (quat): {orientation}")

            # Pose fusion từ SDK
            if zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD) == sl.ERROR_CODE.SUCCESS:
                pose_data = zed_pose.get_translation()
                orient_data = zed_pose.get_orientation().get()
                print(f"[Fusion] Position: {pose_data}, Orientation (quat): {orient_data}")

            # Show
            cv2.imshow("ZED2 LEFT ArUco + Depth", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
