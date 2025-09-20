import os
import cv2
import cv2.aruco as aruco
import numpy as np
import pyzed.sl as sl
import threading
import time
import math

# =========================
# Load calibration từ API YAML
# =========================
def load_calibration_data_api(calib_file="zed2_stereo_intrinsic.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_path = os.path.join(script_dir, calib_file)
    calib_path = os.path.normpath(calib_path)
    print(f"Loading stereo calibration: {calib_path}")

    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise IOError(f"Cannot open calibration file: {calib_path}")

    #  Thông số camera trái
    camera_matrix = fs.getNode("camera_matrix_left").mat()
    dist_coeffs = fs.getNode("dist_coeffs_left").mat()

    #  Rotation & Translation giữa Left -> Right
    R = fs.getNode("R").mat()
    T = fs.getNode("T").mat()
    fs.release()

    return camera_matrix, dist_coeffs, R, T

# =========================
# Chuyển rvec -> yaw
# =========================
def get_yaw(rvec):
    rmat, _ = cv2.Rodrigues(rvec)
    yaw = np.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
    return yaw

# =========================
# Detect ArUco markers
# =========================
def find_aruco_markers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    bboxs, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        aruco.drawDetectedMarkers(img, bboxs, ids)
    return bboxs, ids

# =========================
# Estimate pose + Distance Euclidean từ XYZ map
# =========================
def estimate_pose(bboxs, ids, img, mtx, dist, xyz_map, marker_size=0.173):
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(bboxs, marker_size, mtx, dist)
    if rvecs is not None and tvecs is not None:
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 0.07)
            yaw = get_yaw(rvec)

            corners = bboxs[i][0].astype(int)
            cx, cy = int(np.mean(corners[:,0])), int(np.mean(corners[:,1]))

            # Lấy distance từ ZED XYZ map tại pixel trung tâm
            err, point3d = xyz_map.get_value(cx, cy)
            if err == sl.ERROR_CODE.SUCCESS:
                X, Y, Z, _ = point3d  # X, Y, Z (m), confidence
                distance_cm = np.linalg.norm([X, Y, Z]) * 100
            else:
                distance_cm = -1

            # Pose từ solvePnP (OpenCV)
            x, y, z = tvec[0][0]*100, tvec[0][1]*100, tvec[0][2]*100

            cv2.putText(img,
                        f"ID={ids[i][0]} X={x:.1f} Y={y:.1f} Z={z:.1f} "
                        f"Yaw={yaw:.1f}° Dist={distance_cm:.1f}cm",
                        (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            print(f"[Aruco {ids[i][0]}] Pose: X={x:.1f} Y={y:.1f} Z={z:.1f} cm, "
                  f"Yaw={yaw:.1f}°, Distance={distance_cm:.1f} cm")

# =========================
# Threaded ZED Reader
# =========================
class ZEDReader(threading.Thread):
    def __init__(self, zed):
        super().__init__()
        self.zed = zed
        self.runtime_params = sl.RuntimeParameters()
        self.image_left = sl.Mat()
        self.xyz_map = sl.Mat()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            if self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS:
                self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
                self.zed.retrieve_measure(self.xyz_map, sl.MEASURE.XYZ)
                frame_rgba = self.image_left.get_data()
                if frame_rgba is not None and frame_rgba.size != 0:
                    frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_BGRA2BGR)
                    with self.lock:
                        self.frame = frame_bgr.copy()
            else:
                time.sleep(0.001)

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def get_xyz_map(self):
        return self.xyz_map

    def stop(self):
        self.running = False

# =========================
# Main
# =========================
def main():
    cam_mat, dist_coef, R, T = load_calibration_data_api("zed2_stereo_intrinsic.yaml")

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    zed = sl.Camera()
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Cannot open ZED2 camera")
        return

    reader = ZEDReader(zed)
    reader.start()

    try:
        while True:
            frame = reader.get_frame()
            if frame is not None:
                xyz_map = reader.get_xyz_map()
                bboxs, ids = find_aruco_markers(frame)
                if ids is not None:
                    estimate_pose(bboxs, ids, frame, cam_mat, dist_coef, xyz_map)

                cv2.imshow("ZED2 ArUco + Distance", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        reader.stop()
        reader.join()
        zed.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
