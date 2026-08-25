#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# left_knee_node_ros.py — Local node (l_knee) con ACC+GYRO → probs → ROS
# left_knee_node_ros.py — Local node (l_knee) con QUAT → GUI → ROS
# + CSV logging: raw window + scaled window + probs (1 row per inference)

import os
import sys
import time
import csv
import types
from datetime import datetime
from collections import deque
from ctypes import c_void_p

import numpy as np
import torch
import torch.nn as nn
import joblib

import rospy
from har_msgs.msg import Probs  # Generated in har_msgs/msg/Probs.msg
from std_msgs.msg import Header, Bool
from sensor_msgs.msg import Imu

import warnings
warnings.filterwarnings("ignore")

# =========================== PARAMETERS =================================
SENSOR_ID   = "l_knee"
MAC_ADDR    = "FA:F1:20:99:CB:B4"                       # MAC MetaMotionRL
MODEL_PATH  = "./cnn_lstm_fold1.pth"                    # Trained model (input_dim=6 for acc+gyro)
SCALER_PATH = "./scaler_model_ag_left_knee.pkl"         # Scaler joblib.dump() 6 columns [ax,ay,az,gx,gy,gz]
K_CLASSES   = 6                                         # Number of classes
WINDOW_SIZE = 50                                        # 50 samples (~1s a 50Hz)
TARGET_HZ   = 50                                        # Frequency of inference/publication
CONNECT_RETRIES = 8
TOPIC_PROBS = "har/probs/left_knee"                     # ROS topic to publish
TOPIC_QUAT  = "har/imu_left_knee"                       # ROS topic for publishing quaternions

# CSV logging (window + probs)
LOG_DIR = "./logs"
LOG_RAW_AND_PROBS = True       # ON/OFF
# ========================================================================

# ---------- SDK MetaWear ----------
from mbientlab.metawear import MetaWear, parse_value, libmetawear
try:
    from mbientlab.metawear.cbindings import (
        SensorFusionMode, SensorFusionData, SensorFusionAccRange, SensorFusionGyroRange,
        FnVoid_VoidP_Data, Data
    )
except ImportError:
    from mbientlab.metawear.cbindings import (
        SensorFusionMode, SensorFusionData, SensorFusionAccRange, SensorFusionGyroRange,
        FnVoid_VoidP_DataP as FnVoid_VoidP_Data, Data
    )

# ===================== LOCAL MODEL (CNN+LSTM) ===========================
class CNN_LSTM_Sensor(nn.Module):
    def __init__(self, input_dim=6, cnn_out_channels=16, lstm_hidden=32, lstm_layers=1, output_dim=6):
        super(CNN_LSTM_Sensor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (B, F, T)
        x = self.cnn(x)          # (B, C, T')
        x = x.permute(0, 2, 1)   # (B, T', C)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]   # Last time
        return self.fc(x)

# =========================== MAIN NODE ================================
class LKneeNode:
    def __init__(self):
        self.sensor_id = SENSOR_ID
        self.mac = MAC_ADDR
        self.K = K_CLASSES
        self.window_size = WINDOW_SIZE
        self.period = 1.0 / float(TARGET_HZ)

        # CSV logger (opens at START, closes at STOP)
        self.log_enabled = bool(LOG_RAW_AND_PROBS)
        self.csv_f = None
        self.csv_w = None
        self.csv_path = None
        self.run_id = 0
        self.started = False
        self.logging_active = False
        self.sensor_ready = False
        if self.log_enabled:
            os.makedirs(LOG_DIR, exist_ok=True)

        # Buffer window (6 features: acc xyz + gyro xyz)
        self.buffer = deque(maxlen=self.window_size)

        # Temporary dictionary to join acc+gyro by timestamp
        self._partial = {}

        # ROS Publisher
        self.pub = rospy.Publisher(TOPIC_PROBS, Probs, queue_size=10)
        self.pub_quat = rospy.Publisher(TOPIC_QUAT, Imu, queue_size=10)
        self.seq = 0

        # Model (input_dim=6 for acc+gyro)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"[{self.sensor_id}] Inference device: {self.device}")

        self.model = CNN_LSTM_Sensor(input_dim=6, output_dim=self.K)
        try:
            state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=self.device)

        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self._warmup_model()

        # Shim NumPy 2.x -> 1.x
        if not hasattr(np, "_core") and hasattr(np, "core"):
            mod = types.ModuleType("numpy._core")
            sys.modules["numpy._core"] = mod
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
            try:
                sys.modules["numpy._core.umath"] = np.core.umath
            except AttributeError:
                pass

        # Scaler (joblib) — 6 columns [ax, ay, az, gx, gy, gz]
        rospy.loginfo(f"[{self.sensor_id}] Cargando scaler desde: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)

        # MetaWear
        self.dev = MetaWear(self.mac, hci_mac="54:EF:33:9B:83:78")
        self.sig_cacc = None
        self.sig_cgyr = None
        self.cb_cacc = None
        self.cb_cgyr = None
        self.sig_quat = None
        self.cb_quat = None
        self._quat_toggle = False  # For subsampling 100Hz -> ~50Hz

        # Start topics at the same time
        self.state_sub = rospy.Subscriber(
            "/har/start",
            Bool,
            self._start_cb,
            queue_size=1
        )

    def _warmup_model(self, num_iters: int = 10):
        rospy.loginfo(f"[{self.sensor_id}] Warm-up del modelo ({num_iters} iteraciones)...")

        dummy = torch.zeros((1, self.window_size, 6), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for _ in range(num_iters):
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

                _ = self.model(dummy)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)

        rospy.loginfo(f"[{self.sensor_id}] Warm-up completado.")

    # CSV helpers
    def _open_csv_for_run(self):
        if not self.log_enabled:
            return
        if self.csv_f:
            # If the CSV was open, we close it
            self._close_csv_for_run()

        self.run_id += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(
            LOG_DIR,
            f"{self.sensor_id}_run{self.run_id:03d}_{ts}_raw_window_probs.csv"
        )

        self.csv_f = open(self.csv_path, "w", newline="")
        self.csv_w = csv.writer(self.csv_f)

        header = [
            "date", "time",
            "sensor_id", "run_id", "seq",
            "t0_ns", "t_send_ns", "ros_stamp_ns",
            "window_size", "K"
        ]

        for i in range(self.window_size):
            header += [
                f"raw_{i}_ax", f"raw_{i}_ay", f"raw_{i}_az",
                f"raw_{i}_gx", f"raw_{i}_gy", f"raw_{i}_gz"
            ]

        for i in range(self.window_size):
            header += [
                f"scl_{i}_ax", f"scl_{i}_ay", f"scl_{i}_az",
                f"scl_{i}_gx", f"scl_{i}_gy", f"scl_{i}_gz"
            ]

        for k in range(self.K):
            header += [f"p{k}"]

        self.csv_w.writerow(header)
        self.csv_f.flush()
        rospy.loginfo(f"[{self.sensor_id}] CSV abierto (START) -> {self.csv_path}")

    def _close_csv_for_run(self):
        try:
            if self.csv_f:
                self.csv_f.flush()
                self.csv_f.close()
                rospy.loginfo(f"[{self.sensor_id}] CSV cerrado (STOP) -> {self.csv_path}")
        finally:
            self.csv_f = None
            self.csv_w = None
            self.csv_path = None

    def _log_window_and_probs(self, seq: int, t0_ns: int, probs_np: np.ndarray, X_raw: np.ndarray, X_scaled: np.ndarray):
        if not self.csv_w:
            return
        try:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S.%f")[:-3]

            ros_stamp_ns = int(rospy.Time.now().to_nsec())
            t_send_ns = int(time.time_ns())

            row = [
                date_str, time_str,
                self.sensor_id, int(self.run_id), int(seq),
                int(t0_ns), int(t_send_ns), int(ros_stamp_ns),
                int(self.window_size), int(self.K)
            ]

            row += [float(v) for v in X_raw.reshape(-1)]
            row += [float(v) for v in X_scaled.reshape(-1)]
            row += [float(x) for x in probs_np.ravel()]

            self.csv_w.writerow(row)

            # Flush every 10 rows
            if (seq % 10) == 0 and self.csv_f:
                self.csv_f.flush()
        except Exception as e:
            rospy.logwarn(f"[{self.sensor_id}] CSV log error: {e}")


    # ---------- BLE / Sensor Fusion ----------
    def connect_and_config(self):
        rospy.loginfo(f"[{self.sensor_id}] Conectando a {self.mac} …")
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                self.dev.connect()
                rospy.loginfo(f"[{self.sensor_id}] Conectado (intento {attempt}).")
                break
            except Exception as e:
                rospy.logwarn(f"[{self.sensor_id}] Falló conexión (intento {attempt}): {e}")
                time.sleep(min(8, 2 ** (attempt - 1)))
        else:
            raise RuntimeError(f"No se pudo conectar a {self.mac}")
        
        # Parameters BLE
        libmetawear.mbl_mw_settings_set_connection_parameters(self.dev.board, 7.5, 7.5, 0, 6000)

        # Sensor Fusion → IMU_PLUS (only acc+gyro)
        libmetawear.mbl_mw_sensor_fusion_set_mode(self.dev.board, SensorFusionMode.IMU_PLUS)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(self.dev.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(self.dev.board, SensorFusionGyroRange._1000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.dev.board)

        # Corrected signals
        self.sig_cacc = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.dev.board, SensorFusionData.CORRECTED_ACC)
        self.sig_cgyr = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.dev.board, SensorFusionData.CORRECTED_GYRO)
        self.sig_quat = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.dev.board, SensorFusionData.QUATERNION)

        if not self.sig_cacc or not self.sig_cgyr or not self.sig_quat:
            raise RuntimeError("No se pudieron obtener las señales de ACC/GYRO.")
        
        # Callbacks
        self.cb_cacc = FnVoid_VoidP_Data(self._cb_cacc)
        self.cb_cgyr = FnVoid_VoidP_Data(self._cb_cgyr)
        self.cb_quat = FnVoid_VoidP_Data(self._cb_quat)
        libmetawear.mbl_mw_datasignal_subscribe(self.sig_cacc, None, self.cb_cacc)
        libmetawear.mbl_mw_datasignal_subscribe(self.sig_cgyr, None, self.cb_cgyr)
        libmetawear.mbl_mw_datasignal_subscribe(self.sig_quat, None, self.cb_quat)

        # Enable and start (two streams)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.CORRECTED_ACC)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.CORRECTED_GYRO)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.QUATERNION)
        libmetawear.mbl_mw_sensor_fusion_start(self.dev.board)
        rospy.loginfo(f"[{self.sensor_id}] Sensor Fusion (ACC+GYRO+QUAT) iniciado.")
        self.sensor_ready = True

    def disconnect(self):
        try:
            libmetawear.mbl_mw_sensor_fusion_stop(self.dev.board)
            if self.sig_cacc:
                libmetawear.mbl_mw_datasignal_unsubscribe(self.sig_cacc)
            if self.sig_cgyr:
                libmetawear.mbl_mw_datasignal_unsubscribe(self.sig_cgyr)
            if self.sig_quat:
                libmetawear.mbl_mw_datasignal_unsubscribe(self.sig_quat)
        finally:
            self._close_csv_for_run()
            self.dev.disconnect()
            rospy.loginfo(f"[{self.sensor_id}] Desconectado.")
            self.sensor_ready = False

    # Start callback (/har/start) 
    def _start_cb(self, msg: Bool):
        if msg.data:
            self.started = True
            self.seq = 0
            self.buffer.clear()
            self._partial.clear()
            rospy.loginfo(f"[{self.sensor_id}] RUN=True (START) recibido")
        elif (not msg.data) and self.started:
            self.started = False
            rospy.loginfo(f"[{self.sensor_id}] RUN=False (STOP) recibido")
            if self.logging_active:
                self._close_csv_for_run()
                self.logging_active = False
        else:
            pass

    # Callbacks
    def _cb_cacc(self, ctx: c_void_p, data: c_void_p):
        try:
            a = parse_value(data)
            ts_ns = int(data.contents.epoch * 1e6)
            p = self._partial.setdefault(ts_ns, {'acc': None, 'gyr': None})
            p['acc'] = [a.x, a.y, a.z]
            self._try_flush_sample(ts_ns)
        except Exception:
            pass

    def _cb_cgyr(self, ctx: c_void_p, data: c_void_p):
        try:
            g = parse_value(data)
            ts_ns = int(data.contents.epoch * 1e6)
            p = self._partial.setdefault(ts_ns, {'acc': None, 'gyr': None})
            p['gyr'] = [g.x, g.y, g.z]
            self._try_flush_sample(ts_ns)
        except Exception:
            pass

    def _cb_quat(self, ctx: c_void_p, data: c_void_p):
        # Subsampling 1 out of every 2 samples
        self._quat_toggle = not self._quat_toggle
        if not self._quat_toggle:
            return

        q = parse_value(data)
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.sensor_id

        msg.orientation.w = float(q.w)
        msg.orientation.x = float(q.x)
        msg.orientation.y = float(q.y)
        msg.orientation.z = float(q.z)
        msg.orientation_covariance[0] = -1.0

        msg.angular_velocity_covariance[0] = -1.0
        msg.linear_acceleration_covariance[0] = -1.0

        self.pub_quat.publish(msg)

    def _try_flush_sample(self, ts_ns: int):
        pack = self._partial.get(ts_ns)
        if pack and pack['acc'] is not None and pack['gyr'] is not None:
            sample = pack['acc'] + pack['gyr']     # [ax,ay,az,gx,gy,gz]
            self.buffer.append((ts_ns, sample))
            self._partial.pop(ts_ns, None) # Cleaning that timestamp
        
        # Cleans very old entrances (>50ms)
        old_keys = [k for k in list(self._partial.keys()) if ts_ns - k > int(5e7)]
        for k in old_keys:
            self._partial.pop(k, None)

    # Loop: window → inference → publish
    def run(self):
        rospy.loginfo(f"[{self.sensor_id}] Ejecutando… Ctrl+C para salir.")
        rate = rospy.Rate(TARGET_HZ)
        try:
            while not rospy.is_shutdown():
                if not self.started:
                    rate.sleep()
                    continue

                # If RUN is active, we do not open CSV unless the sensor is already connected
                if self.started and (not self.logging_active) and self.sensor_ready:
                    self._open_csv_for_run()
                    self.logging_active = True

                t0 = time.time()
                if len(self.buffer) >= self.window_size and self.csv_w is not None:
                    window = list(self.buffer)[-self.window_size:]            # [(ts, [6]), ...] × 50
                    t0_ns = window[-1][0]
                    X = np.array([s for (_, s) in window], dtype=np.float32)  # (50, 6)

                    # Scaling (6 columns) [ax,ay,az,gx,gy,gz])
                    X_scaled = self.scaler.transform(X).astype(np.float32, copy=False)
                    if not X_scaled.flags['C_CONTIGUOUS']:
                        X_scaled = np.ascontiguousarray(X_scaled)

                    # Robust tensor creation
                    try:
                        X_t = torch.from_numpy(X_scaled).unsqueeze(0)         # (1, 50, 6)
                    except Exception:
                        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

                    X_t = X_t.to(self.device, non_blocking=True)

                    with torch.no_grad():
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        logits = self.model(X_t)
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()
                    
                    # Log (1 row by inference)
                    if self.log_enabled:
                        self._log_window_and_probs(self.seq, t0_ns, probs, X, X_scaled)
                    
                    self._publish_probs(t0_ns, probs)
                dt = time.time() - t0
                sleep_left = max(0.0, self.period - dt)
                if sleep_left > 0:
                    time.sleep(sleep_left)
                rate.sleep()
        except KeyboardInterrupt:
            pass

    def _publish_probs(self, t0_ns: int, probs_np: np.ndarray):
        msg = Probs()
        msg.header	 = Header()
        msg.header.stamp= rospy.Time.now()
        msg.sensor_id   = self.sensor_id
        msg.seq         = self.seq
        msg.t0_ns       = int(t0_ns)
        msg.t_send_ns   = int(time.time_ns())
        msg.K           = int(self.K)
        msg.probs       = [float(x) for x in probs_np]
        msg.dropped_pct = 0.0
        self.pub.publish(msg)
        self.seq += 1

# ------------------ Main ------------------
if __name__ == "__main__":
    rospy.init_node("left_knee_node", anonymous=False)
    node = LKneeNode()
    try:
        node.connect_and_config()
        node.run()
    finally:
        node.disconnect()
