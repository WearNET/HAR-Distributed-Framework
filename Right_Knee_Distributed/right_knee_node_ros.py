# right_knee_node_ros.py — Local node (r_knee) con QUATERNION → probs → ROS1 and GUI

import os
import sys
import time
from collections import deque
from ctypes import c_void_p

import numpy as np
import torch
import torch.nn as nn
import joblib

# ROS
import rospy
from har_msgs.msg import Probs  # generado en har_msgs/msg/Probs.msg
from std_msgs.msg import Header
from sensor_msgs.msg import Imu 

# ===================== PARÁMETROS (EDITA AQUÍ) =====================
SENSOR_ID   = "r_knee"
MAC_ADDR    = "F9:8C:1E:4A:F5:D0"                       # MAC real del MetaMotionRL
MODEL_PATH  = "./cnn_lstm_fold1.pth"                    # Modelo entrenado (input_dim=4 para quat)
SCALER_PATH = "./scaler_model_q_right_knee.pkl"         # Scaler joblib.dump() de 4 columnas [w,x,y,z]
K_CLASSES   = 6                                         # Número de clases
WINDOW_SIZE = 50                                        # 50 muestras (~1s a 50Hz)
TARGET_HZ   = 50                                        # Frecuencia de inferencia/publicación
CONNECT_RETRIES = 8
TOPIC_PROBS = "har/probs/right_knee"                     # Topic ROS para publicar
TOPIC_QUAT  = "har/imu_right_knee"                       # Topic ROS para publicar quaternion
# ===================================================================

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

# ------------------ Modelo local (CNN+LSTM) ------------------
class CNN_LSTM_Sensor(nn.Module):
    def __init__(self, input_dim=4, cnn_out_channels=16, lstm_hidden=32, lstm_layers=1, output_dim=6):
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
        x = lstm_out[:, -1, :]   # último tiempo
        return self.fc(x)

# ------------------ Nodo principal ------------------
class RKneeNode:
    def __init__(self):
        self.sensor_id = SENSOR_ID
        self.mac = MAC_ADDR
        self.K = K_CLASSES
        self.window_size = WINDOW_SIZE
        self.period = 1.0 / float(TARGET_HZ)

        # Buffer ventana (4 features: quaternion [w, x, y, z])
        self.buffer = deque(maxlen=self.window_size)

        # ROS Publisher
        self.pub = rospy.Publisher(TOPIC_PROBS, Probs, queue_size=10)
        self.pub_quat = rospy.Publisher(TOPIC_QUAT, Imu, queue_size=10)
        self.seq = 0

        # Modelo
        self.model = CNN_LSTM_Sensor(input_dim=4, output_dim=self.K)
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # Shim NumPy 2.x -> 1.x
        import types
        if not hasattr(np, "_core") and hasattr(np, "core"):
            mod = types.ModuleType("numpy._core")
            sys.modules["numpy._core"] = mod
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
            try:
                sys.modules["numpy._core.umath"] = np.core.umath
            except AttributeError:
                pass

        # Scaler (joblib) — 4 columnas [w, x, y, z]
        rospy.loginfo(f"[{self.sensor_id}] Cargando scaler desde: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)

        # MetaWear
        self.dev = MetaWear(self.mac)
        self.sig_quat = None
        self.cb_quat = None

        # Downsampling 100Hz → 50Hz
        self.last_keep_ns = None
        self.min_delta_ns = int(1e9 / 50)  # 20 ms

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

        # BLE params
        libmetawear.mbl_mw_settings_set_connection_parameters(self.dev.board, 7.5, 7.5, 0, 6000)

        # Sensor Fusion → QUATERNION
        libmetawear.mbl_mw_sensor_fusion_set_mode(self.dev.board, SensorFusionMode.NDOF)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(self.dev.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(self.dev.board, SensorFusionGyroRange._1000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.dev.board)

        self.sig_quat = libmetawear.mbl_mw_sensor_fusion_get_data_signal(self.dev.board, SensorFusionData.QUATERNION)
        if not self.sig_quat:
            raise RuntimeError("No se pudo obtener la señal de QUATERNION.")

        self.cb_quat = FnVoid_VoidP_Data(self._cb_quat)
        libmetawear.mbl_mw_datasignal_subscribe(self.sig_quat, None, self.cb_quat)

        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.QUATERNION)
        libmetawear.mbl_mw_sensor_fusion_start(self.dev.board)
        rospy.loginfo(f"[{self.sensor_id}] Sensor Fusion (QUATERNION) iniciado.")

    def disconnect(self):
        try:
            libmetawear.mbl_mw_sensor_fusion_stop(self.dev.board)
            if self.sig_quat:
                libmetawear.mbl_mw_datasignal_unsubscribe(self.sig_quat)
        finally:
            self.dev.disconnect()
            rospy.loginfo(f"[{self.sensor_id}] Desconectado.")

    # ---------- Callback QUATERNION (100Hz) con downsampling a 50Hz ----------
    def _cb_quat(self, ctx: c_void_p, data: c_void_p):
        try:
            q = parse_value(data)  # Quaternion: w, x, y, z
            ts_ns = int(data.contents.epoch * 1e6)  # ms → ns

            # Downsample a 50Hz
            if (self.last_keep_ns is None) or (ts_ns - self.last_keep_ns >= self.min_delta_ns):
                sample = [q.w, q.x, q.y, q.z]  # 4 features
                self.buffer.append((ts_ns, sample))
                self.last_keep_ns = ts_ns

                msg_imu = Imu()
                msg_imu.header = Header()
                msg_imu.header.stamp = rospy.Time.now()
                msg_imu.header.frame_id = self.sensor_id

                msg_imu.orientation.w = float(q.w)
                msg_imu.orientation.x = float(q.x)
                msg_imu.orientation.y = float(q.y)
                msg_imu.orientation.z = float(q.z)
                msg_imu.orientation_covariance[0] = -1.0 
                
                msg_imu.angular_velocity_covariance[0] = -1.0
                msg_imu.linear_acceleration_covariance[0] = -1.0

                self.pub_quat.publish(msg_imu)

        except Exception:
            pass

    # ---------- Loop: ventana → inferencia → publicar ----------
    def run(self):
        rospy.loginfo(f"[{self.sensor_id}] Ejecutando… Ctrl+C para salir.")
        rate = rospy.Rate(TARGET_HZ)
        try:
            while not rospy.is_shutdown():
                t0 = time.time()
                if len(self.buffer) >= self.window_size:
                    window = list(self.buffer)[-self.window_size:]            # [(ts, [4]), ...] × 50
                    t0_ns = window[0][0]
                    X = np.array([s for (_, s) in window], dtype=np.float32)  # (50, 4)

                    # Escalado (4 columnas [w, x, y, z])
                    X_scaled = self.scaler.transform(X).astype(np.float32, copy=False)
                    if not X_scaled.flags['C_CONTIGUOUS']:
                        X_scaled = np.ascontiguousarray(X_scaled)

                    # Tensor robusto
                    try:
                        X_t = torch.from_numpy(X_scaled).unsqueeze(0)         # (1, 50, 4)
                    except Exception:
                        X_t = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)

                    with torch.no_grad():
                        logits = self.model(X_t)
                        probs = torch.softmax(logits, dim=-1).cpu().numpy().ravel()

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

# ------------------ main ------------------
if __name__ == "__main__":
    rospy.init_node("right_knee_node", anonymous=False)
    node = RKneeNode()
    try:
        node.connect_and_config()
        node.run()
    finally:
        node.disconnect()
