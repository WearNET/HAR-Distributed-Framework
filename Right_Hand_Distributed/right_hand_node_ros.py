# right_hand_node_ros.py — Local node (r_hand) con ACC+GYRO → probs → ROS1
# right_hand_node_ros.py — Local node (r_hand) con QUAT → GUI → ROS

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
SENSOR_ID   = "r_hand"
MAC_ADDR    = "EE:1B:72:FA:BF:E8"                       # MAC real del MetaMotionRL
MODEL_PATH  = "./cnn_lstm_fold1.pth"                    # Modelo entrenado (input_dim=6 para acc+gyro)
SCALER_PATH = "./scaler_model_ag_right_hand.pkl"         # Scaler joblib.dump() de 6 columnas [ax,ay,az,gx,gy,gz]
K_CLASSES   = 6                                         # Número de clases
WINDOW_SIZE = 50                                        # 50 muestras (~1s a 50Hz)
TARGET_HZ   = 50                                        # Frecuencia de inferencia/publicación
CONNECT_RETRIES = 8
TOPIC_PROBS = "har/probs/right_hand"                     # Topic ROS para publicar
TOPIC_QUAT  = "har/imu_right_hand"                       # Topic ROS para publicar quaternion
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
        x = lstm_out[:, -1, :]   # último tiempo
        return self.fc(x)

# ------------------ Nodo principal ------------------
class RHandNode:
    def __init__(self):
        self.sensor_id = SENSOR_ID
        self.mac = MAC_ADDR
        self.K = K_CLASSES
        self.window_size = WINDOW_SIZE
        self.period = 1.0 / float(TARGET_HZ)

        # Buffer ventana (6 features: acc xyz + gyro xyz)
        self.buffer = deque(maxlen=self.window_size)

        # ROS Publisher
        self.pub = rospy.Publisher(TOPIC_PROBS, Probs, queue_size=10)
        self.pub_quat = rospy.Publisher(TOPIC_QUAT, Imu, queue_size=10)
        self.seq = 0

        # Modelo (input_dim=6 para acc+gyro)
        self.model = CNN_LSTM_Sensor(input_dim=6, output_dim=self.K)
        try:
            state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval()

        # Shim NumPy 2.x -> 1.x (por si tu scaler lo necesita)
        import types
        if not hasattr(np, "_core") and hasattr(np, "core"):
            mod = types.ModuleType("numpy._core")
            sys.modules["numpy._core"] = mod
            sys.modules["numpy._core.multiarray"] = np.core.multiarray
            try:
                sys.modules["numpy._core.umath"] = np.core.umath
            except AttributeError:
                pass

        # Scaler (joblib) — 6 columnas [ax, ay, az, gx, gy, gz]
        rospy.loginfo(f"[{self.sensor_id}] Cargando scaler desde: {SCALER_PATH}")
        self.scaler = joblib.load(SCALER_PATH)

        # MetaWear
        self.dev = MetaWear(self.mac)
        self.sig_cacc = None
        self.sig_cgyr = None
        self.cb_cacc = None
        self.cb_cgyr = None
        self.sig_quat = None
        self.cb_quat = None
        self._quat_toggle = False  # para submuestreo 100Hz -> ~50Hz

        # Diccionario temporal para unir acc+gyro por timestamp
        self._partial = {}

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

        # Parámetros BLE (opcional)
        libmetawear.mbl_mw_settings_set_connection_parameters(self.dev.board, 7.5, 7.5, 0, 6000)

        # Sensor Fusion → IMU_PLUS (solo acc+gyro)
        libmetawear.mbl_mw_sensor_fusion_set_mode(self.dev.board, SensorFusionMode.IMU_PLUS)
        libmetawear.mbl_mw_sensor_fusion_set_acc_range(self.dev.board, SensorFusionAccRange._8G)
        libmetawear.mbl_mw_sensor_fusion_set_gyro_range(self.dev.board, SensorFusionGyroRange._1000DPS)
        libmetawear.mbl_mw_sensor_fusion_write_config(self.dev.board)

        # Señales corregidas
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

        # Habilitar y arrancar (dos streams)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.CORRECTED_ACC)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.CORRECTED_GYRO)
        libmetawear.mbl_mw_sensor_fusion_enable_data(self.dev.board, SensorFusionData.QUATERNION)
        libmetawear.mbl_mw_sensor_fusion_start(self.dev.board)
        rospy.loginfo(f"[{self.sensor_id}] Sensor Fusion (ACC+GYRO+QUAT) iniciado a ~50Hz.")

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
            self.dev.disconnect()
            rospy.loginfo(f"[{self.sensor_id}] Desconectado.")

    # ---------- Callbacks ----------
    def _cb_cacc(self, ctx: c_void_p, data: c_void_p):
        try:
            a = parse_value(data)  # CorrectedCartesianFloat: x,y,z
            ts_ns = int(data.contents.epoch * 1e6)  # ms → ns
            pack = self._partial.setdefault(ts_ns, {'acc': None, 'gyr': None})
            pack['acc'] = [a.x, a.y, a.z]
            self._try_flush_sample(ts_ns)
        except Exception:
            pass

    def _cb_cgyr(self, ctx: c_void_p, data: c_void_p):
        try:
            g = parse_value(data)  # CorrectedCartesianFloat: x,y,z (deg/s)
            ts_ns = int(data.contents.epoch * 1e6)  # ms → ns
            pack = self._partial.setdefault(ts_ns, {'acc': None, 'gyr': None})
            pack['gyr'] = [g.x, g.y, g.z]
            self._try_flush_sample(ts_ns)
        except Exception:
            pass

    def _cb_quat(self, ctx: c_void_p, data: c_void_p):
        # Submuestreo 1 de cada 2 muestras
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
            sample = pack['acc'] + pack['gyr']
            self.buffer.append((ts_ns, sample))
            self._partial.pop(ts_ns, None)
        old_keys = [k for k in list(self._partial.keys()) if ts_ns - k > int(5e7)]
        for k in old_keys:
            self._partial.pop(k, None)

    # ---------- Loop: ventana → inferencia → publicar ----------
    def run(self):
        rospy.loginfo(f"[{self.sensor_id}] Ejecutando… Ctrl+C para salir.")
        rate = rospy.Rate(TARGET_HZ)
        try:
            while not rospy.is_shutdown():
                t0 = time.time()
                if len(self.buffer) >= self.window_size:
                    window = list(self.buffer)[-self.window_size:]            # [(ts, [6]), ...] × 50
                    t0_ns = window[0][0]
                    X = np.array([s for (_, s) in window], dtype=np.float32)  # (50, 6)

                    # Escalado (6 columnas [ax,ay,az,gx,gy,gz])
                    X_scaled = self.scaler.transform(X).astype(np.float32, copy=False)
                    if not X_scaled.flags['C_CONTIGUOUS']:
                        X_scaled = np.ascontiguousarray(X_scaled)

                    # Tensor robusto
                    try:
                        X_t = torch.from_numpy(X_scaled).unsqueeze(0)         # (1, 50, 6)
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
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
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
    rospy.init_node("right_hand_node", anonymous=False)
    node = RHandNode()
    try:
        node.connect_and_config()
        node.run()
    finally:
        node.disconnect()
