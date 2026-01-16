#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import csv
import time
import threading
from datetime import datetime

import numpy as np
import rospy
from std_msgs.msg import Bool, String, Empty
from har_msgs.msg import Probs

# jtop (jetson-stats)
try:
    from jtop import jtop
    JTOP_AVAILABLE = True
except Exception:
    JTOP_AVAILABLE = False


class ExperimentManager:
    def __init__(self):
        # ================== PARAMETROS ROS ==================
        self.start_topic  = rospy.get_param("~start_topic", "/har/start")
        self.global_topic = rospy.get_param("~global_topic", "/har/probs/global")

        self.out_dir      = rospy.get_param("~out_dir", os.path.expanduser("~/har_runs"))
        self.warmup_sec   = float(rospy.get_param("~warmup_sec", 0.0))
        self.enable_power = bool(rospy.get_param("~enable_power", True))

        # Potencia con jtop
        self.power_hz     = float(rospy.get_param("~power_hz", 10.0))  # 10 Hz típico
        self.power_key    = rospy.get_param("~power_key", "")          # si vacío: auto-selección
        # Ejemplos de power_key (depende del Jetson): "POM_5V_IN", "VDD_IN", etc.

        self.LOCAL_TOPICS = {
            "chest": "/har/probs/chest",
            "left_knee": "/har/probs/left_knee",
            "right_hand": "/har/probs/right_hand",
            "left_hand": "/har/probs/left_hand",
            "right_knee": "/har/probs/right_knee",
        }
        self.LOCAL_ORDER = ["chest", "left_knee", "right_hand", "left_hand", "right_knee"]

        self._local_lock = threading.Lock()
        self.local_lat_ms = {k: None for k in self.LOCAL_ORDER}   # última latencia local por sensor
        self.local_seq    = {k: None for k in self.LOCAL_ORDER}   # opcional: último seq visto

        # ================== ESTADO ==================
        self.running = False
        self.run_id = None
        self.run_path = None
        self.t_run_start = None
        self._warmup_done = (self.warmup_sec <= 0.0)

        self.infer_rows = 0
        self.lat_rows = 0
        self.power_rows = 0

        self.fp_infer = None
        self.fp_lat = None
        self.fp_power = None

        self.csv_infer = None
        self.csv_lat = None
        self.csv_power = None

        # Power sampling thread
        self._power_stop_evt = threading.Event()
        self._power_thread = None

        # ================== PUBLISHERS ==================
        self.pub_status = rospy.Publisher("/har/experiment/status", String, queue_size=1, latch=True)
        self.pub_runid  = rospy.Publisher("/har/experiment/run_id", String, queue_size=1, latch=True)
        self.pub_event  = rospy.Publisher("/har/experiment/event", String, queue_size=10)
        self.pub_marker = rospy.Publisher("/har/experiment/marker", Empty, queue_size=10)

        # ================== SUBSCRIBERS ==================
        self.sub_start  = rospy.Subscriber(self.start_topic, Bool, self._start_cb, queue_size=1)
        self.sub_global = rospy.Subscriber(self.global_topic, Probs, self._global_cb, queue_size=50)

        self.sub_locals = []
        for sid, topic in self.LOCAL_TOPICS.items():
            self.sub_locals.append(
                rospy.Subscriber(topic, Probs, self._local_cb, callback_args=sid, queue_size=50)
            )

        self._set_status("IDLE")
        rospy.on_shutdown(self._on_shutdown)

        if self.enable_power and not JTOP_AVAILABLE:
            rospy.logwarn("[experiment_manager] jetson-stats/jtop no disponible. Instala: sudo pip3 install jetson-stats")

    # ---------- helpers ----------
    def _set_status(self, s: str):
        self.pub_status.publish(String(s))

    def _emit_event(self, s: str):
        self.pub_event.publish(String(s))

    def _new_run_id(self):
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{ts}_run"

    # ---------- start/stop ----------
    def _start_cb(self, msg: Bool):
        if msg.data and not self.running:
            self._start_run()
        elif (not msg.data) and self.running:
            self._stop_run()

    def _start_run(self):
        self.running = True
        self.run_id = self._new_run_id()
        self.run_path = os.path.join(self.out_dir, self.run_id)
        os.makedirs(self.run_path, exist_ok=True)

        self.t_run_start = rospy.Time.now()
        self._warmup_done = (self.warmup_sec <= 0.0)

        infer_path = os.path.join(self.run_path, "inference.csv")
        lat_path   = os.path.join(self.run_path, "latency.csv")
        power_path = os.path.join(self.run_path, "power.csv")

        self.fp_infer = open(infer_path, "w", newline="")
        self.fp_lat   = open(lat_path, "w", newline="")
        self.fp_power = open(power_path, "w", newline="")

        self.csv_infer = csv.writer(self.fp_infer)
        self.csv_lat   = csv.writer(self.fp_lat)
        self.csv_power = csv.writer(self.fp_power)

        # -------- headers --------
        self.csv_infer.writerow([
            "run_id",
            "t_session_sec",
            "t_rx_epoch_sec",
            "sensor_id",
            "seq",
            "K",
            "pred_label",
            "pred_conf",
            "dropped_pct",
            "probs"
        ])

        self.csv_lat.writerow([
            "run_id",
            "t_session_sec",
            "t_rx_ns",
            "t0_ns",
            "t_last_local_send_ns",
            "t_send_ns_global",
            "L_local_ms",
            "L_central_ms",
            "L_net_global_ms",
            "L1_e2e_ms",
            # ---- locales por sensor (último valor disponible) ----
            "L_local_chest_ms",
            "L_local_left_knee_ms",
            "L_local_right_hand_ms",
            "L_local_left_hand_ms",
            "L_local_right_knee_ms",
            "L_local_mean_ms",
            "L_local_max_ms"
        ])

        self.csv_power.writerow([
            "run_id",
            "t_session_sec",
            "t_epoch_sec",
            "power_tot_w",
            "power_tot_avg_w",
            "power_sys5v_w",
            "raw_power_dict"
        ])

        self.infer_rows = 0
        self.lat_rows = 0
        self.power_rows = 0

        # status/run_id/event
        self.pub_runid.publish(String(self.run_id))
        self._set_status("RUNNING")
        self._emit_event("START")
        self.pub_marker.publish(Empty())

        # power logging (thread)
        if self.enable_power and JTOP_AVAILABLE:
            self._start_power_thread()

        rospy.loginfo(f"[experiment_manager] START run_id={self.run_id} path={self.run_path}")

    def _stop_run(self):
        self._set_status("SAVING")
        self._emit_event("STOP")
        self.pub_marker.publish(Empty())

        if self.enable_power and JTOP_AVAILABLE:
            self._stop_power_thread()

        # cerrar CSVs
        try:
            for fp in [self.fp_infer, self.fp_lat, self.fp_power]:
                if fp:
                    fp.flush()
                    fp.close()
        finally:
            self.fp_infer = None
            self.fp_lat = None
            self.fp_power = None

        self._write_summary()

        self.running = False
        self._set_status("STOPPED")
        rospy.loginfo(f"[experiment_manager] STOP run_id={self.run_id}")

    # ---------- global callback ----------
    def _global_cb(self, msg: Probs):
        if not self.running:
            return

        t_rx_ros = rospy.Time.now()
        t_session = (t_rx_ros - self.t_run_start).to_sec()

        # warmup gating
        if (not self._warmup_done) and (t_session < self.warmup_sec):
            return
        if (not self._warmup_done) and (t_session >= self.warmup_sec):
            self._warmup_done = True
            self._emit_event("WARMUP_DONE")

        # --- inferencia (argmax) ---
        probs_list = list(msg.probs) if hasattr(msg, "probs") else []
        K = int(msg.K) if hasattr(msg, "K") and msg.K > 0 else len(probs_list)

        pred_label = ""
        pred_conf = ""
        if len(probs_list) > 0:
            pred_idx = int(np.argmax(probs_list))
            pred_label = str(pred_idx)
            pred_conf = f"{float(probs_list[pred_idx]):.6f}"

        sensor_id = getattr(msg, "sensor_id", "")
        seq = int(getattr(msg, "seq", 0))
        dropped_pct = float(getattr(msg, "dropped_pct", 0.0))
        
        self.csv_infer.writerow([
            self.run_id,
            f"{t_session:.6f}",
            f"{time.time():.6f}",          # epoch sec en el manager
            sensor_id,
            seq,
            K,
            pred_label,
            pred_conf,
            f"{dropped_pct:.3f}",
            "[" + ",".join(f"{p:.6f}" for p in probs_list) + "]"
        ])
        self.infer_rows += 1

        # --- latencia extendida basada en t0_ns / t_last_local_send_ns / t_send_ns ---
        t_rx_ns = time.time_ns()
        t0_ns = int(getattr(msg, "t0_ns", 0))
        t_last_local_send_ns = int(getattr(msg, "t_last_local_send_ns", 0))
        t_send_ns_global = int(getattr(msg, "t_send_ns", 0))

        L_local_ms = ""
        L_central_ms = ""
        L_net_global_ms = ""
        L1_ms = ""

        if t0_ns > 0 and t_last_local_send_ns > 0:
            L_local_ms = f"{(t_last_local_send_ns - t0_ns) / 1e6:.3f}"

        if t_last_local_send_ns > 0 and t_send_ns_global > 0:
            L_central_ms = f"{(t_send_ns_global - t_last_local_send_ns) / 1e6:.3f}"

        if t_send_ns_global > 0:
            L_net_global_ms = f"{(t_rx_ns - t_send_ns_global) / 1e6:.3f}"

        if t0_ns > 0:
            L1_ms = f"{(t_rx_ns - t0_ns) / 1e6:.3f}"

        # --- snapshot de latencias locales (último valor por sensor) ---
        with self._local_lock:
            L_chest = self.local_lat_ms["chest"]
            L_lknee = self.local_lat_ms["left_knee"]
            L_rhand = self.local_lat_ms["right_hand"]
            L_lhand = self.local_lat_ms["left_hand"]
            L_rknee = self.local_lat_ms["right_knee"]

        vals = [v for v in [L_chest, L_lknee, L_rhand, L_lhand, L_rknee] if v is not None]
        L_mean = (sum(vals) / len(vals)) if len(vals) > 0 else None
        L_max  = max(vals) if len(vals) > 0 else None

        self.csv_lat.writerow([
            self.run_id,
            f"{t_session:.6f}",
            str(t_rx_ns),
            str(t0_ns) if t0_ns > 0 else "",
            str(t_last_local_send_ns) if t_last_local_send_ns > 0 else "",
            str(t_send_ns_global) if t_send_ns_global > 0 else "",
            L_local_ms,
            L_central_ms,
            L_net_global_ms,
            L1_ms,
            "" if L_chest is None else f"{L_chest:.3f}",
            "" if L_lknee is None else f"{L_lknee:.3f}",
            "" if L_rhand is None else f"{L_rhand:.3f}",
            "" if L_lhand is None else f"{L_lhand:.3f}",
            "" if L_rknee is None else f"{L_rknee:.3f}",
            "" if L_mean is None else f"{L_mean:.3f}",
            "" if L_max is None else f"{L_max:.3f}",
        ])
        self.lat_rows += 1

    def _local_cb(self, msg: Probs, sensor_key: str):
        # Calcula latencia local "interna" usando timestamps del propio mensaje
        try:
            t0_ns = int(getattr(msg, "t0_ns", 0))
            t_send_ns = int(getattr(msg, "t_send_ns", 0))
            if t0_ns > 0 and t_send_ns > 0 and t_send_ns >= t0_ns:
                lat_ms = (t_send_ns - t0_ns) / 1e6
            else:
                lat_ms = None
        except Exception:
            lat_ms = None

        with self._local_lock:
            self.local_lat_ms[sensor_key] = lat_ms
            try:
                self.local_seq[sensor_key] = int(getattr(msg, "seq", 0))
            except Exception:
                self.local_seq[sensor_key] = None

    # ---------- power sampling (jtop) ----------
    def _start_power_thread(self):
        self._power_stop_evt.clear()
        self._power_thread = threading.Thread(target=self._power_loop, daemon=True)
        self._power_thread.start()
        rospy.loginfo("[experiment_manager] Power sampling (jtop) iniciado.")

    def _stop_power_thread(self):
        self._power_stop_evt.set()
        if self._power_thread:
            self._power_thread.join(timeout=2.0)
        self._power_thread = None
        rospy.loginfo("[experiment_manager] Power sampling (jtop) detenido.")

    def _select_power_rail(self, power_dict: dict) -> str:
        """
        Selección automática:
        - si power_key está configurado y existe, úsalo
        - si no, intenta elegir el rail más representativo (ej. POM_5V_IN / VDD_IN / total)
        """
        if not isinstance(power_dict, dict) or len(power_dict) == 0:
            return ""

        if self.power_key and self.power_key in power_dict:
            return self.power_key

        # candidatos comunes
        for k in ["POM_5V_IN", "VDD_IN", "SYS5V", "SYS", "TOTAL", "VIN_SYS_5V0"]:
            if k in power_dict:
                return k

        # fallback: primer key
        return next(iter(power_dict.keys()))

    def _power_loop(self):
        # Frecuencia
        dt = 1.0 / max(self.power_hz, 0.1)

        def mw_to_w(x):
            try:
                return float(x) / 1000.0
            except Exception:
                return None

        try:
            with jtop() as jetson:
                if not jetson.ok():
                    rospy.logwarn("[experiment_manager] jtop no está OK al iniciar.")

                while jetson.ok() and (not self._power_stop_evt.is_set()):
                    # epoch y sesión
                    t_epoch = time.time()
                    t_session = (rospy.Time.now() - self.t_run_start).to_sec() if self.t_run_start else 0.0

                    # Leer diccionario de potencia
                    try:
                        p_dict = dict(getattr(jetson, "power", {}) or {})
                    except Exception:
                        p_dict = {}

                    # ---------------- Potencia total ----------------
                    p_tot_w = None
                    p_tot_avg_w = None

                    if isinstance(p_dict, dict) and "tot" in p_dict and isinstance(p_dict["tot"], dict):
                        p_tot_w = mw_to_w(p_dict["tot"].get("power", None))
                        p_tot_avg_w = mw_to_w(p_dict["tot"].get("avg", None))

                    # ---------------- Rail opcional: SYS5V ----------------
                    p_sys5v_w = None
                    if isinstance(p_dict, dict) and "rail" in p_dict and isinstance(p_dict["rail"], dict):
                        sys5v = p_dict["rail"].get("SYS5V", None)
                        if isinstance(sys5v, dict):
                            p_sys5v_w = mw_to_w(sys5v.get("power", None))

                    # Si quieres otro rail (CPU/SOC/GPU), puedes copiar el bloque SYS5V y cambiar la llave.

                    # Escribir CSV
                    self.csv_power.writerow([
                        self.run_id,
                        f"{t_session:.6f}",
                        f"{t_epoch:.6f}",
                        "" if p_tot_w is None else f"{p_tot_w:.4f}",
                        "" if p_tot_avg_w is None else f"{p_tot_avg_w:.4f}",
                        "" if p_sys5v_w is None else f"{p_sys5v_w:.4f}",
                        json.dumps(p_dict)
                    ])
                    self.power_rows += 1

                    time.sleep(dt)

        except Exception as e:
            rospy.logwarn(f"[experiment_manager] Power loop error (jtop): {e}")

    # ---------- summary ----------
    def _write_summary(self):
        summary_path = os.path.join(self.run_path, "summary.json")

        data = {
            "run_id": self.run_id,
            "start_time_ros_sec": self.t_run_start.to_sec() if self.t_run_start else None,
            "rows_inference": self.infer_rows,
            "rows_latency": self.lat_rows,
            "rows_power": self.power_rows,
            "power_backend": "jtop" if (self.enable_power and JTOP_AVAILABLE) else "disabled_or_unavailable",
            "notes": [
                "pred_label = argmax(probs)",
                "latency uses t0_ns and t_send_ns from har_msgs/Probs",
                "power_w is derived from jtop.power rail selection; verify rail_name for your Jetson"
            ]
        }

        with open(summary_path, "w") as f:
            json.dump(data, f, indent=2)

        rospy.loginfo(f"[experiment_manager] summary written: {summary_path}")

    def _on_shutdown(self):
        if self.running:
            try:
                self._stop_run()
            except Exception as e:
                rospy.logwarn(f"[experiment_manager] shutdown stop_run error: {e}")


def main():
    rospy.init_node("experiment_manager", anonymous=False)
    ExperimentManager()
    rospy.spin()


if __name__ == "__main__":
    main()
