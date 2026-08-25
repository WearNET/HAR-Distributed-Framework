#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import rospy
import message_filters
import numpy as np

from sensor_msgs.msg import Imu
from har_msgs.msg import Probs
from std_msgs.msg import Float32MultiArray, String

# ===================== TOPICS =====================
PUB_SKELETON_QUATS = "/har/skeleton_quats"
PUB_GLOBAL_UI      = "/har/probs/global_ui"
SUB_GLOBAL_PROBS   = "/har/probs/global"

TOPIC_BY_SENSOR = {
    "chest":  "/har/imu_chest",
    "lhand":  "/har/imu_left_hand",
    "rhand":  "/har/imu_right_hand",
    "lknee":  "/har/imu_left_knee",
    "rknee":  "/har/imu_right_knee",
}
SENSOR_ORDER = ["chest", "lhand", "rhand", "lknee", "rknee"]

# ===================== SYNC (ATS) params =====================
ATS_SLOP_SEC = 0.02
ATS_QUEUE_SIZE = 50

# ===================== ZOH/Failover params =====================
PUBLISH_RATE_HZ      = 50.0
FRESH_MAX_AGE_SEC    = 0.20   # solo diagnóstico
ZOH_MAX_HOLD_SEC     = 2.0    # si supera -> identity quat
FAILOVER_SEC         = 0.35
RECOVER_STABLE_SEC   = 1.00
STARTUP_GRACE_SEC    = 2.00

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

class HarAggregatorAndRelay(object):
    def __init__(self):
        rospy.init_node("har_aggregator_and_relay", anonymous=False)

        # ---- Publishers ----
        self.pub_quats = rospy.Publisher(PUB_SKELETON_QUATS, Float32MultiArray, queue_size=50)
        self.pub_probs_ui = rospy.Publisher(PUB_GLOBAL_UI, Float32MultiArray, queue_size=1)
        self.pub_labels = None  # opcional: String CSV

        # ---- State ----
        self._lock = threading.Lock()

        # Cache por sensor (para ZOH)
        self.last_quat = {s: None for s in SENSOR_ORDER}   # np.array (4,)
        self.last_rx_t = {s: None for s in SENSOR_ORDER}   # float seconds

        # Modo
        self.mode = "SYNC"
        self.t0 = rospy.Time.now().to_sec()
        self.last_sync_cb_t = None
        self.good_since = None

        # ---- Subscribers cache (siempre activos) ----
        self.sub_cache = []
        for s in SENSOR_ORDER:
            self.sub_cache.append(
                rospy.Subscriber(TOPIC_BY_SENSOR[s], Imu, self._cb_cache, callback_args=s, queue_size=50)
            )

        # ---- ATS subscribers (SYNC ideal) ----
        sub_list = [message_filters.Subscriber(TOPIC_BY_SENSOR[s], Imu) for s in SENSOR_ORDER]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            sub_list,
            queue_size=ATS_QUEUE_SIZE,
            slop=ATS_SLOP_SEC,
            allow_headerless=False
        )
        self.ts.registerCallback(self._cb_sync)

        # ---- Relay Probs -> Float32MultiArray ----
        rospy.Subscriber(SUB_GLOBAL_PROBS, Probs, self._cb_probs, queue_size=1)

        # ---- Timer watchdog + ZOH publish ----
        period = 1.0 / float(PUBLISH_RATE_HZ)
        self.timer = rospy.Timer(rospy.Duration(period), self._timer_cb)

        rospy.loginfo("Nodo combinado con ATS(SYNC) + ZOH iniciado:")
        rospy.loginfo("  [QUATS] (5 IMUs) -> %s", PUB_SKELETON_QUATS)
        rospy.loginfo("  [RELAY] %s -> %s (Float32MultiArray)", SUB_GLOBAL_PROBS, PUB_GLOBAL_UI)

    # ---------------- Cache per-sensor (ZOH) ----------------
    def _cb_cache(self, msg: Imu, sensor_id: str):
        q = np.array([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z], dtype=np.float32)
        now = rospy.Time.now().to_sec()
        with self._lock:
            self.last_quat[sensor_id] = q
            self.last_rx_t[sensor_id] = float(now)

    # ---------------- ATS callback (SYNC) ----------------
    def _cb_sync(self, imu_chest, imu_lhand, imu_rhand, imu_lknee, imu_rknee):
        now = rospy.Time.now().to_sec()

        with self._lock:
            self.last_sync_cb_t = float(now)

            # Recover logic
            if self.mode == "ZOH":
                if self.good_since is None:
                    self.good_since = float(now)
                if (now - self.good_since) >= RECOVER_STABLE_SEC:
                    self.mode = "SYNC"
                    self.good_since = None
                    rospy.loginfo("[agg] RECOVER -> SYNC (ATS estable por %.3fs)", RECOVER_STABLE_SEC)
            else:
                self.good_since = None

            # Evitar doble fuente
            if self.mode != "SYNC":
                return

        def q(m: Imu):
            return [m.orientation.w, m.orientation.x, m.orientation.y, m.orientation.z]

        data = (
            q(imu_chest) +
            q(imu_lhand) +
            q(imu_rhand) +
            q(imu_lknee) +
            q(imu_rknee)
        )

        out = Float32MultiArray()
        out.data = data
        self.pub_quats.publish(out)

    # ---------------- Relay probs to UI ----------------
    def _cb_probs(self, msg: Probs):
        self.pub_probs_ui.publish(Float32MultiArray(data=list(msg.probs)))

        if self.pub_labels is not None and getattr(msg, "labels", None):
            if len(msg.labels) > 0:
                self.pub_labels.publish(String(data=",".join(msg.labels)))

    # ---------------- ZOH build ----------------
    def _build_quats_zoh(self, now_sec: float):
        vecs = []
        statuses = {}
        ages = {}

        for s in SENSOR_ORDER:
            q = self.last_quat[s]
            t = self.last_rx_t[s]

            if q is None or t is None:
                vecs.append(IDENTITY_QUAT)
                statuses[s] = "identity_never"
                ages[s] = None
                continue

            age = float(now_sec - t)
            ages[s] = age

            if age <= ZOH_MAX_HOLD_SEC:
                vecs.append(q)
                statuses[s] = "zoh" if age > FRESH_MAX_AGE_SEC else "fresh"
            else:
                vecs.append(IDENTITY_QUAT)
                statuses[s] = "identity_hold_exceeded"

        flat = np.concatenate(vecs, axis=0).astype(np.float32).tolist()  # 5*4 = 20
        return flat, statuses, ages

    # ---------------- Timer watchdog + ZOH publish ----------------
    def _timer_cb(self, _evt):
        now = rospy.Time.now().to_sec()

        with self._lock:
            # Failover decision
            if self.mode == "SYNC":
                if self.last_sync_cb_t is None:
                    if (now - self.t0) > STARTUP_GRACE_SEC:
                        self.mode = "ZOH"
                        self.good_since = None
                        rospy.logwarn("[agg] FAILOVER -> ZOH (ATS no disparó durante startup)")
                else:
                    dt = now - self.last_sync_cb_t
                    if dt >= FAILOVER_SEC:
                        self.mode = "ZOH"
                        self.good_since = None
                        rospy.logwarn("[agg] FAILOVER -> ZOH (ATS silent por %.3fs)", dt)

            if self.mode != "ZOH":
                return

            # Snapshot cache under lock
            data, statuses, ages = self._build_quats_zoh(now)

        # Publish outside lock
        out = Float32MultiArray()
        out.data = data
        self.pub_quats.publish(out)

        st_csv = ",".join(
            f"{s}:{statuses[s]}:{(-1.0 if ages[s] is None else ages[s]):.3f}"
            for s in SENSOR_ORDER
        )
        rospy.loginfo_throttle(1.0, "[agg][ZOH-DBG] %s", st_csv)

def main():
    _ = HarAggregatorAndRelay()
    rospy.spin()

if __name__ == "__main__":
    main()
