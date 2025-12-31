#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import message_filters

from sensor_msgs.msg import Imu
from har_msgs.msg import Probs
from std_msgs.msg import Float32MultiArray, String

# ===================== TOPICS (igual que tus scripts) =====================
PUB_SKELETON_QUATS = "/har/skeleton_quats"
PUB_GLOBAL_UI      = "/har/probs/global_ui"
SUB_GLOBAL_PROBS   = "/har/probs/global"

SUB_IMU_CHEST      = "/har/imu_chest"
SUB_IMU_LEFT_HAND  = "/har/imu_left_hand"
SUB_IMU_RIGHT_HAND = "/har/imu_right_hand"
SUB_IMU_LEFT_KNEE  = "/har/imu_left_knee"
SUB_IMU_RIGHT_KNEE = "/har/imu_right_knee"

# ===================== Publishers globales =====================
pub_quats = None
pub_probs_ui = None
pub_labels = None  # opcional


# ===================== 1) Aggregator de quaternions =====================
def imu_sync_cb(imu_chest, imu_left_hand, imu_right_hand, imu_left_knee, imu_right_knee):
    # Extrae quaternion de cada IMU como (w, x, y, z)
    def q(msg):
        return [msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z]

    # Vector plano: [w1,x1,y1,z1, w2,x2,y2,z2, ... w5,x5,y5,z5]
    data = (
        q(imu_chest)
        + q(imu_left_hand)
        + q(imu_right_hand)
        + q(imu_left_knee)
        + q(imu_right_knee)
    )

    out = Float32MultiArray()
    out.data = data
    pub_quats.publish(out)


# ===================== 2) Relay Probs -> Float32MultiArray =====================
def probs_cb(msg: Probs):
    out = Float32MultiArray(data=list(msg.probs))
    pub_probs_ui.publish(out)

    # Opcional: relay de labels como CSV
    if pub_labels is not None and getattr(msg, "labels", None):
        if len(msg.labels) > 0:
            pub_labels.publish(String(data=",".join(msg.labels)))


def main():
    global pub_quats, pub_probs_ui, pub_labels

    rospy.init_node("har_aggregator_and_relay")

    # ---- Publishers ----
    pub_quats = rospy.Publisher(PUB_SKELETON_QUATS, Float32MultiArray, queue_size=50)
    pub_probs_ui = rospy.Publisher(PUB_GLOBAL_UI, Float32MultiArray, queue_size=1)

    # Descomenta si quieres etiquetas:
    # pub_labels = rospy.Publisher("/har/probs/labels", String, queue_size=1)

    # ---- Subscribers IMU + sincronizador ----
    sub1 = message_filters.Subscriber(SUB_IMU_CHEST, Imu)
    sub2 = message_filters.Subscriber(SUB_IMU_LEFT_HAND, Imu)
    sub3 = message_filters.Subscriber(SUB_IMU_RIGHT_HAND, Imu)
    sub4 = message_filters.Subscriber(SUB_IMU_LEFT_KNEE, Imu)
    sub5 = message_filters.Subscriber(SUB_IMU_RIGHT_KNEE, Imu)

    sync = message_filters.ApproximateTimeSynchronizer(
        [sub1, sub2, sub3, sub4, sub5],
        queue_size=50,
        slop=0.02,
        allow_headerless=False
    )
    sync.registerCallback(imu_sync_cb)

    # ---- Subscriber Probs (relay) ----
    rospy.Subscriber(SUB_GLOBAL_PROBS, Probs, probs_cb, queue_size=1)

    rospy.loginfo("Nodo combinado listo:")
    rospy.loginfo("  [QUATS] %s + (5 IMUs) -> %s", SUB_IMU_CHEST, PUB_SKELETON_QUATS)
    rospy.loginfo("  [RELAY] %s -> %s (Float32MultiArray)", SUB_GLOBAL_PROBS, PUB_GLOBAL_UI)
    if pub_labels is not None:
        rospy.loginfo("  [RELAY] %s -> /har/probs/labels (String CSV)", SUB_GLOBAL_PROBS)

    rospy.spin()


if __name__ == "__main__":
    main()
