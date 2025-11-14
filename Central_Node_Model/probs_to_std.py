#!/usr/bin/env python3
import rospy
from har_msgs.msg import Probs
from std_msgs.msg import Float32MultiArray, String

pub = None
pub_labels = None

def cb(msg):
    out = Float32MultiArray(data=msg.probs)
    pub.publish(out)
    # Si quieres enviar etiquetas como CSV (opcional):
    if pub_labels and msg.labels:
        pub_labels.publish(String(data=",".join(msg.labels)))

if __name__ == "__main__":
    rospy.init_node("probs_to_std_ui")
    pub = rospy.Publisher("/har/probs/global_ui", Float32MultiArray, queue_size=1)
    # Descomenta si quieres etiquetas:
    # pub_labels = rospy.Publisher("/har/probs/labels", String, queue_size=1)
    rospy.Subscriber("/har/probs/global", Probs, cb, queue_size=1)
    rospy.loginfo("Relay listo: /har/probs/global -> /har/probs/global_ui (Float32MultiArray)")
    rospy.spin()
