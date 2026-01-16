#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Bool

NODE_NAME = "experiment_start_latch_relay"

IN_TOPIC  = "/har/start_ui"   # Unity
OUT_TOPIC = "/har/start"      # Latched, consumido por nodos

class ExperimentStateRelay:
    def __init__(self):
        rospy.loginfo("[%s] Relay %s -> %s (latched Bool)", NODE_NAME, IN_TOPIC, OUT_TOPIC)

        self.pub = rospy.Publisher(OUT_TOPIC, Bool, queue_size=1, latch=True)
        self.sub = rospy.Subscriber(IN_TOPIC, Bool, self._cb, queue_size=1)

        self.last = None

    def _cb(self, msg: Bool):
        self.last = bool(msg.data)
        self.pub.publish(Bool(data=self.last))
        rospy.logwarn("[%s] State latched -> %s", NODE_NAME, "RUN" if self.last else "STOP")

def main():
    rospy.init_node(NODE_NAME, anonymous=False)
    _ = ExperimentStateRelay()
    rospy.spin()

if __name__ == "__main__":
    main()
