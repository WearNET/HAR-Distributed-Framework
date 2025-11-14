#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import torch
import torch.nn as nn
import numpy as np
from har_msgs.msg import Probs
from std_msgs.msg import Header
import message_filters

# ================= CONFIGURACIÓN =================
NODE_NAME    = "central_har_node"
MODEL_PATH   = "nn_central_fold1.pth"
PUB_TOPIC    = "/har/probs/global"

TOPIC_CHEST      = "/har/probs/chest"
TOPIC_L_KNEE     = "/har/probs/left_knee"
TOPIC_R_HAND     = "/har/probs/right_hand"
TOPIC_L_HAND     = "/har/probs/left_hand"
TOPIC_R_KNEE     = "/har/probs/right_knee"

SYNC_SLOP_SEC = 0.1
QUEUE_SIZE    = 10

# ============ MODELO CENTRAL =============
class CentralHARNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64):
        super(CentralHARNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def load_model(input_dim, num_classes):
    model = CentralHARNet(input_dim=input_dim, num_classes=num_classes)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        rospy.loginfo("[central] Modelo cargado desde: %s", MODEL_PATH)
    else:
        rospy.logwarn("[central] NO se encontró el modelo en %s, usando pesos aleatorios", MODEL_PATH)
    model.eval()
    return model

class CentralNode(object):
    def __init__(self):
        rospy.loginfo("[central] Iniciando nodo central (sin scaler)...")
        self.pub = rospy.Publisher(PUB_TOPIC, Probs, queue_size=10)

        # Suscriptores con message_filters
        sub_chest  = message_filters.Subscriber(TOPIC_CHEST, Probs)
        sub_lknee  = message_filters.Subscriber(TOPIC_L_KNEE, Probs)
        sub_rhand  = message_filters.Subscriber(TOPIC_R_HAND, Probs)
        sub_lhand  = message_filters.Subscriber(TOPIC_L_HAND, Probs)
        sub_rknee  = message_filters.Subscriber(TOPIC_R_KNEE, Probs)

        # Sincronizador aproximado
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [sub_chest, sub_lhand, sub_rhand, sub_lknee, sub_rknee],
            queue_size=QUEUE_SIZE,
            slop=SYNC_SLOP_SEC,
            allow_headerless=False
        )
        self.ts.registerCallback(self.callback_sync)

        # Descubrir dimensiones en el primer batch
        self.model = None
        self.input_dim = None
        self.num_classes = None

        rospy.loginfo("[central] Esperando probabilidades de los 5 sensores...")

    def callback_sync(self, msg_chest, msg_lhand, msg_rhand, msg_lknee, msg_rknee):
        p_chest = np.array(msg_chest.probs, dtype=np.float32)
        p_lknee = np.array(msg_lknee.probs, dtype=np.float32)
        p_rhand = np.array(msg_rhand.probs, dtype=np.float32)
        p_lhand = np.array(msg_lhand.probs, dtype=np.float32)
        p_rknee = np.array(msg_rknee.probs, dtype=np.float32)

        # Comprobar mismo tamaño K
        K = len(p_chest)
        if not (len(p_lknee) == len(p_rhand) == len(p_lhand) == len(p_rknee) == K):
            rospy.logwarn("[central] Vectores de prob no tienen mismo tamaño, se ignora este batch.")
            return

        # Concatenación en orden fijo
        # [ Chest (AG), Left Knee (AG), Right Hand (AG), Left Hand (Q), Right Knee (Q) ]
        x = np.concatenate([p_chest, p_lknee, p_rhand, p_lhand, p_rknee], axis=0) 

        # Inicializar modelo la primera vez
        if self.model is None:
            self.input_dim = x.shape[0]
            self.num_classes = K
            self.model = load_model(self.input_dim, self.num_classes)

        # A tensor y forward
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x_tensor)
            probs_global = torch.softmax(logits, dim=1).cpu().numpy().flatten()

        # Publicar
        out_msg = Probs()
        out_msg.header = Header()
        out_msg.header.stamp = rospy.Time.now()
        out_msg.sensor_id = "central"
        out_msg.probs = probs_global.astype(np.float32).tolist()
        self.pub.publish(out_msg)

        pred_class = int(np.argmax(probs_global))
        rospy.loginfo_throttle(1.0, "[central] pred=%d, probs=%s",
                               pred_class,
                               np.array2string(probs_global, precision=3, floatmode='fixed'))

def main():
    rospy.init_node(NODE_NAME, anonymous=False)
    node = CentralNode()
    rospy.spin()

if __name__ == "__main__":
    main()
