#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import threading

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

# [ Chest, Left Knee, Right Hand, Left Hand, Right Knee ]
SENSOR_ORDER = ["chest", "lknee", "rhand", "lhand", "rknee"]
TOPIC_BY_SENSOR = {
    "chest": TOPIC_CHEST,
    "lknee": TOPIC_L_KNEE,
    "rhand": TOPIC_R_HAND,
    "lhand": TOPIC_L_HAND,
    "rknee": TOPIC_R_KNEE,
}

# Parametros ApproximateTimeSynchronizer (ATS)
SYNC_SLOP_SEC = 0.05
QUEUE_SIZE    = 10

# Parametro Retenedor Orden Cero (ZOH)
CENTRAL_RATE_HZ     = 50.0
FRESH_MAX_AGE_SEC   = 0.20   # solo diagnóstico / “fresh” lógico
ZOH_MAX_HOLD_SEC    = 2.0    # Si un sensor supera esto sin actualizar -> uniform

# Parametros de conmutación ATS-ZOH
FAILOVER_SEC        = 0.35   # Si ATS no dispara por este tiempo cambia a ZOH
RECOVER_STABLE_SEC  = 1.00   # Si ATS vuelve y se mantiene estamos SYNC
STARTUP_GRACE_SEC   = 2.00   # Espera inicial antes de declarar failover

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
        rospy.loginfo("[central] Iniciando nodo central ATS(SYNC) + ZOH(Uniform) con failover/recover...")
        self.pub = rospy.Publisher(PUB_TOPIC, Probs, queue_size=10)
        self._lock = threading.Lock()

        # Cache por sensor (para ZOH)
        self.last_probs = {s: None for s in SENSOR_ORDER}   # np.array shape (K,)
        self.last_rx_t  = {s: None for s in SENSOR_ORDER}   # tiempo local (central) de recepción

        self.K = None
        self.model = None
        self.input_dim = None
        self.num_classes = None

        # Estado de modos
        self.mode = "SYNC"
        self.t0 = rospy.Time.now().to_sec()
        self.last_sync_cb_t = None     # Ultima vez que ATS se disparó
        self.good_since = None         # para RECOVER_STABLE_SEC

        # Suscripciones para cache (ZOH)
        self.sub_cache = []
        for s in SENSOR_ORDER:
            topic = TOPIC_BY_SENSOR[s]
            self.sub_cache.append(
                rospy.Subscriber(topic, Probs, self._cb_cache, callback_args=s, queue_size=50)
            )

        # Suscriptores con message_filters - ATS(SYNC)
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

        # Timer para ZOH y failover watchdog
        period = 1.0 / float(CENTRAL_RATE_HZ)
        self.timer = rospy.Timer(rospy.Duration(period), self._timer_cb)

        rospy.loginfo("[central] Publicando %s. Modo inicial: SYNC (ATS).", PUB_TOPIC)

    # Callback para ZOH(Uniform)
    def _cb_cache(self, msg, sensor_id):
        p = np.array(msg.probs, dtype=np.float32)
        now_sec = rospy.Time.now().to_sec()

        with self._lock:
            if self.K is None:
                self.K = len(p)
                rospy.loginfo("[central] K detectado (desde cache): %d", self.K)
            elif len(p) != self.K:
                rospy.logwarn_throttle(
                    1.0,
                    "[central] Tamaño inconsistente en cache %s: len=%d (esperado %d). Ignorando.",
                    sensor_id, len(p), self.K
                )
                return

            self.last_probs[sensor_id] = p
            self.last_rx_t[sensor_id]  = float(now_sec)

    # ATS callback (SYNC)
    def callback_sync(self, msg_chest, msg_lhand, msg_rhand, msg_lknee, msg_rknee):
        now_sec = rospy.Time.now().to_sec()

        # Registrar que ATS está vivo (clave para failover/recover)
        with self._lock:
            self.last_sync_cb_t = float(now_sec)

            # Si estamos en ZOH, esto marca “buenos eventos” para recuperar
            if self.mode == "ZOH":
                if self.good_since is None:
                    self.good_since = float(now_sec)
                if (now_sec - self.good_since) >= RECOVER_STABLE_SEC:
                    self.mode = "SYNC"
                    self.good_since = None
                    rospy.loginfo("[central] RECOVER -> SYNC (ATS estable por %.3fs)", RECOVER_STABLE_SEC)
            else:
                self.good_since = None

            # Si no estamos en SYNC, no publicamos por ATS (evita doble fuente)
            if self.mode != "SYNC":
                return
        
        # Extraer probs
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

        # Inicializar K/model si no estaban
        with self._lock:
            if self.K is None:
                self.K = K
                rospy.loginfo("[central] K detectado (desde ATS): %d", self.K)
            elif self.K != K:
                rospy.logwarn_throttle(1.0, "[central][SYNC] K cambió (cache=%d ats=%d). Ignorando batch.", self.K, K)
                return

        # Concatenación en orden fijo
        # [ Chest (AG), Left Knee (AG), Right Hand (AG), Left Hand (Q), Right Knee (Q) ]
        x = np.concatenate([p_chest, p_lknee, p_rhand, p_lhand, p_rknee], axis=0) 

        # Inferencia + publish
        self._infer_and_publish(x, mode_tag="SYNC")

        # Debug: span de stamps del ATS (solo diagnóstico)
        stamps = np.array([
            msg_chest.header.stamp.to_sec(),
            msg_lhand.header.stamp.to_sec(),
            msg_rhand.header.stamp.to_sec(),
            msg_lknee.header.stamp.to_sec(),
            msg_rknee.header.stamp.to_sec()
        ], dtype=np.float64)
        span = float(stamps.max() - stamps.min())
        rospy.loginfo_throttle(1.0, "[central][ATS-DBG] span=%.6f", span)

    # ZOH builder
    def _build_input_zoh_uniform(self, now_sec):
        vecs = []
        statuses = {}
        ages = {}

        for s in SENSOR_ORDER:
            p = self.last_probs[s]
            t = self.last_rx_t[s]

            if p is None or t is None:
                v = np.ones((self.K,), dtype=np.float32) / float(self.K)
                statuses[s] = "uniform_never"
                ages[s] = None
            else:
                age = float(now_sec - t)
                ages[s] = age
                if age <= ZOH_MAX_HOLD_SEC:
                    v = p
                    statuses[s] = "zoh" if age > FRESH_MAX_AGE_SEC else "fresh"
                else:
                    v = np.ones((self.K,), dtype=np.float32) / float(self.K)
                    statuses[s] = "uniform_hold_exceeded"

            vecs.append(v)

        x = np.concatenate(vecs, axis=0)
        return x, statuses, ages

    # Inference + publish
    def _infer_and_publish(self, x, mode_tag):   
        if self.model is None:
            self.input_dim = x.shape[0]
            self.num_classes = self.K
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
        rospy.loginfo_throttle(
            1.0, 
            "[central][%s] pred=%d, probs=%s", 
            mode_tag,
            pred_class,
            np.array2string(probs_global, precision=3, floatmode='fixed')
        )

    # Timer watchdog + ZOH publish
    def _timer_cb(self, _evt):
        now_sec = rospy.Time.now().to_sec()
        with self._lock:
            # Failover: si ATS no ha disparado recientemente
            if self.mode == "SYNC":
                if self.last_sync_cb_t is None:
                    # gracia inicial para permitir “arranque”
                    if (now_sec - self.t0) > STARTUP_GRACE_SEC:
                        self.mode = "ZOH"
                        self.good_since = None
                        rospy.logwarn("[central] FAILOVER -> ZOH (ATS no disparó durante startup)")
                else:
                    dt = now_sec - self.last_sync_cb_t
                    if dt >= FAILOVER_SEC:
                        self.mode = "ZOH"
                        self.good_since = None
                        rospy.logwarn("[central] FAILOVER -> ZOH (ATS silent por %.3fs)", dt)

            # Si estamos en SYNC, no publicamos por timer.
            if self.mode != "ZOH":
                return

            # No podemos inferir en ZOH hasta conocer K
            if self.K is None:
                return

            x, statuses, ages = self._build_input_zoh_uniform(now_sec)

        # Inferencia + publish en ZOH (fuera del lock)
        self._infer_and_publish(x, mode_tag="ZOH")

        st_csv = ",".join(
            f"{s}:{statuses[s]}:{(-1.0 if ages[s] is None else ages[s]):.3f}"
            for s in SENSOR_ORDER
        )
        rospy.loginfo_throttle(1.0, "[central][ZOH-DBG] %s", st_csv)

def main():
    rospy.init_node(NODE_NAME, anonymous=False)
    _ = CentralNode()
    rospy.spin()

if __name__ == "__main__":
    main()
