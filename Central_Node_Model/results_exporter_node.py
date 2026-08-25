#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import rospy
from std_msgs.msg import Empty, String

# ================== CONFIG ==================
EXPORT_TOPIC = "/har/ui/export_results"     # Unity -> ROS (pide zip)
URL_TOPIC    = "/har/ui/results_url"       # ROS -> Unity (opcional)

RUNS_ROOT  = os.path.expanduser("~/har_runs")
PUBLIC_DIR = os.path.join(RUNS_ROOT, "_public")
ZIP_NAME   = "latest_results.zip"

HTTP_HOST  = "0.0.0.0"
HTTP_PORT  = 8000
# ===========================================

def get_latest_run_dir(root: str):
    if not os.path.isdir(root):
        return None
    # solo directorios, excluye _public
    dirs = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path) and name != "_public":
            dirs.append(path)
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]

def make_zip_from_dir(src_dir: str, out_zip: str):
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                full = os.path.join(root, f)
                rel  = os.path.relpath(full, src_dir)
                zf.write(full, arcname=rel)

class PublicDirHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

def http_server_thread():
    httpd = ThreadingHTTPServer((HTTP_HOST, HTTP_PORT), PublicDirHandler)
    rospy.loginfo(f"[exporter] HTTP serving {PUBLIC_DIR} on :{HTTP_PORT}")
    httpd.serve_forever()

class ResultsExporter:
    def __init__(self):
        os.makedirs(PUBLIC_DIR, exist_ok=True)

        self.pub_url = rospy.Publisher(URL_TOPIC, String, queue_size=1, latch=True)
        rospy.Subscriber(EXPORT_TOPIC, Empty, self.on_export)

        t = threading.Thread(target=http_server_thread, daemon=True)
        t.start()

        rospy.loginfo(f"[exporter] RUNS_ROOT={RUNS_ROOT}")
        rospy.loginfo("[exporter] Ready. Waiting for /har/ui/export_results ...")

    def on_export(self, _msg):
        latest = get_latest_run_dir(RUNS_ROOT)
        if latest is None:
            rospy.logwarn(f"[exporter] No runs found in {RUNS_ROOT}")
            return

        out_zip = os.path.join(PUBLIC_DIR, ZIP_NAME)
        rospy.loginfo(f"[exporter] Export requested. Zipping: {latest} -> {out_zip}")

        try:
            make_zip_from_dir(latest, out_zip)
        except Exception as e:
            rospy.logerr(f"[exporter] Zip failed: {e}")
            return

        # Publicamos URL (opcional). Unity puede construirlo usando su ROS IP.
        # Aquí publicamos una URL con 'localhost' como placeholder.
        url = f"http://localhost:{HTTP_PORT}/{ZIP_NAME}"
        self.pub_url.publish(String(url))

        rospy.loginfo(f"[exporter] Done. ZIP ready: {out_zip}")

if __name__ == "__main__":
    rospy.init_node("results_exporter_node")
    ResultsExporter()
    rospy.spin()
