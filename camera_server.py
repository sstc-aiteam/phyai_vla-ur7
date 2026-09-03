#!/usr/bin/env python
"""Streams a local camera over TCP for the "remote" CameraConfig backend.

Run this on the machine the camera is physically attached to (e.g. the
robot's control PC) when that's a *different* machine than the one running
policy inference (e.g. a separate GPU machine) -- see
`infer_groot_open_trashcan.py --cam-wrist-backend remote`.

Wire format: a continuous stream of frames, each a 4-byte big-endian length
prefix followed by that many bytes of JPEG-encoded image data. Serves one
client at a time; a client disconnecting (or the inference script being
restarted) just gets replaced by the next connection -- no need to restart
this process to match.

Usage:
    python camera_server.py --backend realsense --index 0 --port 6000
    python camera_server.py --backend usb --index 0 --port 6000
"""

import argparse
import socket
import struct

import cv2

from ur7e_recorder.camera import RealSenseCamera, USBCamera
from ur7e_recorder.config import CameraConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="realsense", choices=("usb", "realsense"))
    parser.add_argument("--index", type=int, default=0, help="Camera/device index (as in CameraConfig)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30,
                         help="Camera capture rate -- independent of the client's control-loop rate; "
                              "the client always reads whatever frame is newest, so this just sets "
                              "how fresh that frame can be")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=6000)
    return parser


def main():
    args = build_arg_parser().parse_args()

    cfg = CameraConfig(index=args.index, width=args.width, height=args.height,
                        fps=args.fps, backend=args.backend)
    camera_cls = RealSenseCamera if args.backend == "realsense" else USBCamera
    camera = camera_cls("cam_server", cfg)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"[OK] camera_server listening on {args.host}:{args.port} "
          f"(backend={args.backend}, index={args.index}, {args.width}x{args.height}@{args.fps})")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            print("Waiting for a client to connect...")
            conn, addr = server.accept()
            print(f"[OK] Client connected: {addr}")
            try:
                with conn:
                    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    while True:
                        frame = camera.read()
                        if frame is None:
                            continue
                        ok, buf = cv2.imencode(".jpg", frame, encode_params)
                        if not ok:
                            continue
                        payload = buf.tobytes()
                        conn.sendall(struct.pack(">I", len(payload)) + payload)
            except (ConnectionError, OSError) as e:
                print(f"[WARN] Client disconnected ({e}); waiting for a new connection...")
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C pressed.")
    finally:
        camera.release()
        server.close()
        print("[DONE] camera_server stopped.")


if __name__ == "__main__":
    main()
