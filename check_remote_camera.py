#!/usr/bin/env python
"""Diagnostic check for a camera_server.py stream.

Connects to camera_server.py the same way infer_groot_open_trashcan.py's
--cam-wrist-backend remote does, reports whether frames are actually
arriving and at what real rate (via RemoteCamera.frame_count, not just how
often this script happens to poll), and saves a snapshot JPEG to disk so
you can eyeball what the camera sees -- useful to verify the stream
end-to-end (network reachability, camera focus/exposure, frame rate) before
trusting it for real-robot inference.

Usage:
    python check_remote_camera.py --host <camera-machine-ip> --port 6000

    # Also open a live preview window (needs a display on this machine):
    python check_remote_camera.py --host <camera-machine-ip> --port 6000 --show
"""

import argparse
import sys
import time

import cv2

from ur7e_recorder.camera import RemoteCamera
from ur7e_recorder.config import CameraConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="camera_server.py's host/IP")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--connect-timeout", type=float, default=15.0, help="Seconds to wait for the first frame")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds to measure frame rate over")
    parser.add_argument("--snapshot", default="camera_snapshot.jpg", help="Where to save a sample frame")
    parser.add_argument("--show", action="store_true", help="Also open a live cv2 preview window (needs a display)")
    return parser


def main():
    args = build_arg_parser().parse_args()

    print(f"Connecting to camera_server.py at {args.host}:{args.port} ...")
    cam = RemoteCamera("check", CameraConfig(index=0, backend="remote", host=args.host, port=args.port))

    deadline = time.time() + args.connect_timeout
    frame = None
    while time.time() < deadline:
        frame = cam.read()
        if frame is not None:
            break
        time.sleep(0.1)

    if frame is None:
        cam.release()
        sys.exit(f"[FAIL] No frame received within {args.connect_timeout}s -- is camera_server.py "
                  f"running and reachable at {args.host}:{args.port}? (firewall, wrong IP/port, "
                  f"or the camera itself failed to open on the server side -- check its console output)")

    print(f"[OK] First frame received: shape={frame.shape}, dtype={frame.dtype}")
    if cv2.imwrite(args.snapshot, frame):
        print(f"[OK] Snapshot saved to {args.snapshot} -- open it to confirm the camera sees the right thing")
    else:
        print(f"[WARN] Failed to write snapshot to {args.snapshot}")

    print(f"Measuring frame rate over {args.duration}s (Ctrl+C / 'q' in the preview window to stop early)...")
    start_count = cam.frame_count
    start = time.time()
    try:
        while time.time() - start < args.duration:
            if args.show:
                shown = cam.read()
                if shown is not None:
                    cv2.imshow("remote camera check", shown)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    elapsed = time.time() - start
    frames_seen = cam.frame_count - start_count
    fps = frames_seen / elapsed if elapsed > 0 else 0.0

    print(f"[OK] {frames_seen} new frames arrived in {elapsed:.1f}s (~{fps:.1f} fps)")
    if frames_seen == 0:
        print("[WARN] No new frames arrived during the measurement window -- the stream connected but "
              "then stalled (camera_server.py's camera may have died, or the link is very slow/dropping).")

    if args.show:
        cv2.destroyAllWindows()
    cam.release()
    print("[DONE]")


if __name__ == "__main__":
    main()
