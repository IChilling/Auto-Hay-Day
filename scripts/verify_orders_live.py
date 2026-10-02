"""Run a bounded order session on one discovered MuMu device, preserving its state."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hayday.adb import AdbClient
from hayday.emulators import discover_mumu_instances
from hayday.orders import OrderRunner
from hayday.storage import _default_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Required when multiple MuMu devices are running.")
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--deliveries", type=int, default=1)
    parser.add_argument("--capture-all", type=Path, help="Save every frame for recognition debugging.")
    args = parser.parse_args()
    instances = [instance for instance in discover_mumu_instances()
                 if args.endpoint is None or instance.endpoint == args.endpoint]
    if len(instances) != 1:
        parser.error("Select exactly one running MuMu device with --endpoint.")
    instance = instances[0]
    # Order state is JSON; this verifier does not use the desktop activity DB.
    data_root = _default_root()
    client = AdbClient(str(instance.adb_path), serial=instance.endpoint)
    runner = OrderRunner(
        client, data_root / "diagnostics" / "orders",
        max_seconds=args.seconds, max_deliveries=args.deliveries,
        progress=lambda update: print(datetime.now(UTC).isoformat(), update["message"], flush=True),
    )
    if args.capture_all:
        capture_root = args.capture_all / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        capture_root.mkdir(parents=True)
        capture = client.capture
        sequence = 0

        def record_capture():
            nonlocal sequence
            frame = capture()
            sequence += 1
            (capture_root / f"{sequence:04d}.png").write_bytes(frame.png)
            return frame

        client.capture = record_capture
        print(f"Frames: {capture_root}", flush=True)
    try:
        client.connect(instance.endpoint)
        result = runner.run()
        print(json.dumps({"status": result.status, "message": result.message,
                          "deliveries": result.deliveries, "diagnostics": str(result.diagnostics)}),
              flush=True)
        return 0 if result.success else 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
