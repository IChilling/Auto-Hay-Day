"""Check the actual Mac runtime, image assets and on-device OCR before activation."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import flet_desktop  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hayday.ad_text import AdTextReader  # noqa: E402
from hayday.order_vision import OrderReader  # noqa: E402
from hayday.wheating_crop import WheatFarmingVision  # noqa: E402
from hayday.wheating_recovery import WheatLauncherVision  # noqa: E402
from hayday.wheating_vision import WheatingVision  # noqa: E402


def main():
    if sys.platform != "darwin":
        raise RuntimeError("The native macOS check must run on a Mac.")
    client = Path(flet_desktop.ensure_client_cached())
    if not any(client.glob("**/*.app/Contents/MacOS/*")):
        raise RuntimeError(f"The Flet desktop client is incomplete: {client}")
    WheatingVision()
    WheatFarmingVision()
    WheatLauncherVision()
    orders = OrderReader()
    if orders._error:
        raise RuntimeError(orders._error)
    sample = Image.new("RGB", (800, 150), "white")
    ImageDraw.Draw(sample).text((30, 35), "Reward granted", fill="black",
                               font=ImageFont.load_default(size=52))
    stream = io.BytesIO()
    sample.save(stream, "PNG")
    result = AdTextReader(timeout_seconds=10).read(stream.getvalue())
    if result.error or not result.reward_granted:
        raise RuntimeError(f"Apple Vision OCR check failed: {result.error or result.text}")
    print("Desktop client, image recognition assets and Apple Vision OCR are ready.")


if __name__ == "__main__":
    main()
