"""Prepare the native client and verify source-install dependencies, without ADB."""
import io
from pathlib import Path

import comtypes.client
import flet_desktop
from PIL import Image, ImageDraw, ImageFont

from hayday.ad_text import AdTextReader
from hayday.order_vision import OrderReader
from hayday.wheating_crop import WheatFarmingVision
from hayday.wheating_harvest_resume import can_reselect  # noqa: F401
from hayday.wheating_recovery import WheatLauncherVision
from hayday.wheating_vision import WheatingVision


def main():
    client = Path(flet_desktop.ensure_client_cached())/'flet'/'flet.exe'
    if not client.is_file():
        raise RuntimeError(f'The desktop client is incomplete: {client.parent}')
    WheatingVision()
    WheatFarmingVision()
    WheatLauncherVision()
    orders = OrderReader()
    if orders._error:
        raise RuntimeError(orders._error)
    types = comtypes.client.GetModule('UIAutomationCore.dll')
    automation = comtypes.client.CreateObject(types.CUIAutomation8, interface=types.IUIAutomation2)
    automation.ConnectionTimeout = 1000
    del automation
    sample = Image.new('RGB', (800, 150), 'white')
    ImageDraw.Draw(sample).text((30, 35), 'Reward granted', fill='black',
                               font=ImageFont.load_default(size=52))
    stream = io.BytesIO()
    sample.save(stream, 'PNG')
    observation = AdTextReader(timeout_seconds=10).read(stream.getvalue())
    if observation.error and 'English Windows OCR recognizer is unavailable' in observation.error:
        print('Optional English Windows OCR is unavailable; visual recognition remains available.')
    elif observation.error or not observation.reward_granted:
        raise RuntimeError(f'Windows text recognition failed: {observation.error or observation.text}')
    print('Desktop client, recognition libraries, and Windows popup support are ready.')


if __name__ == '__main__':
    main()
