"""Exercise the distributed executable without touching the user's game/data."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch


def run(report_path):
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': False, 'frozen': bool(getattr(sys, 'frozen', False)), 'errors': []}
    started = time.perf_counter()
    try:
        import comtypes.client
        from PIL import Image, ImageDraw, ImageFont

        import hayday
        from hayday.ad_text import AdTextReader
        from hayday.order_vision import OrderReader
        from hayday.wheating_crop import WheatFarmingVision
        from hayday.wheating_recovery import WheatLauncherVision
        from hayday.wheating_vision import WheatingVision

        root = Path(sys._MEIPASS)
        manifest = json.loads((root/'bundle-manifest.json').read_text('utf-8'))
        for name, digest in manifest['assets'].items():
            target = Path(hayday.__file__).parent/'assets'/name
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Bundled asset differs or is missing: {name}')
        report['assets_verified'] = len(manifest['assets'])
        WheatingVision()
        WheatFarmingVision()
        WheatLauncherVision()
        orders = OrderReader()
        if orders._error:
            raise RuntimeError(orders._error)
        report['recognition_loaded'] = True

        # Type library activation only; no desktop inspection or input.
        types = comtypes.client.GetModule('UIAutomationCore.dll')
        automation = comtypes.client.CreateObject(types.CUIAutomation8, interface=types.IUIAutomation2)
        automation.ConnectionTimeout = 1000
        del automation
        report['popup_runtime_loaded'] = True

        sample = Image.new('RGB', (800, 150), 'white')
        ImageDraw.Draw(sample).text((30, 35), 'Reward granted', fill='black',
                                   font=ImageFont.load_default(size=52))
        stream = io.BytesIO()
        sample.save(stream, 'PNG')
        ocr_started = time.perf_counter()
        observation = AdTextReader().read(stream.getvalue())
        report['ocr_seconds'] = round(time.perf_counter()-ocr_started, 3)
        report['ocr_text'] = observation.text
        if observation.error and 'English Windows OCR recognizer is unavailable' in observation.error:
            report['warnings'] = ['Optional English Windows OCR is not installed on this computer.']
        elif observation.error or not observation.reward_granted:
            raise RuntimeError(f'Packaged OCR failed: {observation.error or observation.text}')

        import ui_smoke
        ui_report = report_path.with_name('ui-smoke.json')
        arguments = [sys.argv[0], '--report', str(ui_report), '--hidden',
                     '--screenshots', str(report_path.parent/'screenshots')]
        with patch.object(sys, 'argv', arguments), patch(
                'flet_desktop.ensure_client_cached',
                side_effect=AssertionError('Packaged app must use its bundled desktop client')):
            result = ui_smoke.main()
        report['ui'] = json.loads(ui_report.read_text('utf-8'))
        if result:
            raise RuntimeError('The packaged desktop UI check failed.')
        report['passed'] = True
    except Exception:
        report['errors'].append(traceback.format_exc())
    finally:
        report['seconds'] = round(time.perf_counter()-started, 3)
        report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return 0 if report['passed'] else 1
