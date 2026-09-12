"""Bounded live Wheating run with timestamped progress and a device session lock."""
import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hayday.adb import AdbClient
from hayday.storage import AppData
from hayday.wheating import WheatingRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=360)
    parser.add_argument('--simulate-silo', action='store_true',
                        help='Interrupt one real harvest halfway, then exercise the silo recovery workflow.')
    parser.add_argument('--warmup-harvests', type=int, default=0,
                        help='Keep wheat for this many harvests before the simulated interruption.')
    args = parser.parse_args()
    data = AppData()
    settings = json.loads((data.root / 'settings.json').read_text())
    def progress(update):
        print(datetime.now(UTC).isoformat(), update['message'], flush=True)
    client = AdbClient(settings['adb_path'], serial=settings['selected_serial'], timeout_seconds=30)
    try:
        runner = WheatingRunner(client, data.root / 'diagnostics' / 'wheating',
                               progress=progress, max_seconds=args.seconds)
        if args.simulate_silo:
            original_drag = client.drag_path
            injected = False
            harvests = 0
            if args.warmup_harvests:
                from hayday.wheating_shop import WheatShop
                original_service = WheatShop.service
                def warmup_service(shop, *a, **kw):
                    return original_service(shop, *a, **kw) if injected else True
                WheatShop.service = warmup_service
            def interrupt_harvest(points, **options):
                global injected, harvests
                if not injected and options.get('min_waypoint_ms') in (16, 40) and len(points) > 6:
                    harvests += 1
                    if harvests <= args.warmup_harvests:
                        return original_drag(points, **options)
                    injected = True
                    partial = points[:max(3, len(points)//2)]
                    options = dict(options)
                    if options.get('time_factors'):
                        options['time_factors'] = options['time_factors'][:len(partial)-1]
                    original_drag(partial, **options)
                    print('TEST: interrupted one harvest halfway; simulating Silo Full recovery.', flush=True)
                    runner.save_json(runner.diagnostics/'test_injection.json', {
                        'kind': 'simulated_silo_interruption',
                        'route_points': len(points), 'executed_points': len(partial),
                        'at': datetime.now(UTC).isoformat(),
                    })
                    from hayday.wheating_silo import SiloFullDetected
                    raise SiloFullDetected(runner._capture_raw(), None)
                return original_drag(points, **options)
            client.drag_path = interrupt_harvest
        result = runner.run()
        print(result.status, result.message, result.planted, result.listed, result.diagnostics, flush=True)
    finally:
        client.close()
