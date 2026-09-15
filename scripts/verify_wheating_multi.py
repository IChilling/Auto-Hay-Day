"""Run selected discovered instances together, recording separate live evidence."""
import argparse
import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from hayday.emulators import discover_mumu_instances
from hayday.storage import AppData
from hayday.wheating import WheatingRunner
from hayday.wheating_fleet import WheatingFleet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', action='append', required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reset-restart-cooldown', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Measure capture and recognition time separately for each worker.')
    parser.add_argument('--audit-crops', action='store_true', help='Retain each crop attempt and its clear field for coverage analysis.')
    parser.add_argument('--miss-drag-once', choices=('harvest', 'seed'), help='Execute part of one real crop drag to test partial recovery (35 percent harvest, 70 percent seed).')
    parser.add_argument('--fault-endpoint', help='Restrict the one-time partial drag to this selected instance.')
    args = parser.parse_args()
    choices = {i.endpoint: i for i in discover_mumu_instances()}
    missing = set(args.endpoint)-choices.keys()
    if missing:
        parser.error(f'Instances are not running: {sorted(missing)}')
    args.output.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    counters = {}
    previous = {}
    real_data = AppData()

    def runner(client, root, **options):
        result = WheatingRunner(client, root, **options)
        result.lock_path = real_data.root/'diagnostics'/'orders'/result.device_root.name/'session.lock'
        if args.audit_crops or args.miss_drag_once:
            from hayday.wheating_fields import WheatFields
            audit = args.output/client.serial.replace(':', '_')/'crops'
            audit.mkdir(parents=True, exist_ok=True)
            context = {'stage': None, 'injected': False, 'sequence': 0}

            def fields_factory(run):
                fields = WheatFields(run)
                worker = fields.worker
                record = worker._record

                def recorded(key, stage, **details):
                    record(key, stage, **details)
                    context['stage'] = stage
                    if args.audit_crops:
                        context['sequence'] += 1
                        folder = audit/f'{context["sequence"]:04d}-{stage}'
                        folder.mkdir(exist_ok=True)
                        entry = worker.state['items'][key]
                        (folder/'entry.json').write_text(json.dumps(entry, indent=2), encoding='utf-8')
                        if run.frame is not None:
                            (folder/'observed.png').write_bytes(run.frame.png)
                        for name in ('soil_frame', 'plant_before'):
                            frame = getattr(worker, name, None)
                            if frame is not None:
                                (folder/(name+'.png')).write_bytes(frame.png)
                        for name in ('plant_before', 'harvest_before', 'replant_grid', 'replant_evidence'):
                            proof = entry.get(name)
                            if isinstance(proof, dict) and proof.get('file'):
                                source = worker.state_path.parent/'farming_evidence'/proof['file']
                                if source.is_file():
                                    (folder/(name+'.png')).write_bytes(source.read_bytes())
                worker._record = recorded
                return fields
            result.fields_factory = fields_factory
            drag = client.drag_path

            def partial_drag(points, **kw):
                wanted = {'harvest': 'harvest_attempted', 'seed': 'plant_attempted'}.get(args.miss_drag_once)
                if (wanted and not context['injected'] and context['stage'] == wanted and len(points) > 8
                        and (args.fault_endpoint is None or args.fault_endpoint == client.serial)):
                    context['injected'] = True
                    fraction = .35 if args.miss_drag_once == 'harvest' else .7
                    partial = points[:max(3, round(len(points)*fraction))]
                    kw = dict(kw)
                    if kw.get('time_factors'):
                        kw['time_factors'] = kw['time_factors'][:len(partial)-1]
                    (audit/'fault.json').write_text(json.dumps({'kind': args.miss_drag_once,
                        'intended': points, 'executed': partial, 'at': datetime.now(UTC).isoformat()}), encoding='utf-8')
                    print(f'TEST: {client.serial} executing {len(partial)}/{len(points)} {args.miss_drag_once} route points.', flush=True)
                    return drag(partial, **kw)
                return drag(points, **kw)
            client.drag_path = partial_drag
        if args.profile:
            metrics = {}
            attached = set()

            def measure(call, label):
                def measured(*a, **kw):
                    started = time.perf_counter()
                    try:
                        return call(*a, **kw)
                    finally:
                        elapsed = time.perf_counter()-started
                        row = metrics.setdefault(label, dict(calls=0, seconds=0., max_seconds=0.))
                        row['calls'] += 1
                        row['seconds'] += elapsed
                        row['max_seconds'] = max(row['max_seconds'], elapsed)
                return measured

            raw = result._capture_raw

            def capture(*a, **kw):
                for obj, prefix, names in (
                    (result.vision, 'vision', ('farm', 'plots', 'shop', 'shop_building', 'crop_controls', 'field_bounds')),
                    (result._recovery, 'recovery', ('possible', 'process')),
                ):
                    if obj is not None and id(obj) not in attached:
                        attached.add(id(obj))
                        for name in names:
                            setattr(obj, name, measure(getattr(obj, name), prefix+'.'+name))
                return raw(*a, **kw)

            result._capture_raw = measure(capture, 'capture_raw')
            run = result.run

            def profiled_run():
                try:
                    return run()
                finally:
                    for row in metrics.values():
                        row['mean_ms'] = row['seconds']/row['calls']*1000
                    path = args.output/(client.serial.replace(':', '_')+'-timings.json')
                    path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            result.run = profiled_run
        return result

    def progress(update):
        with lock:
            key = update['instance_key']
            state = update['instances'][key]
            if previous.get(key, 0) >= update['sequence']:
                return
            previous[key] = update['sequence']
            serial = state['serial']
            folder = args.output/serial.replace(':', '_')
            folder.mkdir(exist_ok=True)
            if key not in counters:
                counters[key] = max((int(p.stem) for p in folder.glob('*.png') if p.stem.isdecimal()), default=0)
            counters[key] += 1
            record = {k: v for k,v in state.items() if k not in {'frame','key'}}
            record['at'] = datetime.now(UTC).isoformat()
            if state.get('frame'):
                name = f'{counters[key]:04d}.png'
                (folder/name).write_bytes(state['frame'].png)
                record['screenshot'] = name
            with (folder/'progress.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)

    fleet = WheatingFleet([choices[p] for p in args.endpoint],args.output/'wheating',
        progress=progress, cancel_event=stop, max_seconds=args.seconds, runner_factory=runner,
        reset_restart_cooldown=args.reset_restart_cooldown)

    def watch():
        stopped = set()
        while not stop.wait(.2):
            if (args.output/'stop').exists():
                stop.set()
            for instance in fleet.instances:
                port = instance.endpoint.rsplit(':', 1)[-1]
                if instance.key not in stopped and (args.output/f'stop-{port}').exists():
                    stopped.add(instance.key)
                    fleet.cancel(instance.key)
    threading.Thread(target=watch, daemon=True).start()
    try:
        result = fleet.run()
        records = {key: {k: str(v) if isinstance(v,Path) else v for k,v in r.__dict__.items()
                         if k != 'frame'} for key,r in result.results.items()}
        (args.output/'result.json').write_text(json.dumps(records, indent=2),encoding='utf-8')
        print(json.dumps(records),flush=True)
    finally:
        stop.set()
        real_data.close()


if __name__ == '__main__':
    main()
