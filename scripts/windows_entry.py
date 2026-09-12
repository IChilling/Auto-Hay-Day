"""Single-file desktop entry point, including the bounded internal OCR worker."""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def main():
    multiprocessing.freeze_support()
    if len(sys.argv) == 4 and sys.argv[1] == '--hayday-ocr-worker':
        from hayday.ad_text import _worker_main
        _worker_main(sys.argv[2], sys.argv[3])
        return 0

    if getattr(sys, 'frozen', False):
        client = Path(sys._MEIPASS)/'flet-client'
        if not (client/'flet.exe').is_file():
            raise RuntimeError('The bundled desktop client is missing. Rebuild the executable.')
        os.environ['FLET_VIEW_PATH'] = str(client)

    if len(sys.argv) == 3 and sys.argv[1] == '--self-test':
        from windows_smoke import run
        return run(Path(sys.argv[2]))

    from hayday.ui import run
    run()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
