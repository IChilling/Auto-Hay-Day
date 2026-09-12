"""Flet controls for the independent Wheating session."""
from __future__ import annotations

import asyncio
import threading

import flet as ft


class WheatingUI:
    def _init_wheating(self):
        self.wheating_running = False
        self.wheating_cancel = None
        self.wheating_message = 'Wheating plants wheat, refills the shop, and uses free advertisements.'
        self.wheating_listed = self.wheating_planted = 0

    def build_wheating_controls(self):
        self.wheating_button = self.button(
            'Stop wheating' if self.wheating_running else 'Start wheating',
            ft.Icons.STOP_ROUNDED if self.wheating_running else ft.Icons.GRASS_ROUNDED,
            self.start_wheating, True, not self.services.connected_serial or self.orders_running or self.board_testing)
        if self.wheating_running:
            self.wheating_button.disabled = False
            self.wheating_button.bgcolor = '#AF4A3B'
            self.wheating_button.color = '#FFFFFF'
            self.order_button.disabled = self.board_button.disabled = self.preview_switch.disabled = True
        self.wheating_reset_button = self.button(
            'Reset app', ft.Icons.RESTART_ALT_ROUNDED, self.reset_wheating,
            disabled=not self.services.connected_serial or self.orders_running or self.board_testing,
        )
        # Reset is intentionally available while Wheating is running so a
        # wedged session can be cancelled, archived, and relaunched in one
        # background operation.
        if self.wheating_running:
            self.wheating_reset_button.disabled = False
        self.wheating_status = ft.Text(self.wheating_message, size=12, color='#758074', expand=True)
        self.wheating_count = ft.Text(self._wheating_count_text(), size=12, color='#377651', weight=ft.FontWeight.W_600)

    def _wheating_count_text(self):
        return f'Planted {self.wheating_planted} · Listed {self.wheating_listed} wheat'

    async def start_wheating(self, e=None):
        if self.wheating_running:
            if self.wheating_cancel:
                self.wheating_cancel.set()
            self.services.cancel_wheating()
            self.wheating_message = 'Stopping Wheating… Waiting for the current action to finish.'
            self.navigate(self.route)
            return
        if self.busy or self.closing or self.board_testing or self.orders_running:
            return
        self.wheating_running = True
        self.wheating_cancel = threading.Event()
        self.wheating_planted = self.wheating_listed = 0
        self.wheating_message = 'Starting Wheating…'
        loop = asyncio.get_running_loop()
        updates = asyncio.Queue(maxsize=1)
        finalized = False

        def enqueue(update):
            if updates.full():
                updates.get_nowait()
            updates.put_nowait(update)

        def progress(update):
            loop.call_soon_threadsafe(enqueue, dict(update))

        async def show_progress():
            while True:
                update = await updates.get()
                if update is None:
                    return
                if self.closing or finalized:
                    continue
                if not self.wheating_cancel or not self.wheating_cancel.is_set():
                    self.wheating_message = update.get('message', self.wheating_message)
                self.wheating_planted = int(update.get('planted', self.wheating_planted))
                self.wheating_listed = int(update.get('listed', self.wheating_listed))
                self.last_message, self.message_error = self.wheating_message, False
                if self.route == 'overview':
                    self.wheating_status.value = self.wheating_message
                    self.wheating_count.value = self._wheating_count_text()
                    if update.get('frame') and not self.reference_shown:
                        frame = update['frame']
                        self.preview_image.src = frame.png
                        self.frame_info.value = self.frame_description()
                        self.resolution_text.value = f'{frame.width} × {frame.height}'
                    self.page.update()

        def finished(result):
            nonlocal finalized
            finalized = True
            self.reference_shown = False
            self.wheating_message = result.message
            self.wheating_planted, self.wheating_listed = result.planted, result.listed

        consumer = asyncio.create_task(show_progress())
        try:
            await self.stop_preview()
            self.reference_shown = False
            await self.job('Starting Wheating', lambda: self.services.start_wheating(
                cancel_event=self.wheating_cancel, progress=progress,
                reset_restart_cooldown=True), 'wheating', finished)
            if self.message_error:
                self.wheating_message = self.last_message
        finally:
            finalized = True
            enqueue(None)
            await consumer
            self.wheating_running = False
            self.wheating_cancel = None
            self.navigate(self.route)

    async def reset_wheating(self, e=None):
        if self.closing or self.orders_running or self.board_testing:
            return
        if self.wheating_running:
            if self.wheating_cancel:
                self.wheating_cancel.set()
            self.services.cancel_wheating()
        self.wheating_planted = self.wheating_listed = 0
        self.wheating_message = 'Resetting app… Stopping Wheating and clearing saved state.'
        self.last_message, self.message_error = self.wheating_message, False
        self.navigate(self.route)
        try:
            result = await asyncio.to_thread(self.services.reset_wheating)
            self.wheating_message = str(result)
            self.last_message, self.message_error = self.wheating_message, False
        except Exception as exc:
            self.wheating_message = str(exc)
            self.last_message, self.message_error = self.wheating_message, True
            self.services.data.log('ERROR', self.wheating_message, 'wheating')
        finally:
            self.wheating_running = False
            self.wheating_cancel = None
            self.navigate(self.route)
