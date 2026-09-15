from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import flet as ft

from hayday import __version__
from hayday.board_test import BoardTestResult
from hayday.services import Services
from hayday.wheating_ui import WheatingUI

BG = "#F5F6F0"
WHITE = "#FFFFFF"
INK = "#223B2C"
MUTED = "#758074"
GREEN = "#377651"
PALE = "#EAF0E6"
LINE = "#E2E6DC"
GOLD = "#B27B28"
REFERENCE = Path(__file__).parent / "assets" / "farm-reference.png"


def label(text: str, size: int = 13, color: str = INK, bold: bool = False, **kw):
    return ft.Text(
        text,
        size=size,
        color=color,
        weight=ft.FontWeight.W_600 if bold else ft.FontWeight.W_400,
        **kw,
    )


def badge(text: str, color: str = GREEN, bg: str = PALE):
    return ft.Container(
        label(text, 11, color, True),
        bgcolor=bg,
        padding=ft.Padding.symmetric(horizontal=11, vertical=6),
        border_radius=20,
    )


def card(content: ft.Control, **kw):
    return ft.Container(
        content, bgcolor=WHITE, border=ft.Border.all(1, LINE), border_radius=16, padding=22, **kw
    )


class MainWindow(WheatingUI):
    def __init__(self, page: ft.Page, services: Services | None = None):
        self.page = page
        self.services = services or Services()
        self.route = "overview"
        self.busy = False
        self.closing = False
        self.preview_running = False
        self.preview_task: asyncio.Task | None = None
        self.preview_stop = asyncio.Event()
        self.jobs: set[asyncio.Task] = set()
        self.reference_shown = True
        self.last_message = "Ready to connect. Choose your emulator instance to get started."
        self.message_error = False
        self.board_testing = False
        self.board_cancel: threading.Event | None = None
        self.orders_running = False
        self.order_cancel: threading.Event | None = None
        self.order_message = "Start an order session when your farm is ready."
        self.order_deliveries = 0
        self.order_limit = 10
        self.order_seconds = 1200
        self._init_wheating()
        self.search = ""
        self._setup_page()
        self.content = ft.Container(
            expand=True, padding=ft.Padding.only(left=30, right=30, top=25, bottom=18)
        )
        self.nav = ft.Column(spacing=6)
        self.status = label("SETUP WORKSPACE", 10, MUTED, True)
        self.progress = ft.ProgressBar(height=2, color=GREEN, bgcolor=BG, visible=False)
        sidebar = ft.Container(
            ft.Column(
                [
                    ft.Container(
                        ft.Row(
                            [
                                ft.Container(
                                    ft.Icon(ft.Icons.GRASS_ROUNDED, color=WHITE, size=29),
                                    bgcolor=GREEN,
                                    width=46,
                                    height=46,
                                    border_radius=14,
                                    alignment=ft.Alignment.CENTER,
                                ),
                                ft.Column(
                                    [
                                        label("Hay Day", 21, WHITE, True),
                                        label("A U T O M A T I O N", 9, "#B6CBB7"),
                                    ],
                                    spacing=2,
                                ),
                            ],
                            spacing=12,
                        ),
                        padding=ft.Padding.only(bottom=38, top=14),
                    ),
                    label("WORKSPACE", 10, "#92AF99", True),
                    self.nav,
                    ft.Container(expand=True),
                    ft.Container(
                        ft.Column(
                            [
                                ft.Icon(ft.Icons.LOCAL_SHIPPING_OUTLINED, color="#C8DCAC", size=28),
                                label("A foundation for\nevery farm.", 16, WHITE, True),
                                label(
                                    "Truck orders and\nobserved resource work.", 12, "#ADC1AD"
                                ),
                            ],
                            spacing=12,
                        ),
                        bgcolor="#2C4935",
                        padding=18,
                        border_radius=14,
                    ),
                    ft.Container(
                        label(f"FOUNDATION BUILD   /   {__version__}", 9, "#92AF99"),
                        padding=ft.Padding.only(top=18, bottom=6),
                    ),
                ],
                expand=True,
                spacing=16,
            ),
            width=225,
            bgcolor=INK,
            padding=22,
        )
        top = ft.Container(
            ft.Row(
                [
                    label("WORKSPACE  /  HAY DAY AUTOMATION", 10, MUTED, True, expand=True),
                    self.status,
                    badge("Truck orders"),
                ]
            ),
            padding=ft.Padding.symmetric(horizontal=30, vertical=16),
            border=ft.Border.only(bottom=ft.BorderSide(1, LINE)),
        )
        self.shell = ft.Row(
            [sidebar, ft.Column([top, self.progress, self.content], spacing=0, expand=True)],
            spacing=0,
            expand=True,
        )
        self.page.add(self.shell)
        self.navigate("overview")

    def _setup_page(self):
        p = self.page
        p.title = "Hay Day Automation"
        p.bgcolor = BG
        p.padding = 0
        p.theme_mode = ft.ThemeMode.LIGHT
        p.theme = ft.Theme(color_scheme_seed=GREEN, font_family="Segoe UI")
        p.window.width = 1320
        p.window.height = 900
        p.window.min_width = 1050
        p.window.min_height = 740
        p.window.prevent_close = True
        p.window.on_event = self._window_event
        p.on_disconnect = self._disconnect_event

    async def initialize(self):
        await self.job("Discovering emulators", self.services.discover, category="connection")

    def button(self, text, icon, handler, primary=False, disabled=False):
        disabled = disabled or self.busy
        return ft.Button(
            text,
            icon=icon,
            on_click=handler,
            height=40,
            bgcolor=LINE if disabled else GREEN if primary else PALE,
            color=MUTED if disabled else WHITE if primary else GREEN,
            disabled=disabled,
            style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=9)),
        )

    def navigate(self, route: str):
        if self.closing:
            return
        self.route = route
        items = [
            ("overview", "Overview", ft.Icons.DASHBOARD_OUTLINED),
            ("devices", "Emulators", ft.Icons.DEVICES_ROUNDED),
            ("captures", "Capture library", ft.Icons.PHOTO_LIBRARY_OUTLINED),
            ("activity", "Activity", ft.Icons.RECEIPT_LONG_OUTLINED),
            ("settings", "Settings", ft.Icons.TUNE_ROUNDED),
        ]
        self.nav.controls = [
            ft.Container(
                ft.Row(
                    [
                        ft.Icon(icon, size=20, color=WHITE if key == route else "#ADC1AD"),
                        label(title, 13, WHITE if key == route else "#ADC1AD", key == route),
                    ],
                    spacing=12,
                ),
                padding=13,
                border_radius=9,
                bgcolor="#3D5A42" if key == route else None,
                on_click=lambda e, key=key: self.navigate(key),
                ink=True,
            )
            for key, title, icon in items
        ]
        self.status.value = (
            f"WHEATING  ·  {len(self.services.settings.wheating_instances)} FARMS SELECTED"
            if self.services.settings.wheating_instances else
            f"CONNECTED  ·  {self.services.connected_serial}"
            if self.services.connected_serial
            else "NO DEVICE SELECTED"
        )
        self.progress.visible = self.busy
        build = {
            "overview": self.overview,
            "devices": self.devices,
            "captures": self.captures,
            "activity": self.activity,
            "settings": self.settings,
        }
        self.content.content = build[route]()
        self.page.update()

    def heading(self, eyebrow, title, subtitle, action=None):
        return ft.Row(
            [
                ft.Column(
                    [
                        label(eyebrow.upper(), 10, GREEN, True),
                        label(title, 30, INK, True),
                        label(subtitle, 13, MUTED),
                    ],
                    spacing=6,
                    expand=True,
                ),
                *([action] if action else []),
            ],
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def notice(self):
        return ft.Container(
            ft.Row(
                [
                    ft.Icon(
                        ft.Icons.ERROR_OUTLINE if self.message_error else ft.Icons.INFO_OUTLINE,
                        size=17,
                        color="#AF4A3B" if self.message_error else GREEN,
                    ),
                    label(
                        self.last_message,
                        12,
                        "#AF4A3B" if self.message_error else GREEN,
                        expand=True,
                    ),
                ],
                spacing=9,
            ),
            bgcolor="#FFF0EB" if self.message_error else PALE,
            border_radius=9,
            padding=12,
        )

    def overview(self):
        svc = self.services
        connected = bool(svc.connected_serial)
        frame = svc.frame
        using_reference = self.reference_shown or frame is None
        source = (
            str(REFERENCE)
            if using_reference and REFERENCE.is_file()
            else frame.png
            if frame
            else None
        )
        preview = (
            ft.Image(
                src=source,
                fit=ft.BoxFit.CONTAIN,
                gapless_playback=True,
                semantics_label="Reference farm screenshot"
                if using_reference
                else "Captured emulator screen",
                expand=True,
            )
            if source
            else label("Connect an emulator to see your farm.", 16, MUTED)
        )
        self.preview_image = preview
        self.resolution_text = label(
            f"{frame.width} × {frame.height}" if frame else "No capture yet", 17, INK, True
        )
        self.frame_badge = badge("REFERENCE IMAGE" if using_reference else "DEVICE CAPTURE")
        self.frame_info = label(
            "Supplied farm image · visual reference only"
            if using_reference
            else self.frame_description(),
            11,
            MUTED,
        )
        self.capture_button = self.button(
            "Capture screen",
            ft.Icons.PHOTO_CAMERA_OUTLINED,
            self.capture_screen,
            True,
            not connected,
        )
        self.save_button = self.button(
            "Save PNG", ft.Icons.SAVE_ALT_ROUNDED, self.save_capture, disabled=using_reference
        )
        self.preview_switch = ft.Switch(
            label="Live preview",
            value=self.preview_running,
            on_change=self.toggle_preview,
            disabled=not connected or self.busy,
            active_color=GREEN,
        )
        self.board_button = self.button(
            "Cancel test" if self.board_testing else "Test quest board",
            ft.Icons.STOP_ROUNDED if self.board_testing else ft.Icons.MANAGE_SEARCH_ROUNDED,
            self.test_quest_board,
            True,
            not connected,
        )
        if self.board_testing:
            self.board_button.disabled = False
            self.board_button.bgcolor = "#AF4A3B"
            self.board_button.color = WHITE
        self.order_button = self.button(
            "Stop orders" if self.orders_running else "Start orders",
            ft.Icons.STOP_ROUNDED if self.orders_running else ft.Icons.LOCAL_SHIPPING_OUTLINED,
            self.start_orders,
            True,
            not connected,
        )
        if self.orders_running:
            self.order_button.disabled = False
            self.order_button.bgcolor = "#AF4A3B"
            self.order_button.color = WHITE
        self.order_status = label(self.order_message, 12, MUTED, expand=True)
        self.order_count = label(
            f"Delivered {self.order_deliveries}/{self.order_limit} · {self.order_seconds // 60} min limit",
            12, GREEN, True,
        )
        self.build_wheating_controls()
        board_result = self.services.last_board_test
        board_summary = (
            "Searching and checking the board… Use Cancel test to stop."
            if self.board_testing
            else board_result.message
            if board_result
            else "Test finds the board, taps it once, and verifies the order panel. It does not send or delete orders."
        )
        preview_card = card(
            ft.Column(
                [
                    ft.Row(
                        [
                            label("Farm viewport", 16, INK, True, expand=True),
                            self.frame_badge,
                        ],
                    ),
                    ft.Row(
                        [
                            self.board_button,
                            self.order_button,
                            self.wheating_button,
                            self.wheating_reset_button,
                        ],
                        wrap=True,
                        run_spacing=8,
                    ),
                    label(
                        board_summary,
                        12,
                        "#AF4A3B"
                        if board_result and not board_result.success and not self.board_testing
                        else MUTED,
                    ),
                    ft.Row([self.order_status, self.order_count], spacing=12),
                    ft.Row([self.wheating_status, self.wheating_count], spacing=12),
                    self.wheating_instances_panel,
                    ft.Container(
                        preview,
                        bgcolor="#E8EDDF",
                        border_radius=10,
                        height=330,
                        alignment=ft.Alignment.CENTER,
                        clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
                    ),
                    ft.Row(
                        [
                            self.frame_info,
                            ft.Container(expand=True),
                            ft.TextButton(
                                "Show reference" if not using_reference else "Show device",
                                on_click=self.toggle_source,
                                disabled=frame is None,
                            ),
                        ]
                    ),
                    ft.Row(
                        [
                            self.capture_button,
                            self.save_button,
                            ft.Container(expand=True),
                            self.preview_switch,
                        ]
                    ),
                ],
                spacing=12,
            )
        )
        summary = ft.ResponsiveRow(
            [
                card(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.DEVICES_OUTLINED, color=GREEN),
                            ft.Column(
                                [
                                    label("EMULATOR", 9, MUTED, True),
                                    label(
                                        f'{len(svc.settings.wheating_instances)} farms selected'
                                        if svc.settings.wheating_instances else
                                        "Connected" if connected else "Awaiting device",
                                        17,
                                        INK,
                                        True,
                                    ),
                                ],
                                spacing=4,
                            ),
                        ]
                    ),
                    col=4,
                ),
                card(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.CROP_FREE_ROUNDED, color=GREEN),
                            ft.Column(
                                [
                                    label("CAPTURE SIZE", 9, MUTED, True),
                                    self.resolution_text,
                                ],
                                spacing=4,
                            ),
                        ]
                    ),
                    col=4,
                ),
                card(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.VISIBILITY_OUTLINED, color=GREEN),
                            ft.Column(
                                [
                                    label("SESSION MODE", 9, MUTED, True),
                                    label("Wheating running" if self.wheating_running else
                                          "Orders running" if self.orders_running else "Ready", 17, INK, True),
                                ],
                                spacing=4,
                            ),
                        ]
                    ),
                    col=4,
                ),
            ],
            spacing=14,
            run_spacing=14,
        )
        roadmap = ft.Row(
            [
                self.module(
                    "Order planning",
                    "Goods & inventory requirements",
                    ft.Icons.LOCAL_SHIPPING_OUTLINED,
                ),
                self.module(
                    "Crops & animals", "Growing, tending & collecting", ft.Icons.GRASS_ROUNDED
                ),
                self.module("Scheduling", "Long-term production planning", ft.Icons.FACTORY_OUTLINED),
            ],
            spacing=14,
        )
        return ft.Column(
            [
                self.heading(
                    "Farm workspace",
                    f"Welcome to {svc.settings.farm_name}.",
                    "Your connection, captures and project workspace in one place.",
                    self.button(
                        "Manage device",
                        ft.Icons.ARROW_FORWARD_ROUNDED,
                        lambda e: self.navigate("devices"),
                    ),
                ),
                summary,
                preview_card,
                ft.Row(
                    [
                        label("Built for what comes next", 15, INK, True, expand=True),
                        label("Gameplay features · planned", 11, MUTED),
                    ]
                ),
                roadmap,
                self.notice(),
            ],
            spacing=18,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def module(self, title, description, icon):
        return ft.Container(
            ft.Row(
                [
                    ft.Icon(icon, color=GOLD, size=26),
                    ft.Column(
                        [label(title, 13, INK, True), label(description, 11, MUTED)],
                        spacing=5,
                        expand=True,
                    ),
                ]
            ),
            expand=True,
            padding=16,
            bgcolor="#F1EEE3",
            border_radius=12,
        )

    def devices(self):
        s = self.services.settings
        self.adb_field = ft.TextField(
            label="ADB executable",
            value=s.adb_path,
            hint_text="Choose a discovered instance or browse to adb.exe" if os.name == "nt" else "/path/to/platform-tools/adb",
            expand=True,
            text_size=13,
            border_color=LINE,
        )
        self.endpoint_field = ft.TextField(
            label="Local ADB endpoint", value=s.endpoint, width=260, text_size=13, border_color=LINE
        )
        rows = []
        instance_rows = []
        for instance in self.services.instances:
            def select_wheating(e, key=instance.key):
                selected = list(self.services.settings.wheating_instances)
                if e.control.value and key not in selected:
                    selected.append(key)
                elif not e.control.value and key in selected:
                    selected.remove(key)
                self.services.select_wheating_instances(selected)

            async def connect_known(e, key=instance.key):
                await self.stop_preview()
                await self.job('Connecting emulator', lambda: self.services.connect_instance(key), 'connection')

            instance_rows.append(ft.Row([
                ft.Checkbox(label='Wheat', value=instance.key in s.wheating_instances,
                            on_change=select_wheating, disabled=self.wheating_running),
                label(f'{instance.emulator} · {instance.display_name} · {instance.endpoint}', expand=True),
                self.button('Connect instance', ft.Icons.LINK_ROUNDED, connect_known),
            ]))
        for device in self.services.devices:

            async def use(e, serial=device.serial):
                await self.select_device(serial)

            rows.append(
                ft.Container(
                    ft.Row(
                        [
                            ft.Icon(ft.Icons.PHONE_ANDROID_ROUNDED, color=GREEN),
                            ft.Column(
                                [
                                    label(device.model or "Android instance", 14, INK, True),
                                    label(device.serial, 12, MUTED),
                                ],
                                expand=True,
                                spacing=5,
                            ),
                            badge(device.state, GREEN if device.online else GOLD),
                            self.button(
                                "Selected"
                                if device.serial == self.services.connected_serial
                                else "Use device",
                                ft.Icons.LINK_ROUNDED,
                                use,
                                True,
                                not device.online
                                or device.serial == self.services.connected_serial,
                            ),
                        ],
                        spacing=16,
                    ),
                    padding=16,
                    border=ft.Border.all(1, LINE),
                    border_radius=10,
                )
            )
        self.disconnect_button = self.button(
            "Disconnect session", ft.Icons.LINK_OFF_ROUNDED, self.disconnect,
            disabled=not bool(self.services.connected_serial),
        )
        if (self.orders_running or self.board_testing or self.wheating_running) and self.services.connected_serial:
            self.disconnect_button.disabled = False
            self.disconnect_button.bgcolor = PALE
            self.disconnect_button.color = GREEN

        def choose_wheat(keys):
            self.services.select_wheating_instances(keys)
            self.navigate('devices')

        return ft.Column(
            [
                self.heading(
                    "Connection",
                    "Connect your farm.",
                    "Connect BlueStacks or MuMu Player through ADB.",
                    self.button("Discover", ft.Icons.SEARCH_ROUNDED, self.discover),
                ),
                card(
                    ft.Column(
                        [
                            label("1  ·  Choose the connection", 17, INK, True),
                            label('Select Wheat for each farm to automate together. Start them from Overview.', 12, MUTED),
                            ft.Row([
                                self.button('Select all for wheat', ft.Icons.CHECK_BOX_OUTLINED,
                                    lambda e: choose_wheat([i.key for i in self.services.instances]),
                                    disabled=self.wheating_running or not self.services.instances),
                                self.button('Clear selection', ft.Icons.CHECK_BOX_OUTLINE_BLANK,
                                    lambda e: choose_wheat([]), disabled=self.wheating_running),
                            ]),
                            *instance_rows,
                            ft.Row(
                                [
                                    self.adb_field,
                                    self.button(
                                        "Browse", ft.Icons.FOLDER_OPEN_OUTLINED, self.browse_adb
                                    ),
                                ]
                            ),
                            ft.Row(
                                [
                                    self.button(
                                        "Apply path & refresh",
                                        ft.Icons.REFRESH_ROUNDED,
                                        self.apply_path,
                                        True,
                                    ),
                                    self.endpoint_field,
                                    self.button(
                                        "Connect endpoint",
                                        ft.Icons.LINK_ROUNDED,
                                        self.connect_endpoint,
                                    ),
                                ],
                                wrap=True,
                            ),
                            label(
                                "BlueStacks: enable Android Debug Bridge in Settings → Advanced.\nMuMu: start the Android device, then Discover and Connect instance. For manual setup, use its reported ADB port.",
                                12,
                                MUTED,
                            ),
                        ],
                        spacing=18,
                    )
                ),
                card(
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    label(
                                        "2  ·  Select an online instance",
                                        17,
                                        INK,
                                        True,
                                        expand=True,
                                    ),
                                    self.button(
                                        "Refresh", ft.Icons.REFRESH_ROUNDED, self.refresh_devices
                                    ),
                                ]
                            ),
                            *rows,
                        ]
                        if rows
                        else [
                            label("2  ·  Select an online instance", 17, INK, True),
                            label(
                                "No devices found. Start BlueStacks or MuMu, enable ADB, then Discover or connect its local endpoint.",
                                13,
                                MUTED,
                            ),
                        ],
                        spacing=14,
                    )
                ),
                self.disconnect_button,
                self.notice(),
                label(
                    "Selecting a device only captures its screen. Use Test quest board on Overview to open and verify the board.",
                    12,
                    MUTED,
                ),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def captures(self):
        files = sorted(self.services.data.captures_dir.glob("*.png"), reverse=True)[:60]
        tiles = []
        for path in files:

            async def view(e, path=path):
                await self.open_capture(path)

            tiles.append(
                ft.Container(
                    ft.Column(
                        [
                            ft.Container(
                                ft.Image(src=str(path), fit=ft.BoxFit.CONTAIN, height=140),
                                bgcolor=PALE,
                                border_radius=8,
                                alignment=ft.Alignment.CENTER,
                            ),
                            label(
                                path.stem, 11, MUTED, max_lines=1, overflow=ft.TextOverflow.ELLIPSIS
                            ),
                            ft.TextButton(
                                "View capture", icon=ft.Icons.OPEN_IN_FULL_ROUNDED, on_click=view
                            ),
                        ],
                        spacing=10,
                    ),
                    width=270,
                    bgcolor=WHITE,
                    border=ft.Border.all(1, LINE),
                    border_radius=12,
                    padding=14,
                )
            )
        return ft.Column(
            [
                self.heading(
                    "Screenshots",
                    "Capture library",
                    "Original-resolution PNGs saved from your selected device.",
                    self.button(
                        "Open folder",
                        ft.Icons.FOLDER_OPEN_OUTLINED,
                        lambda e: self.open_path(self.services.data.captures_dir),
                    ),
                ),
                label(
                    f"Showing {len(files)} most recent captures. Open folder for all files.",
                    12,
                    MUTED,
                ),
                ft.Row(tiles, wrap=True, spacing=16, run_spacing=16)
                if files
                else self.empty_captures(),
                self.notice(),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def empty_captures(self):
        return card(
            ft.Column(
                [
                    ft.Icon(ft.Icons.ADD_PHOTO_ALTERNATE_OUTLINED, size=48, color=GREEN),
                    label("Your first capture starts here.", 20, INK, True),
                    label(
                        "Connect a device, capture its screen, then choose Save PNG on Overview.",
                        13,
                        MUTED,
                    ),
                    self.button(
                        "Go to Overview",
                        ft.Icons.ARROW_FORWARD_ROUNDED,
                        lambda e: self.navigate("overview"),
                    ),
                ],
                spacing=18,
            )
        )

    def activity(self):
        self.search_field = ft.TextField(
            label="Search activity",
            value=self.search,
            expand=True,
            text_size=13,
            on_submit=self.filter_activity,
            border_color=LINE,
        )
        records = self.services.data.recent_activity(query=self.search)
        rows = [
            ft.Container(
                ft.Row(
                    [
                        label(item.timestamp.replace("T", " ")[:19], 11, MUTED, width=145),
                        badge(item.level, "#AF4A3B" if item.level == "ERROR" else GREEN),
                        ft.Column(
                            [
                                label(item.message, 12, INK, selectable=True),
                                label(item.category, 10, MUTED),
                            ],
                            expand=True,
                            spacing=4,
                        ),
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                padding=12,
                border=ft.Border.only(bottom=ft.BorderSide(1, LINE)),
            )
            for item in records
        ]
        return ft.Column(
            [
                self.heading(
                    "Session history",
                    "Activity",
                    "Connection, captures and settings, recorded locally.",
                    self.button("Export JSON", ft.Icons.DOWNLOAD_ROUNDED, self.export_activity),
                ),
                ft.Row(
                    [
                        self.search_field,
                        self.button("Search", ft.Icons.SEARCH_ROUNDED, self.filter_activity),
                        self.button(
                            "Refresh", ft.Icons.REFRESH_ROUNDED, lambda e: self.navigate("activity")
                        ),
                    ]
                ),
                card(ft.Column(rows or [label("No matching activity.", 13, MUTED)], spacing=0)),
                self.notice(),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def settings(self):
        s = self.services.settings
        self.farm_field = ft.TextField(
            label="Workspace name",
            value=s.farm_name,
            max_length=80,
            text_size=13,
            border_color=LINE,
        )
        self.interval_field = ft.TextField(
            label="Preview interval (seconds)",
            value=str(s.preview_interval_seconds),
            width=280,
            text_size=13,
            border_color=LINE,
        )
        return ft.Column(
            [
                self.heading(
                    "Preferences",
                    "Make it your workspace.",
                    "Settings are saved on this computer and restored when you return.",
                ),
                card(
                    ft.Column(
                        [
                            label("Workspace", 17, INK, True),
                            self.farm_field,
                            self.interval_field,
                            label(
                                "Live preview refreshes every 1–30 seconds, plus capture time. It starts only when you enable it.\nFrames stay in memory until you save a PNG.",
                                12,
                                MUTED,
                            ),
                            self.button(
                                "Save settings", ft.Icons.CHECK_ROUNDED, self.save_settings, True
                            ),
                        ],
                        spacing=18,
                    )
                ),
                card(
                    ft.Column(
                        [
                            label("Local data", 17, INK, True),
                            label(str(self.services.data.root), 12, MUTED, selectable=True),
                            self.button(
                                "Open data folder",
                                ft.Icons.FOLDER_OPEN_OUTLINED,
                                lambda e: self.open_path(self.services.data.root),
                            ),
                        ],
                        spacing=14,
                    )
                ),
                card(
                    ft.Column(
                        [
                            ft.Row(
                                [
                                    label("About this build", 17, INK, True, expand=True),
                                    badge(f"v{__version__}"),
                                ]
                            ),
                            label(
                                "Python desktop application · Flet interface · direct ADB connection",
                                13,
                                MUTED,
                            ),
                            label(
                                "Start orders runs a bounded truck-order session with a Stop orders control. Test quest board remains a separate diagnostic that taps once and verifies the panel. Sessions stop after 10 deliveries or 20 minutes. Resource support follows the observed reference set; unknown views stop. No fixed farm positions are stored.",
                                13,
                                MUTED,
                            ),
                        ],
                        spacing=14,
                    )
                ),
                self.notice(),
            ],
            spacing=20,
            scroll=ft.ScrollMode.AUTO,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    async def job(self, title: str, work: Callable[[], Any], category="app", after=None):
        if self.busy or self.closing:
            return
        task = asyncio.current_task()
        if task:
            self.jobs.add(task)
        self.busy = True
        self.last_message, self.message_error = f"{title}…", False
        self.navigate(self.route)
        try:
            result = await asyncio.to_thread(work)
            if not self.closing:
                if after:
                    after(result)
                if isinstance(result, BoardTestResult) or category in {"orders", "wheating"}:
                    self.last_message = result.message
                    self.message_error = not result.success and result.status != "cancelled"
                else:
                    self.last_message = (
                        str(result) if isinstance(result, (str, Path)) else f"{title} complete."
                    )
                    self.services.data.log("INFO", self.last_message, category)
        except Exception as exc:
            if not self.closing:
                self.last_message, self.message_error = str(exc), True
                self.services.data.log("ERROR", str(exc), category)
        finally:
            self.busy = False
            if task:
                self.jobs.discard(task)
            self.navigate(self.route)

    async def discover(self, e=None):
        await self.job("Discovering emulators", self.services.discover, "connection")

    async def refresh_devices(self, e=None):
        await self.stop_preview()
        await self.job("Refreshing devices", self.services.refresh_devices, "connection")

    async def apply_path(self, e=None):
        value = (self.adb_field.value or "").strip().strip('"')
        await self.stop_preview()

        def apply():
            if not Path(value).is_file():
                raise ValueError("Choose an existing ADB executable (HD-Adb.exe, adb.exe, or adb_server.exe).")
            self.services.disconnect()
            self.services.save_settings(replace(self.services.settings, adb_path=value))
            return self.services.refresh_devices()

        await self.job("Applying ADB path", apply, "connection")

    async def connect_endpoint(self, e=None):
        value = self.endpoint_field.value or ""
        await self.stop_preview()
        await self.job(
            "Connecting local endpoint", lambda: self.services.connect_endpoint(value), "connection"
        )

    async def select_device(self, serial):
        await self.stop_preview()

        def show(result):
            self.reference_shown = False
            self.route = "overview"

        await self.job(
            "Connecting device", lambda: self.services.select_device(serial), "connection", show
        )

    async def capture_screen(self, e=None):
        await self.job(
            "Capturing device screen",
            self.services.capture,
            "capture",
            lambda result: setattr(self, "reference_shown", False),
        )

    async def test_quest_board(self, e=None):
        if self.board_testing:
            if self.board_cancel:
                self.board_cancel.set()
            self.services.cancel_board_test()
            return
        if self.busy or self.closing or self.orders_running or self.wheating_running:
            return
        self.board_testing = True
        self.board_cancel = threading.Event()
        try:
            await self.stop_preview()
            await self.job(
                "Testing quest board",
                lambda: self.services.test_quest_board(cancel_event=self.board_cancel),
                "quest_board",
                lambda result: setattr(self, "reference_shown", False),
            )
        finally:
            self.board_testing = False
            self.board_cancel = None
            self.navigate(self.route)

    async def start_orders(self, e=None):
        if self.orders_running:
            if self.order_cancel:
                self.order_cancel.set()
            self.services.cancel_orders()
            self.order_message = "Stopping orders… Waiting for the current action to finish."
            self.navigate(self.route)
            return
        if self.busy or self.closing or self.board_testing or self.wheating_running:
            return
        self.orders_running = True
        self.order_cancel = threading.Event()
        self.order_deliveries = 0
        self.order_message = "Starting order session…"
        loop = asyncio.get_running_loop()
        updates: asyncio.Queue = asyncio.Queue(maxsize=1)
        finalized = False

        def enqueue(update):
            if updates.full():
                updates.get_nowait()
            updates.put_nowait(update)

        def progress(update: dict):
            # The runner never touches a Flet control from its worker thread.
            loop.call_soon_threadsafe(enqueue, dict(update))

        async def show_progress():
            while True:
                update = await updates.get()
                if update is None:
                    return
                if self.closing or finalized:
                    continue
                if not self.order_cancel or not self.order_cancel.is_set():
                    self.order_message = update.get("message", self.order_message)
                self.order_deliveries = int(update.get("deliveries", self.order_deliveries))
                self.last_message, self.message_error = self.order_message, False
                if self.route == "overview":
                    self.order_status.value = self.order_message
                    self.order_count.value = (
                        f"Delivered {self.order_deliveries}/{self.order_limit} · "
                        f"{self.order_seconds // 60} min limit"
                    )
                    if update.get("frame") and not self.reference_shown:
                        frame = update["frame"]
                        self.preview_image.src = frame.png
                        self.frame_info.value = self.frame_description()
                        self.resolution_text.value = f"{frame.width} × {frame.height}"
                    self.page.update()

        def finished(result):
            nonlocal finalized
            finalized = True
            self.reference_shown = False
            self.order_message = result.message
            self.order_deliveries = result.deliveries

        consumer = asyncio.create_task(show_progress())
        try:
            await self.stop_preview()
            self.reference_shown = False
            await self.job(
                "Starting orders",
                lambda: self.services.start_orders(
                    cancel_event=self.order_cancel, progress=progress,
                    max_deliveries=self.order_limit, max_seconds=self.order_seconds,
                ),
                "orders", finished,
            )
            if self.message_error:
                self.order_message = self.last_message
        finally:
            finalized = True
            enqueue(None)
            await consumer
            self.orders_running = False
            self.order_cancel = None
            self.navigate(self.route)

    async def save_capture(self, e=None):
        await self.job("Saving capture", self.services.save_frame, "capture")

    async def disconnect(self, e=None):
        if self.orders_running or self.board_testing or self.wheating_running:
            if self.wheating_cancel:
                self.wheating_cancel.set()
            if self.order_cancel:
                self.order_cancel.set()
            if self.board_cancel:
                self.board_cancel.set()
            self.services.disconnect()
            self.last_message = "Disconnected. Active game work was stopped."
            self.navigate(self.route)
            return
        await self.stop_preview()
        await self.job("Disconnecting session", self.services.disconnect, "connection")

    async def browse_adb(self, e=None):
        result = await ft.FilePicker().pick_files(
            dialog_title="Select emulator ADB executable" if os.name == "nt" else "Select ADB executable",
            allow_multiple=False,
            file_type=ft.FilePickerFileType.CUSTOM if os.name == "nt" else ft.FilePickerFileType.ANY,
            allowed_extensions=["exe"] if os.name == "nt" else None,
        )
        if result and result[0].path and self.route == "devices":
            self.adb_field.value = result[0].path
            self.adb_field.update()

    async def save_settings(self, e=None):
        name, interval = self.farm_field.value or "", self.interval_field.value or ""

        def save():
            self.services.save_settings(
                replace(
                    self.services.settings, farm_name=name, preview_interval_seconds=float(interval)
                )
            )
            return "Settings saved."

        await self.job("Saving settings", save, "settings")

    async def export_activity(self, e=None):
        await self.job("Exporting activity", self.services.data.export_activity, "activity")

    async def filter_activity(self, e=None):
        self.search = self.search_field.value or ""
        self.navigate("activity")

    def open_path(self, path):
        try:
            os.startfile(str(path))
        except OSError as exc:
            self.last_message, self.message_error = str(exc), True
            self.navigate(self.route)

    async def open_capture(self, path):
        self.page.show_dialog(
            ft.AlertDialog(
                title=label(path.name, 13, INK, True),
                content=ft.Container(
                    ft.Image(src=str(path), fit=ft.BoxFit.CONTAIN), width=950, height=510
                ),
                actions=[ft.TextButton("Close", on_click=lambda e: self.page.pop_dialog())],
            )
        )

    def toggle_source(self, e=None):
        self.reference_shown = not self.reference_shown
        self.navigate(self.route)

    def frame_description(self):
        f = self.services.frame
        if not f:
            return "No capture yet"
        return f"{f.width} × {f.height} · {self.services.frame_serial} · captured {f.captured_at.replace('T', ' ')[:19]} UTC"

    async def toggle_preview(self, e=None):
        if self.preview_running:
            await self.stop_preview()
        elif self.services.connected_serial and not self.busy and not self.orders_running and not self.wheating_running:
            self.preview_running = True
            self.preview_stop.clear()
            self.reference_shown = False
            self.preview_task = asyncio.create_task(self.preview_loop())
            self.services.data.log("INFO", "Live preview started.", "capture")
        self.navigate(self.route)

    async def preview_loop(self):
        try:
            while self.preview_running and not self.closing:
                try:
                    await asyncio.wait_for(
                        self.preview_stop.wait(), self.services.settings.preview_interval_seconds
                    )
                except TimeoutError:
                    pass
                if not self.preview_running or self.closing:
                    break
                if self.busy:
                    continue
                self.busy = True
                try:
                    await asyncio.to_thread(self.services.capture)
                    if self.route == "overview" and not self.reference_shown and not self.closing:
                        self.preview_image.src = self.services.frame.png
                        self.frame_info.value = self.frame_description()
                        self.resolution_text.value = (
                            f"{self.services.frame.width} × {self.services.frame.height}"
                        )
                        self.page.update(self.preview_image, self.frame_info, self.resolution_text)
                except Exception as exc:
                    self.preview_running = False
                    self.last_message, self.message_error = f"Preview stopped: {exc}", True
                    if not self.closing:
                        self.services.data.log("ERROR", self.last_message, "capture")
                finally:
                    self.busy = False
            if not self.closing:
                self.navigate(self.route)
        except asyncio.CancelledError:
            raise

    async def stop_preview(self):
        self.preview_running = False
        self.preview_stop.set()
        if self.preview_task and not self.preview_task.done():
            # Do not cancel to_thread: wait until the owned ADB operation finishes.
            await self.preview_task
        self.preview_task = None

    async def shutdown(self):
        if self.closing:
            return
        self.closing = True
        if self.wheating_cancel:
            self.wheating_cancel.set()
        self.preview_running = False
        if self.order_cancel:
            self.order_cancel.set()
        if self.board_cancel:
            self.board_cancel.set()
        self.services.disconnect()
        await self.stop_preview()
        other_jobs = [task for task in self.jobs if task is not asyncio.current_task()]
        if other_jobs:
            await asyncio.gather(*other_jobs, return_exceptions=True)
        self.services.close()

    async def _window_event(self, event):
        if event.type == ft.WindowEventType.CLOSE:
            await self.shutdown()
            await self.page.window.destroy()

    async def _disconnect_event(self, event):
        await self.shutdown()


async def main(page: ft.Page):
    window = MainWindow(page)
    await window.initialize()


def run():
    log_root = (
        Path(
            os.environ.get("HAYDAY_DATA_DIR")
            or Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "HayDayAutomation"
        )
        / "logs"
    )
    log_root.mkdir(parents=True, exist_ok=True)
    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            RotatingFileHandler(
                log_root / "desktop.log", maxBytes=2_000_000, backupCount=2, encoding="utf-8"
            )
        ],
    )
    ft.run(main, assets_dir=str(REFERENCE.parent))


if __name__ == "__main__":
    run()
