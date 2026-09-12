# Single-file macOS setup

Copy **Setup Hay Day macOS.command** to the Mac. It contains the application
source and image assets; no other project files need to be copied.

In Terminal, type `/bin/bash ` (including the space), drag the setup file into
Terminal, and press Return. This works even when copying from Windows loses
the executable permission. If executable permission was preserved, the file
can also be opened from Finder. Follow macOS approval prompts if shown.

Requires an Apple Silicon Mac, macOS 13+, 8 GB RAM, at least 16 GB free disk
space, and internet access. Intel Macs are not supported by BlueStacks Air.
BlueStacks itself supports macOS 11+, but this project's OpenCV wheel needs 13+.

Setup automatically installs a private Python 3.12 runtime, the Python
libraries, Google Android Platform Tools, the Flet desktop client, and
BlueStacks Air if absent. It creates **~/Applications/Hay Day Automation.app**
and opens the app, BlueStacks and first-run instructions. BlueStacks may ask
for an administrator password. Homebrew, Xcode and a preinstalled Python
are not required. The first download can be large.

The user must complete game-store/account sign-in, install Hay Day inside
BlueStacks and enable Android Debug Bridge in BlueStacks Settings > Advanced.
Use the port displayed there to connect in the tool. These first-run actions
cannot be completed unattended by an installer without the user's account.

The Mac build uses Apple's local Vision OCR. Native Windows desktop popup
handling is Windows-specific. BlueStacks touch gestures and game automation
still require validation on a real Mac; this package was assembled on Windows.
Setup runs native OCR/asset checks and a desktop UI smoke test on the Mac
before activating the new installation.

App data and setup logs live under
`~/Library/Application Support/HayDayAutomation/`. Rerunning setup creates a
new release and preserves data and previous releases. Downloads are cached.
Use `--no-launch` to finish without opening apps, or `--help` for usage.

Developers can rebuild the one-file output after source changes:

```text
python scripts/build_macos_setup.py
```

`requirements-macos.lock` pins and hashes the full Python dependency set.
Runtime tools are downloaded from their publishers with pinned SHA-256
checks. This is an online bootstrap installer, not an offline app bundle or
an Apple-signed/notarized installer.

Sources checked when building:
- [BlueStacks Air requirements](https://support.bluestacks.com/hc/en-us/articles/32272913555597-System-specifications-for-installing-BlueStacks-Air)
- [BlueStacks package metadata](https://formulae.brew.sh/api/cask/bluestacks.json)
- [Android Platform Tools metadata](https://formulae.brew.sh/api/cask/android-platform-tools.json)
- [uv 0.12.13](https://github.com/astral-sh/uv/releases/tag/0.12.13)
- [Flet 0.86.0](https://github.com/flet-dev/flet/releases/tag/v0.86.0)
- [PyObjC Vision bindings](https://pyobjc.readthedocs.io/en/latest/apinotes/Vision.html)
