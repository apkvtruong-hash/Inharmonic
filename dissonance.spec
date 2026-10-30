# PyInstaller build for Dissonance Studio and Dissonance Quick.
# Build:  .\.venv\Scripts\python.exe -m PyInstaller dissonance.spec --noconfirm
# Output: dist\DissonanceStudio\DissonanceStudio.exe, dist\DissonanceQuick\DissonanceQuick.exe

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

EXCLUDES = ["matplotlib", "tkinter", "IPython", "jupyter", "pytest", "scipy", "pandas"]

APPS = [
    ("dissonance_studio.py", "DissonanceStudio"),
    ("dissonance_quick.py", "DissonanceQuick"),
]

common_datas = (collect_data_files("pyqtgraph") + collect_data_files("qtawesome")
                + collect_data_files("_sounddevice_data"))
common_hidden = (collect_submodules("OpenGL.platform") + collect_submodules("OpenGL.arrays")
                 + collect_submodules("pyqtgraph.opengl"))

# Qt modules the QML plugin sweep drags in but neither app uses (WebEngine alone is ~200 MB).
UNUSED_QT = ("webengine", "webview", "qt63d", "/qt3d", "\\qt3d", "qt6pdf", "quick\\pdf",
             "quick/pdf")


def _used(entries):
    return [e for e in entries if not any(s in e[0].lower() for s in UNUSED_QT)]


for script, name in APPS:
    a = Analysis(
        [script],
        datas=common_datas,
        hiddenimports=common_hidden,
        excludes=EXCLUDES,
        runtime_hooks=["pyinstaller_rthook.py"],
        noarchive=False,
    )
    a.binaries = _used(a.binaries)
    a.datas = _used(a.datas)
    pyz = PYZ(a.pure)
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=name,
        console=False,
        upx=False,
    )
    COLLECT(exe, a.binaries, a.datas, name=name, upx=False)
