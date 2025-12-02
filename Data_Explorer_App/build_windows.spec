# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# OPTIMIZATION: Exclude heavy modules not used by Dash/Plotly/GNN
excluded_modules = [
    'matplotlib', 'tkinter', 'tcl', 'tk', 'ipython', 'jupyter', 
    'notebook', 'nbconvert', 'nbformat', 'jedi', 'docutils', 
    'pygments', 'sqlite3', 'test', 'unittest', 'pydoc', 'email',
    # NEW: Exclude timezone and language databases to save thousands of files
    'pytz', 'babel',
    # NEW: Exclude unused data science/system libraries
    'sklearn', 'PIL', 'openpyxl', 'xlrd', 'xlsxwriter', 'sqlalchemy', 
    'lxml', 'curses', 'distutils', 'setuptools', 'pkg_resources'
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/assets', 'src/assets'),
        ('src/pages', 'src/pages'),
        # Include the model file
        ('src/utils/phantom_gnn_model.pth', 'src/utils')
    ],
    hiddenimports=[
        'dash',
        'dash_bootstrap_components',
        'pandas',
        'plotly',
        'scipy.spatial',
        # 'sklearn.neighbors', # Removed as sklearn is excluded
        # Add torch if your GNN uses it, otherwise remove
        'torch' 
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excluded_modules,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DataExplorer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, # Keep True so user sees "Starting..." message
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DataExplorer',
)
