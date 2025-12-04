# gui/main_window.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QFileDialog,
    QSplitter,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QMessageBox,
    QTabWidget,
    QCheckBox,
    QStatusBar,
    QInputDialog,
    QDialog,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calibration.calib_config import CalibConfig  # noqa: E402
import bhtom_api  # noqa: E402
from gui.bhtom_login_dialog import BHTOMLoginDialog  # noqa: E402


# ----------------------------------------------------------------------
# Matplotlib-based FITS viewer
# ----------------------------------------------------------------------
class FitsCanvas(FigureCanvasQTAgg):
    def __init__(self, parent: QWidget | None = None) -> None:
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self._image = None
        self._data: Optional[np.ndarray] = None
        self._cmap = "gray"

        self.fig.tight_layout()
        self.mpl_connect("motion_notify_event", self._on_motion)

        self.coord_callback = None  # type: ignore[assignment]

    # Called by MainWindow
    def show_fits(self, data: np.ndarray) -> None:
        self.ax.clear()
        self._data = data

        # Stretch using 2–98 percentile for something DS9-like
        vmin, vmax = np.percentile(data, (2, 98))
        self._image = self.ax.imshow(
            data,
            origin="lower",
            cmap=self._cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.draw()

    def set_cmap(self, cmap: str) -> None:
        self._cmap = cmap
        if self._image is not None:
            self._image.set_cmap(cmap)
            self.draw_idle()

    def zoom_to_fit(self) -> None:
        self.ax.autoscale(True)
        self.fig.tight_layout()
        self.draw_idle()

    # Internal: mouse move -> pixel value
    def _on_motion(self, event) -> None:
        if self._data is None or event.inaxes != self.ax:
            if self.coord_callback:
                self.coord_callback(None, None, None)
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if 0 <= y < self._data.shape[0] and 0 <= x < self._data.shape[1]:
            value = float(self._data[y, x])
            if self.coord_callback:
                self.coord_callback(x, y, value)
        else:
            if self.coord_callback:
                self.coord_callback(None, None, None)


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Calibration & FITS Viewer")
        self.resize(1400, 820)

        self.current_config_path: Optional[Path] = None
        self.current_config: Optional[CalibConfig] = None

        self.loaded_frames: List[tuple[Path, np.ndarray]] = []
        self.current_frame_index: int = 0
        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(500)
        self.blink_timer.timeout.connect(self._on_blink_timer)

        # BHTOM session
        self.bhtom_token: Optional[str] = None
        self.bhtom_username: Optional[str] = None

        # Calibrated frames ready for upload
        self.calibrated_files: List[Path] = []

        self._create_actions()
        self._create_menu_and_toolbar()
        self._create_status_bar()
        self._create_central_layout()
        self._restore_bhtom_session()
        self._load_config_profiles()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _create_actions(self) -> None:
        self.act_open = QAction("Open FITS…", self)
        self.act_open.setShortcut("Ctrl+O")

        self.act_quit = QAction("Quit", self)
        self.act_quit.setShortcut("Ctrl+Q")

    def _create_menu_and_toolbar(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        m_file.addAction(self.act_open)
        m_file.addSeparator()
        m_file.addAction(self.act_quit)

        self.toolbar = self.addToolBar("Main")
        self.toolbar.setMovable(False)
        self.toolbar.addAction(self.act_open)

        self.act_open.triggered.connect(self.open_fits_dialog)
        self.act_quit.triggered.connect(self.close)

    def _create_status_bar(self) -> None:
        sb = QStatusBar()
        self.setStatusBar(sb)
        self.lbl_status = QLabel("Ready")
        self.lbl_coords = QLabel("x: –, y: –, I: –")
        sb.addWidget(self.lbl_status)
        sb.addPermanentWidget(self.lbl_coords)

    def _create_central_layout(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # Left: viewer + basic controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        # Info bar
        info_box = QGroupBox("Frame info")
        info_layout = QFormLayout(info_box)
        self.lbl_file = QLabel("—")
        self.lbl_dim = QLabel("—")
        info_layout.addRow("File:", self.lbl_file)
        info_layout.addRow("Size:", self.lbl_dim)

        # Viewer
        self.canvas = FitsCanvas()
        self.canvas.coord_callback = self._update_coords

        self.nav_toolbar = NavigationToolbar2QT(self.canvas, self)
        self.nav_toolbar.setIconSize(self.nav_toolbar.iconSize() * 0.9)

        # Custom viewer controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_zoom_fit = QPushButton("Zoom to fit")
        self.btn_zoom_fit.setProperty("variant", "secondary")
        self.btn_zoom_fit.clicked.connect(self.canvas.zoom_to_fit)

        self.btn_blink = QCheckBox("Blink")
        self.btn_blink.stateChanged.connect(self._toggle_blink)

        self.cmb_cmap = QComboBox()
        self.cmb_cmap.addItems(["gray", "viridis", "magma", "plasma"])
        self.cmb_cmap.setCurrentText("gray")
        self.cmb_cmap.currentTextChanged.connect(self.canvas.set_cmap)

        controls_layout.addWidget(self.btn_zoom_fit)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(QLabel("Colormap:"))
        controls_layout.addWidget(self.cmb_cmap)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_blink)

        left_layout.addWidget(info_box)
        left_layout.addWidget(self.nav_toolbar)
        left_layout.addWidget(self.canvas)
        left_layout.addLayout(controls_layout)

        # Right: tabs with calibration & integrations
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(8)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_calibration(), "Calibration")
        tabs.addTab(self._build_tab_config(), "Config & Paths")
        tabs.addTab(self._build_tab_integrations(), "Integrations")

        right_layout.addWidget(tabs)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _build_tab_calibration(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Config / input
        box_run = QGroupBox("Run calibration")
        form = QFormLayout(box_run)

        self.cmb_config = QComboBox()
        form.addRow("Config profile:", self.cmb_config)

        self.txt_input_dir = QLineEdit()
        self.txt_input_dir.setPlaceholderText("Directory with raw FITS…")
        btn_browse = QPushButton("Browse…")
        btn_browse.setProperty("variant", "secondary")
        btn_browse.clicked.connect(self._browse_input_dir)

        h = QHBoxLayout()
        h.addWidget(self.txt_input_dir)
        h.addWidget(btn_browse)
        form.addRow("Raw data directory:", h)

        # Buttons
        btn_prepare = QPushButton("1. Prepare file lists")
        btn_prepare.setProperty("variant", "secondary")
        btn_prepare.clicked.connect(self._run_prepare_lists)

        btn_masters = QPushButton("2. Create master frames")
        btn_masters.setProperty("variant", "secondary")
        btn_masters.clicked.connect(self._run_create_masters)

        btn_full = QPushButton("3. Run full calibration")
        btn_full.clicked.connect(self._run_full_calibration)

        # Upload to BHTOM – appears only when we detect calibrated files
        self.btn_upload_bhtom = QPushButton("Upload calibrated frames to BHTOM")
        self.btn_upload_bhtom.setProperty("variant", "primary")
        self.btn_upload_bhtom.setVisible(False)
        self.btn_upload_bhtom.clicked.connect(self._on_upload_calibrated_to_bhtom)

        layout.addWidget(box_run)
        layout.addWidget(btn_prepare)
        layout.addWidget(btn_masters)
        layout.addWidget(btn_full)
        layout.addWidget(self.btn_upload_bhtom)
        layout.addStretch()

        return w

    def _build_tab_config(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        box_obs = QGroupBox("Instrument")
        f_obs = QFormLayout(box_obs)
        self.lbl_obs_name = QLabel("—")
        self.lbl_telescope = QLabel("—")
        self.lbl_camera = QLabel("—")
        f_obs.addRow("Observatory:", self.lbl_obs_name)
        f_obs.addRow("Telescope:", self.lbl_telescope)
        f_obs.addRow("Camera:", self.lbl_camera)

        box_paths = QGroupBox("Data structure (from config)")
        f_paths = QFormLayout(box_paths)
        self.lbl_work_dir = QLabel("—")
        self.lbl_results_dir = QLabel("—")
        self.lbl_aux_dir = QLabel("—")
        f_paths.addRow("Working dir:", self.lbl_work_dir)
        f_paths.addRow("Results dir:", self.lbl_results_dir)
        f_paths.addRow("Aux dir:", self.lbl_aux_dir)

        layout.addWidget(box_obs)
        layout.addWidget(box_paths)
        layout.addStretch()
        return w

    def _build_tab_integrations(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # BHTOM
        box_bhtom = QGroupBox("BHTOM")
        v_bhtom = QVBoxLayout(box_bhtom)
        lbl = QLabel("Connect your BHTOM account to upload calibrated files.")
        lbl.setWordWrap(True)

        self.btn_bhtom_connect = QPushButton("Connect to BHTOM…")
        self.btn_bhtom_connect.clicked.connect(self._on_bhtom_connect_clicked)

        self.lbl_bhtom_status = QLabel("Not connected")
        self.lbl_bhtom_status.setObjectName("sectionTitle")

        v_bhtom.addWidget(lbl)
        v_bhtom.addSpacing(4)
        v_bhtom.addWidget(self.btn_bhtom_connect)
        v_bhtom.addSpacing(6)
        v_bhtom.addWidget(self.lbl_bhtom_status)

        # Astrometry.net (future)
        box_ast = QGroupBox("Astrometry.net (future)")
        v_ast = QVBoxLayout(box_ast)
        lbl_ast = QLabel(
            "Uses plate-solver parameters from the selected config.\n"
            "In a later step, this tab will let you submit solved frames "
            "directly to astrometry.net."
        )
        lbl_ast.setWordWrap(True)
        v_ast.addWidget(lbl_ast)

        layout.addWidget(box_bhtom)
        layout.addWidget(box_ast)
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Config handling
    # ------------------------------------------------------------------
    def _load_config_profiles(self) -> None:
        self.cmb_config.blockSignals(True)
        self.cmb_config.clear()

        if CONFIG_DIR.exists():
            ini_files = sorted(CONFIG_DIR.glob("*.ini"))
        else:
            ini_files = []

        for path in ini_files:
            self.cmb_config.addItem(path.name, path)

        self.cmb_config.blockSignals(False)
        self.cmb_config.currentIndexChanged.connect(self._on_config_changed)

        if ini_files:
            self.cmb_config.setCurrentIndex(0)
            self._on_config_changed(0)
        else:
            self.lbl_status.setText("No .ini files found in configs/")

    @Slot(int)
    def _on_config_changed(self, index: int) -> None:
        path = self.cmb_config.itemData(index)
        if not isinstance(path, Path):
            self.current_config_path = None
            self.current_config = None
            return

        self.current_config_path = path
        self.current_config = CalibConfig(str(path))

        cfg = self.current_config

        self.lbl_obs_name.setText(str(cfg.get("GENERAL", "observatory_name")))
        self.lbl_telescope.setText(str(cfg.get("GENERAL", "telescope")))
        self.lbl_camera.setText(str(cfg.get("GENERAL", "camera")))

        self.lbl_work_dir.setText(str(cfg.get("DATA_STRUCTURE", "working_dir")))
        self.lbl_results_dir.setText(str(cfg.get("DATA_STRUCTURE", "results_dir")))
        self.lbl_aux_dir.setText(str(cfg.get("DATA_STRUCTURE", "results_aux_dir")))

        self.lbl_status.setText(f"Config loaded: {path.name}")

    # ------------------------------------------------------------------
    # FITS loading / viewer
    # ------------------------------------------------------------------
    def open_fits_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open FITS files",
            "",
            "FITS files (*.fits *.fit);;All files (*)",
        )
        if not paths:
            return

        self.loaded_frames.clear()
        for p in paths:
            try:
                arr = fits.getdata(p).astype(np.float32)
                if arr.ndim != 2:
                    raise ValueError("Only 2D images are supported")
                self.loaded_frames.append((Path(p), arr))
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(self, "FITS error", f"Could not open {p}:\n{e}")

        if not self.loaded_frames:
            return

        self.current_frame_index = 0
        self._show_current_frame()

    def _show_current_frame(self) -> None:
        path, data = self.loaded_frames[self.current_frame_index]
        self.canvas.show_fits(data)
        self.lbl_file.setText(path.name)
        self.lbl_dim.setText(f"{data.shape[1]} × {data.shape[0]}")
        self.lbl_status.setText(
            f"Showing frame {self.current_frame_index + 1}/{len(self.loaded_frames)}"
        )

    def _on_blink_timer(self) -> None:
        if not self.loaded_frames:
            return
        self.current_frame_index = (self.current_frame_index + 1) % len(self.loaded_frames)
        self._show_current_frame()

    def _toggle_blink(self, state: int) -> None:
        if state == Qt.Checked and len(self.loaded_frames) > 1:
            self.blink_timer.start()
        else:
            self.blink_timer.stop()

    def _update_coords(self, x: Optional[int], y: Optional[int], val: Optional[float]) -> None:
        if x is None or y is None or val is None:
            self.lbl_coords.setText("x: –, y: –, I: –")
        else:
            self.lbl_coords.setText(f"x: {x}, y: {y}, I: {val:.2f}")

    # ------------------------------------------------------------------
    # Calibration buttons (hook into your pipeline)
    # ------------------------------------------------------------------
    def _get_input_dir(self) -> Optional[Path]:
        text = self.txt_input_dir.text().strip()
        if not text:
            QMessageBox.warning(
                self, "Input directory", "Please select a directory with raw FITS files."
            )
            return None
        path = Path(text)
        if not path.is_dir():
            QMessageBox.warning(self, "Input directory", f"'{path}' is not a directory.")
            return None
        return path

    def _browse_input_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select raw FITS directory")
        if dir_path:
            self.txt_input_dir.setText(dir_path)

    def _ensure_config_and_dir(self) -> Optional[tuple[CalibConfig, Path]]:
        if self.current_config is None or self.current_config_path is None:
            QMessageBox.warning(
                self, "Config", "Please select a valid configuration profile first."
            )
            return None
        input_dir = self._get_input_dir()
        if input_dir is None:
            return None
        return self.current_config, input_dir

    def _run_prepare_lists(self) -> None:
        res = self._ensure_config_and_dir()
        if res is None:
            return
        _, input_dir = res
        from calibration import calib_prep_lists

        base_list = calib_prep_lists.build_lists_from_directory(str(input_dir))
        self.lbl_status.setText(f"Prepared lists from {input_dir.name} → {base_list}")
        QMessageBox.information(
            self, "Lists created", f"Calibration lists created for '{input_dir.name}'."
        )

    def _run_create_masters(self) -> None:
        res = self._ensure_config_and_dir()
        if res is None:
            return
        cfg, input_dir = res

        # Here we call your command-line style scripts programmatically.
        # Adjust list names to match calib_prep_lists SUFFIXES if needed.
        base_name = input_dir.name
        list_in = Path(f"{base_name}.lst")
        list_b = Path(f"{base_name}-b.lst")

        from calibration.mkmasterbias import create_master_bias  # type: ignore
        from calibration.mkmasterdark import find_dark_frames, make_master_dark  # type: ignore
        from calibration.mkmasterflats import process_flats  # type: ignore
        from calibration.calib_prep_lists import read_list_file  # type: ignore

        if not list_in.is_file():
            QMessageBox.warning(
                self,
                "Master frames",
                f"List file '{list_in}' not found. Run 'Prepare file lists' first.",
            )
            return

        if not list_b.is_file():
            QMessageBox.warning(
                self,
                "Master frames",
                f"List file '{list_b}' not found. Run 'Prepare file lists' first.",
            )
            return

        # Master bias
        files = read_list_file(list_in)  # list[str] of filenames
        create_master_bias(files)  # uses config inside that module

        # Master dark
        dark_candidates = read_list_file(list_b)
        dark_files = find_dark_frames(dark_candidates)
        if dark_files:
            make_master_dark(dark_files, "masterdark.fits")

        # Master flats
        flats_files = read_list_file(list_b)
        process_flats(flats_files)

        QMessageBox.information(
            self,
            "Master frames",
            "Master bias/dark/flats have been created (see work/ & results/aux).",
        )

    def _run_full_calibration(self) -> None:
        res = self._ensure_config_and_dir()
        if res is None:
            return
        cfg, input_dir = res

        # TODO: hook your real full calibration pipeline here.
        QMessageBox.information(
            self,
            "Full calibration",
            "Here you will hook your full calibration pipeline\n"
            "(bias + dark + flats + science calibration).",
        )

        # After the pipeline, try to discover calibrated frames
        self._find_calibrated_frames()

    def _find_calibrated_frames(self) -> None:
        """Heuristic: look for calibrated FITS files in results_dir or its 'calibrated' subfolder."""
        self.calibrated_files = []
        if self.current_config is None:
            self.btn_upload_bhtom.setVisible(False)
            return

        results_dir_str = self.current_config.get("DATA_STRUCTURE", "results_dir")
        results_dir = Path(results_dir_str)
        if not results_dir.is_absolute():
            results_dir = ROOT_DIR / results_dir

        candidates: List[Path] = []

        calib_dir = results_dir / "calibrated"
        if calib_dir.is_dir():
            candidates.extend(sorted(calib_dir.glob("*.fit*")))
        elif results_dir.is_dir():
            candidates.extend(sorted(results_dir.glob("*.fit*")))

        self.calibrated_files = candidates
        has_files = bool(self.calibrated_files)
        self.btn_upload_bhtom.setVisible(has_files)

        if has_files:
            self.lbl_status.setText(
                f"Calibration completed: {len(self.calibrated_files)} calibrated frame(s) found."
            )
        else:
            self.lbl_status.setText("Calibration completed, but no calibrated frames were found.")

    # ------------------------------------------------------------------
    # BHTOM integration – session + login
    # ------------------------------------------------------------------
    def _restore_bhtom_session(self) -> None:
        """Restore BHTOM session from saved credentials, if any."""
        username, token = bhtom_api.load_credentials()
        if username and token:
            self.bhtom_username = username
            self.bhtom_token = token
            self.lbl_status.setText(f"BHTOM: restored session as {username}")
        # lbl_bhtom_status is created later in _build_tab_integrations,
        # but we may call _restore_bhtom_session() before that.
        # So we update UI after tabs are built, in __init__.
        # Here we just ensure attributes exist when UI is ready.
        # _update_bhtom_ui() is called in _build_tab_integrations via __init__
        # after central widget is created.

    def _update_bhtom_ui(self) -> None:
        if getattr(self, "lbl_bhtom_status", None) is None:
            return
        if self.bhtom_token:
            self.lbl_bhtom_status.setText(f"Connected as {self.bhtom_username}")
            self.btn_bhtom_connect.setText("Disconnect")
        else:
            self.lbl_bhtom_status.setText("Not connected")
            self.btn_bhtom_connect.setText("Connect to BHTOM…")

    def _on_bhtom_connect_clicked(self) -> None:
        if self.bhtom_token:
            # Already connected → offer logout
            resp = QMessageBox.question(
                self,
                "Disconnect BHTOM",
                "You are currently connected to BHTOM.\n"
                "Do you want to disconnect this account?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if resp == QMessageBox.Yes:
                self.bhtom_token = None
                self.bhtom_username = None
                bhtom_api.clear_credentials()
                self._update_bhtom_ui()
                self.lbl_status.setText("BHTOM: disconnected")
            return

        # Not connected → open login dialog
        self._connect_bhtom()

    def _connect_bhtom(self) -> None:
        dlg = BHTOMLoginDialog(self)
        if dlg.exec() != QDialog.Accepted:
            # User cancelled or closed dialog
            return

        if not dlg.username or not dlg.token:
            # "Continue without login"
            return

        self.bhtom_username = dlg.username
        self.bhtom_token = dlg.token

        if dlg.remember_me:
            bhtom_api.save_credentials(dlg.username, dlg.token)
        else:
            bhtom_api.clear_credentials()

        self._update_bhtom_ui()
        self.lbl_status.setText("BHTOM: authenticated")

    def _ensure_bhtom_logged_in(self) -> bool:
        """Return True if we have a valid BHTOM token; otherwise prompt login."""
        if self.bhtom_token:
            return True
        self._connect_bhtom()
        return self.bhtom_token is not None

    # ------------------------------------------------------------------
    # Upload calibrated frames to BHTOM
    # ------------------------------------------------------------------
    def _on_upload_calibrated_to_bhtom(self) -> None:
        if not self.calibrated_files:
            QMessageBox.information(
                self,
                "No calibrated frames",
                "No calibrated frames were detected for upload.\n"
                "Run the full calibration first.",
            )
            return

        if not self._ensure_bhtom_logged_in():
            return

        # Preview list of files
        max_preview = 10
        file_lines = [f"- {p.name}" for p in self.calibrated_files[:max_preview]]
        if len(self.calibrated_files) > max_preview:
            file_lines.append(f"... and {len(self.calibrated_files) - max_preview} more")

        preview_text = (
            f"User: {self.bhtom_username}\n"
            f"Config: {self.current_config_path.name if self.current_config_path else '—'}\n\n"
            f"Ready to upload {len(self.calibrated_files)} calibrated frame(s) to BHTOM.\n\n"
            "Files:\n" + "\n".join(file_lines) + "\n\n"
            "Do you want to continue?"
        )

        resp = QMessageBox.question(
            self,
            "Upload calibrated frames to BHTOM",
            preview_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if resp != QMessageBox.Yes:
            return

        self._do_bhtom_upload()

    def _do_bhtom_upload(self) -> None:
        if not self.bhtom_token:
            return

        # Ask user for required metadata.
        target, ok = QInputDialog.getText(
            self, "Target name", "BHTOM target name:"
        )
        if not ok or not target.strip():
            return

        default_obs = self.lbl_obs_name.text()
        if default_obs == "—":
            default_obs = ""
        observatory, ok = QInputDialog.getText(
            self,
            "Observatory",
            "Observatory name in BHTOM:",
            text=default_obs,
        )
        if not ok or not observatory.strip():
            return

        filter_name, ok = QInputDialog.getText(
            self,
            "Filter",
            "Filter name:",
            text="GaiaSP/any",
        )
        if not ok or not filter_name.strip():
            return

        try:
            bhtom_api.upload_calibrated_files(
                self.calibrated_files,
                self.bhtom_token,
                target.strip(),
                observatory.strip(),
                filter_name.strip(),
            )
        except bhtom_api.BHTOMUploadError as exc:  # type: ignore[attr-defined]
            QMessageBox.critical(self, "Upload failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Upload finished",
            f"Uploaded {len(self.calibrated_files)} calibrated frame(s) to BHTOM.",
        )
        self.lbl_status.setText("BHTOM: upload finished")

    # After UI is fully built, make sure BHTOM status label matches session
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._update_bhtom_ui()
