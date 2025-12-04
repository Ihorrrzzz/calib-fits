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
    QApplication,
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
    QDialog,
    QInputDialog,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT_DIR / "configs"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from calibration.calib_config import CalibConfig  # noqa: E402
import bhtom_api  # noqa: E402
from calibration.calib_core import CalibrationPipeline  # noqa: E402
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

        # BHTOM auth
        self.bhtom_token: Optional[str] = None
        self.bhtom_username: Optional[str] = None

        # Remember last used BHTOM parameters for convenience
        self.last_bhtom_target: Optional[str] = None
        self.last_bhtom_observatory: Optional[str] = None
        self.last_bhtom_filter: Optional[str] = None

        # List of fully calibrated science frames (-bdf) for potential upload
        self.calibrated_files: List[Path] = []

        self._create_actions()
        self._create_menu_and_toolbar()
        self._create_status_bar()
        self._create_central_layout()
        self._load_config_profiles()

        # Try restore saved BHTOM session
        username, token = bhtom_api.load_credentials()
        if username and token:
            self.bhtom_username = username
            self.bhtom_token = token

        # Initialize BHTOM UI (button text + status label + upload enabled/disabled)
        self._refresh_bhtom_ui()

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

        layout.addWidget(box_run)
        layout.addWidget(btn_prepare)
        layout.addWidget(btn_masters)
        layout.addWidget(btn_full)
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

        # store button as an attribute so we can change its text
        self.btn_bhtom_connect = QPushButton()
        self.btn_bhtom_connect.clicked.connect(self._on_bhtom_connect_or_logout)

        self.lbl_bhtom_status = QLabel("Not connected")
        self.lbl_bhtom_status.setObjectName("sectionTitle")

        self.btn_upload_calibrated = QPushButton("Upload calibrated frames to BHTOM…")
        self.btn_upload_calibrated.setVisible(False)
        self.btn_upload_calibrated.setEnabled(False)
        self.btn_upload_calibrated.clicked.connect(self._on_upload_calibrated_clicked)

        v_bhtom.addWidget(lbl)
        v_bhtom.addSpacing(4)
        v_bhtom.addWidget(self.btn_bhtom_connect)
        v_bhtom.addSpacing(6)
        v_bhtom.addWidget(self.lbl_bhtom_status)
        v_bhtom.addSpacing(6)
        v_bhtom.addWidget(self.btn_upload_calibrated)

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

    def _refresh_bhtom_ui(self) -> None:
        if self.bhtom_token:
            self.btn_bhtom_connect.setText("Log out")
            self.lbl_bhtom_status.setText(
                f"Connected as {self.bhtom_username or 'unknown'}"
            )
            if self.calibrated_files:
                self.btn_upload_calibrated.setEnabled(True)
        else:
            self.btn_bhtom_connect.setText("Connect to BHTOM…")
            if not self.bhtom_username:
                self.lbl_bhtom_status.setText("Not connected")
            self.btn_upload_calibrated.setEnabled(False)

    def _on_bhtom_connect_or_logout(self) -> None:
        if self.bhtom_token:
            # logout
            ans = QMessageBox.question(
                self,
                "Log out from BHTOM",
                "Do you really want to log out from BHTOM?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
            bhtom_api.clear_credentials()
            self.bhtom_token = None
            self.bhtom_username = None
            self._refresh_bhtom_ui()
            self.lbl_status.setText("BHTOM: logged out")
        else:
            # connect
            self._connect_bhtom()

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
        self.current_frame_index = (self.current_frame_index + 1) % len(
            self.loaded_frames
        )
        self._show_current_frame()

    def _toggle_blink(self, state: int) -> None:
        if state == Qt.Checked and len(self.loaded_frames) > 1:
            self.blink_timer.start()
        else:
            self.blink_timer.stop()

    def _update_coords(
        self, x: Optional[int], y: Optional[int], val: Optional[float]
    ) -> None:
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
                self,
                "Input directory",
                "Please select a directory with raw FITS files.",
            )
            return None
        path = Path(text)
        if not path.is_dir():
            QMessageBox.warning(
                self, "Input directory", f"'{path}' is not a directory."
            )
            return None
        return path

    def _browse_input_dir(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select raw FITS directory"
        )
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
            self,
            "Lists created",
            f"Calibration lists created for '{input_dir.name}'.",
        )

    def _run_create_masters(self) -> None:
        res = self._ensure_config_and_dir()
        if res is None:
            return
        _, input_dir = res

        from calibration import mkmasterbias, mkmasterdark, mkmasterflats

        # Build a temporary pipeline just to get cfg with absolute directories
        try:
            pipeline = CalibrationPipeline(
                str(self.current_config_path),
                root_dir=str(input_dir),
            )
        except TypeError:
            # fallback for older CalibrationPipeline without root_dir
            pipeline = CalibrationPipeline(str(self.current_config_path))

        cfg = pipeline.cfg  # this cfg now has working_dir/results_dir under input_dir

        # Collect all FITS files in the selected directory
        dir_path = Path(str(input_dir)).expanduser().resolve()
        fits_files = sorted(
            str(p.resolve())
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix in {".fits", ".fit", ".FITS", ".FIT"}
        )
        if not fits_files:
            QMessageBox.warning(
                self, "Master frames", "No FITS files found in the selected directory."
            )
            return

        # Master bias
        masterbias_path = mkmasterbias.create_master_bias(
            fits_files,
            cfg,
            output_filename="masterbias.fits",
            method=cfg.get("IMAGE_PROCESSING", "bias_subtraction_method"),
            sigma=cfg.get("IMAGE_PROCESSING", "bias_subtraction_sigma"),
            make_png_flag=False,
            verbose=True,
        )

        # Master dark
        masterdark_path = mkmasterdark.make_master_dark(
            fits_files,
            cfg,
            output_filename="masterdark.fits",
            method=cfg.get("IMAGE_PROCESSING", "dark_correction_method"),
            make_png_flag=False,
            verbose=True,
        )

        # Master flats
        mkmasterflats.create_master_flats(fits_files, cfg, verbose=True)

        QMessageBox.information(
            self,
            "Master frames",
            f"Master bias created at:\n{masterbias_path}\n\n"
            f"Master dark created at:\n{masterdark_path}\n\n"
            "Master flats created (see work/ and results/aux under your raw directory).",
        )

    def _run_full_calibration(self) -> None:
        """
        Run full CCD calibration pipeline via CalibrationPipeline.

        All work/results/aux directories are created INSIDE the selected
        raw directory, e.g. Gaia19cuu/work, Gaia19cuu/results, Gaia19cuu/results/aux.
        """
        res = self._ensure_config_and_dir()
        if res is None:
            return
        cfg, input_dir = res

        if self.current_config_path is None:
            QMessageBox.warning(
                self, "Config", "No config file path found for the selected profile."
            )
            return

        # *** important: use the selected directory as root_dir ***
        try:
            pipeline = CalibrationPipeline(
                str(self.current_config_path),
                root_dir=str(input_dir),  # <— this is new
            )
        except TypeError:
            # fallback if older CalibrationPipeline without root_dir argument
            pipeline = CalibrationPipeline(str(self.current_config_path))

        log_lines: List[str] = []

        def log(msg: str) -> None:
            log_lines.append(msg)
            print(msg)

        try:
            final_files = pipeline.run_full_calibration(
                raw_source=str(input_dir),
                source_type="directory",
                verbose=True,
                log_callback=log,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Calibration error",
                f"Full calibration failed:\n{e}",
            )
            return

        # If pipeline returned nothing, still try to pick up any -bdf frames from results/.
        if not final_files:
            results_dir = Path(input_dir) / "results"
            if results_dir.is_dir():
                final_files = [
                    str(p)
                    for p in sorted(results_dir.glob("*.fits"))
                    if "-bdf" in p.name.lower()
                ]

        self.calibrated_files = [Path(p) for p in final_files]

        self.lbl_status.setText(
            f"Calibration complete: {len(self.calibrated_files)} calibrated science frames."
        )

        preview_count = min(5, len(self.calibrated_files))
        preview_names = "\n".join(p.name for p in self.calibrated_files[:preview_count])
        extra = ""
        if len(self.calibrated_files) > preview_count:
            extra = f"\n… and {len(self.calibrated_files) - preview_count} more."

        QMessageBox.information(
            self,
            "Full calibration",
            f"Calibration finished.\n"
            f"Config: {Path(self.current_config_path).name}\n"
            f"Input directory: {input_dir}\n"
            f"Calibrated frames: {len(self.calibrated_files)}\n\n"
            f"Examples:\n{preview_names}{extra}",
        )

        # Show 'Upload to BHTOM' if we have frames
        self.btn_upload_calibrated.setVisible(bool(self.calibrated_files))
        self.btn_upload_calibrated.setEnabled(
            bool(self.calibrated_files) and self.bhtom_token is not None
        )

        # Automatically open first calibrated frame in the viewer
        if self.calibrated_files:
            try:
                arr = fits.getdata(str(self.calibrated_files[0])).astype(np.float32)
                self.loaded_frames = [(self.calibrated_files[0], arr)]
                self.current_frame_index = 0
                self._show_current_frame()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # BHTOM integration
    # ------------------------------------------------------------------
    def _connect_bhtom(self) -> None:
        dlg = BHTOMLoginDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return

        username = dlg.username
        token = dlg.token
        remember = dlg.remember_me

        if not username or not token:
            QMessageBox.warning(
                self, "BHTOM", "No valid session returned from login dialog."
            )
            return

        if remember:
            bhtom_api.save_credentials(username, token)
        else:
            bhtom_api.clear_credentials()

        self.bhtom_username = username
        self.bhtom_token = token
        self.lbl_status.setText("BHTOM: authenticated")
        self._refresh_bhtom_ui()

        # If we already have calibrated frames, allow upload
        if self.calibrated_files:
            self.btn_upload_calibrated.setEnabled(True)

    def _on_upload_calibrated_clicked(self) -> None:
        """
        Upload currently calibrated files (self.calibrated_files) to BHTOM
        using the REST API from bhtom_api.upload_calibrated_files.
        """
        # 1) Ensure we have calibrated frames
        if not self.calibrated_files:
            QMessageBox.warning(
                self,
                "Upload to BHTOM",
                "No calibrated frames available. Run '3. Run full calibration' first.",
            )
            return

        # 2) Ensure user is logged in
        if not self.bhtom_token:
            self._connect_bhtom()
            if not self.bhtom_token:
                # user cancelled login
                return

        # 3) Quick preview/confirmation of what will be uploaded
        max_preview = 10
        names = [p.name for p in self.calibrated_files[:max_preview]]
        preview = "\n".join(names)
        extra = ""
        if len(self.calibrated_files) > max_preview:
            extra = f"\n… and {len(self.calibrated_files) - max_preview} more"

        answer = QMessageBox.question(
            self,
            "Upload calibrated frames to BHTOM",
            f"User: {self.bhtom_username or 'unknown'}\n"
            f"Frames to upload: {len(self.calibrated_files)}\n\n"
            f"Preview:\n{preview}{extra}\n\n"
            "Do you want to proceed?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        # 4) Ask for target name
        default_target = self.last_bhtom_target or ""
        target_name, ok = QInputDialog.getText(
            self,
            "BHTOM target name",
            "Enter existing BHTOM target name:",
            QLineEdit.Normal,
            default_target,
        )
        if not ok or not target_name.strip():
            return
        target_name = target_name.strip()

        # 5) Ask for observatory ONAME
        #    (example: 'AZT-8_C4-16000', 'BIALKOW_ANDOR-DW432', etc.)
        default_oname = self.last_bhtom_observatory or ""
        observatory_oname, ok = QInputDialog.getText(
            self,
            "Observatory / camera ONAME",
            "Enter observatory/camera ONAME (e.g. 'AZT-8_C4-16000'):",
            QLineEdit.Normal,
            default_oname,
        )
        if not ok or not observatory_oname.strip():
            return
        observatory_oname = observatory_oname.strip()

        # 6) Ask for filter string (optional – default GaiaSP/any)
        default_filter = self.last_bhtom_filter or "GaiaSP/any"
        filter_name, ok = QInputDialog.getText(
            self,
            "Filter identifier",
            "Enter BHTOM filter name (e.g. 'GaiaSP/any').\n"
            "Leave empty to use the default:",
            QLineEdit.Normal,
            default_filter,
        )
        if not ok:
            return
        filter_name = (filter_name or "").strip() or "GaiaSP/any"

        # 7) Remember last used values for convenience
        self.last_bhtom_target = target_name
        self.last_bhtom_observatory = observatory_oname
        self.last_bhtom_filter = filter_name

        # 8) Perform upload
        paths = [Path(p) for p in self.calibrated_files]

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            bhtom_api.upload_calibrated_files(
                files=paths,
                token=self.bhtom_token,
                target=target_name,
                observatory=observatory_oname,
                filter_name=filter_name,
            )
        except bhtom_api.BHTOMUploadError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Upload failed", str(exc))
            return
        except Exception as exc:  # safety net
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(
                self,
                "Unexpected error",
                f"An unexpected error occurred while uploading:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        QMessageBox.information(
            self,
            "Upload completed",
            (
                f"Successfully uploaded {len(paths)} calibrated file(s) to BHTOM.\n\n"
                f"Target: {target_name}\n"
                f"Observatory: {observatory_oname}\n"
                f"Filter: {filter_name}"
            ),
        )