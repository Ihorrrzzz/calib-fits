# gui/bhtom_login_dialog.py

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from bhtom_api import get_auth_token, BHTOMAuthError


ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class BHTOMLoginDialog(QDialog):
    """
    Modern-looking login dialog for BHTOM.

    On success:
        self.username  -> str
        self.token     -> str
        self.remember_me -> bool

    On "Continue without login" it also returns Accepted, but token is None.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Connect to BHTOM")
        self.setModal(True)
        self.setFixedSize(540, 640)
        self.setAttribute(Qt.WA_DeleteOnClose)

        # Public results
        self.username: str | None = None
        self.token: str | None = None
        self.remember_me: bool = False

        self._build_ui()
        self._center_on_parent()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(40, 32, 40, 32)
        root_layout.setSpacing(16)

        # Top logo + title
        logo_label = QLabel()
        logo_path = ASSETS_DIR / "logo.png"
        if logo_path.exists():
            pix = QPixmap(str(logo_path))
            pix = pix.scaledToWidth(157, Qt.SmoothTransformation)
            logo_label.setPixmap(pix)
            logo_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        title_label = QLabel("Welcome back")
        title_label.setAlignment(Qt.AlignHCenter)
        title_label.setStyleSheet(
            "color: white; font-family: 'Lexend'; font-size: 32px; font-weight: 700;"
        )

        subtitle_label = QLabel("Sign in to your BHTOM account")
        subtitle_label.setAlignment(Qt.AlignHCenter)
        subtitle_label.setStyleSheet(
            "color: #A09FA0; font-family: 'Lexend'; font-size: 16px;"
        )

        header_box = QVBoxLayout()
        header_box.setSpacing(8)
        header_box.addWidget(logo_label)
        header_box.addWidget(title_label)
        header_box.addWidget(subtitle_label)

        root_layout.addLayout(header_box)

        # Fields
        form_widget = QWidget()
        form_layout = QGridLayout(form_widget)
        form_layout.setContentsMargins(0, 24, 0, 0)
        form_layout.setHorizontalSpacing(0)
        form_layout.setVerticalSpacing(20)

        # Username
        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("example.username")
        self.username_edit.setClearButtonEnabled(True)

        # Password + eye icon
        self.password_edit = QLineEdit()
        self.password_edit.setPlaceholderText("Enter at least 8+ characters")
        self.password_edit.setEchoMode(QLineEdit.Password)

        self.toggle_pw_btn = QPushButton()
        self.toggle_pw_btn.setFlat(True)
        self.toggle_pw_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.toggle_pw_btn.setFixedSize(24, 24)
        self.toggle_pw_btn.setFocusPolicy(Qt.NoFocus)
        self.toggle_pw_btn.setStyleSheet("border: none; background: transparent;")

        hidden_path = ASSETS_DIR / "hidden.png"
        visible_path = ASSETS_DIR / "visible.png"
        self._hidden_icon = (
            QPixmap(str(hidden_path)).scaled(
                16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            if hidden_path.exists()
            else QPixmap()
        )
        self._visible_icon = (
            QPixmap(str(visible_path)).scaled(
                16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            if visible_path.exists()
            else QPixmap()
        )
        if not self._hidden_icon.isNull():
            self.toggle_pw_btn.setIcon(self._hidden_icon)
        self.toggle_pw_btn.clicked.connect(self._toggle_password_visibility)

        pw_container = QWidget()
        pw_layout = QHBoxLayout(pw_container)
        pw_layout.setContentsMargins(0, 0, 0, 0)
        pw_layout.addWidget(self.password_edit)
        pw_layout.addWidget(self.toggle_pw_btn)

        # Labels above inputs
        user_label = QLabel("Username")
        user_label.setStyleSheet(
            "color: #171A1F; font-family: 'Manrope'; font-size: 14px; font-weight: 700;"
        )
        pw_label = QLabel("Password")
        pw_label.setStyleSheet(
            "color: #171A1F; font-family: 'Manrope'; font-size: 14px; font-weight: 700;"
        )

        # Card-like containers
        user_card = self._make_input_card(user_label, self.username_edit)
        pw_card = self._make_input_card(pw_label, pw_container)

        form_layout.addWidget(user_card, 0, 0)
        form_layout.addWidget(pw_card, 1, 0)

        root_layout.addWidget(form_widget)

        # Remember + forgot
        remember_forgot_layout = QHBoxLayout()
        remember_forgot_layout.setContentsMargins(0, 8, 0, 0)

        self.remember_cb = QCheckBox("Remember me")
        self.remember_cb.setStyleSheet(
            "color: #BDC1CA; font-family: 'Manrope'; font-size: 14px;"
        )

        forgot_label = QLabel(
            '<a href="https://bh-tom2.astrolabs.pl">Forgot password?</a>'
        )
        forgot_label.setOpenExternalLinks(True)
        forgot_label.setStyleSheet(
            "color: #BDC1CA; font-family: 'Manrope'; font-size: 14px;"
        )

        remember_forgot_layout.addWidget(self.remember_cb)
        remember_forgot_layout.addStretch(1)
        remember_forgot_layout.addWidget(forgot_label)

        root_layout.addLayout(remember_forgot_layout)

        # Sign in button
        self.sign_in_btn = QPushButton("Sign in")
        self.sign_in_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.sign_in_btn.clicked.connect(self._on_sign_in_clicked)
        self.sign_in_btn.setFixedHeight(44)
        self.sign_in_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #DE3B40;
                color: white;
                border-radius: 4px;
                font-family: 'Manrope';
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #f04c52;
            }
            QPushButton:pressed {
                background-color: #ba3136;
            }
            """
        )

        root_layout.addWidget(self.sign_in_btn)

        # Continue without login
        self.continue_btn = QPushButton("Continue without login")
        self.continue_btn.setCursor(QCursor(Qt.PointingHandCursor))
        self.continue_btn.clicked.connect(self._on_continue_without_login)
        self.continue_btn.setFixedHeight(40)
        self.continue_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2B2B2B;
                color: #BDC1CA;
                border-radius: 4px;
                font-family: 'Manrope';
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #383838;
            }
            QPushButton:pressed {
                background-color: #1f1f1f;
            }
            """
        )

        root_layout.addWidget(self.continue_btn)

        # Spacer at bottom
        root_layout.addSpacerItem(QSpacerItem(0, 20))

        # Dialog background
        self.setStyleSheet(
            """
            QDialog {
                background-color: #151515;
            }
            QLineEdit {
                border: none;
                background: transparent;
                color: #171A1F;
                font-family: 'Manrope';
                font-size: 14px;
            }
            QLineEdit:placeholder {
                color: #BDC1CA;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            """
        )

    def _make_input_card(self, label: QLabel, field_widget: QWidget) -> QWidget:
        card = QWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(4)
        layout.addWidget(label)
        layout.addWidget(field_widget)
        card.setStyleSheet(
            """
            QWidget {
                background-color: #FFFFFF;
                border-radius: 4px;
            }
            """
        )
        return card

    def _center_on_parent(self) -> None:
        if self.parent() is None:
            return
        parent_geom = self.parent().geometry()
        self.move(
            parent_geom.center().x() - self.width() // 2,
            parent_geom.center().y() - self.height() // 2,
        )

    # ------------------------------------------------------------------ slots

    def _toggle_password_visibility(self) -> None:
        if self.password_edit.echoMode() == QLineEdit.Password:
            self.password_edit.setEchoMode(QLineEdit.Normal)
            if not self._visible_icon.isNull():
                self.toggle_pw_btn.setIcon(self._visible_icon)
        else:
            self.password_edit.setEchoMode(QLineEdit.Password)
            if not self._hidden_icon.isNull():
                self.toggle_pw_btn.setIcon(self._hidden_icon)

    def _on_sign_in_clicked(self) -> None:
        username = self.username_edit.text().strip()
        password = self.password_edit.text()
        remember = self.remember_cb.isChecked()

        if not username or not password:
            QMessageBox.warning(
                self, "Missing data", "Please enter both username and password."
            )
            return

        self.sign_in_btn.setEnabled(False)
        self.sign_in_btn.setText("Signing in...")

        try:
            token = get_auth_token(username, password)
        except BHTOMAuthError as exc:
            QMessageBox.critical(self, "Login failed", str(exc))
            self.sign_in_btn.setEnabled(True)
            self.sign_in_btn.setText("Sign in")
            return

        # Success
        self.username = username
        self.token = token
        self.remember_me = remember
        self.accept()

    def _on_continue_without_login(self) -> None:
        self.username = None
        self.token = None
        self.remember_me = False
        self.accept()
