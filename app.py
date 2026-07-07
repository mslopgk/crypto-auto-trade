"""Application entry point.

``multiprocessing.freeze_support()`` MUST be the first statement in ``main``
(research brief §4.3): the optimizer spawns a ProcessPoolExecutor and, under a
PyInstaller onedir build on Windows, child processes re-enter here.
"""
from __future__ import annotations

import logging
import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()

    from PySide6.QtWidgets import QApplication

    from gui.event_bus import bus
    from gui.panels.log_panel import attach_log_handler
    from gui.theme import apply_dark_theme

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_dark_theme(app)

    # create the bus in the GUI thread so worker emissions queue into this loop
    bus()
    attach_log_handler(logging.INFO)

    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()

    logging.getLogger(__name__).info("application started")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
