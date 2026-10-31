"""DPS-QKD Simulator -- entrypoint.

Run with::

    python main.py

Requires PyQt6, pyqtgraph, numpy, and the bundled ``sequence`` backend
package (plus its own dependencies -- see requirements.txt).
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setOrganizationName("DPS-QKD Simulator")
    app.setApplicationName("DPS-QKD Simulator")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
