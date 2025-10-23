from __future__ import annotations
import sys, os, math, csv
import numpy as np

# --- Qt compatibility layer (PyQt6 preferred, fallback to PyQt5) ---
QT_API = None
try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QHBoxLayout,
        QPushButton, QTabWidget, QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QFormLayout,
        QGroupBox, QTextEdit, QSizePolicy
    )
    QT_API = "PyQt6"
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QMainWindow, QLabel, QLineEdit, QGridLayout, QVBoxLayout, QHBoxLayout,
        QPushButton, QTabWidget, QComboBox, QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QFormLayout,
        QGroupBox, QTextEdit, QSizePolicy
    )
    QT_API = "PyQt5"

# Matplotlib (Qt5/Qt6-agnostic qtagg backend)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from heston import HestonParams, HestonModel, bs_price_call_put, bs_implied_vol

def _spinbox(parent: QWidget, val: float, lo: float, hi: float, step: float, decimals: int=6) -> QDoubleSpinBox:
    sb = QDoubleSpinBox(parent)
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setSingleStep(step)
    sb.setValue(val)
    return sb

def _intbox(parent: QWidget, val: int, lo: int, hi: int, step: int=1) -> QSpinBox:
    sb = QSpinBox(parent)
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setValue(val)
    return sb

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.ax = ax
        self.fig = fig
        self.fig.tight_layout()

class HestonGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Heston Model Lab — Professional Edition ({QT_API})")
        self.resize(1200, 800)

        self.params = HestonParams()
        self.model = HestonModel(self.params)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self._init_params_panel()
        self._init_pricing_tab()
        self._init_sim_tab()
        self._init_smile_tab()

    # ------------------------
    # Parameter panel (top)
    # ------------------------
    def _init_params_panel(self):
        panel = QWidget()
        layout = QGridLayout(panel)

        # Row 0
        self.S0 = _spinbox(panel, 100.0, 1e-6, 1e6, 1.0, 4)
        self.v0 = _spinbox(panel, 0.04, 1e-8, 10.0, 0.001, 6)
        self.kappa = _spinbox(panel, 1.5, 1e-8, 50.0, 0.05, 6)
        self.theta = _spinbox(panel, 0.04, 1e-8, 10.0, 0.001, 6)
        self.xi = _spinbox(panel, 0.5, 1e-8, 5.0, 0.01, 6)
        self.rho = _spinbox(panel, -0.7, -0.999, 0.999, 0.01, 6)
        self.r = _spinbox(panel, 0.01, -0.5, 1.0, 0.005, 6)
        self.q = _spinbox(panel, 0.0, -0.5, 1.0, 0.005, 6)

        labels = ["S0", "v0 (variance)", "kappa", "theta", "xi (vol of vol)", "rho", "r", "q"]
        widgets = [self.S0, self.v0, self.kappa, self.theta, self.xi, self.rho, self.r, self.q]
        for i, (lab, w) in enumerate(zip(labels, widgets)):
            layout.addWidget(QLabel(lab), 0, i)
            layout.addWidget(w, 1, i)

        self.btn_apply = QPushButton("Apply Params")
        self.btn_apply.clicked.connect(self.apply_params)
        layout.addWidget(self.btn_apply, 2, 0, 1, len(labels))

        # PyQt6 uses Qt.Corner.TopRightCorner; PyQt5 uses Qt.TopRightCorner
        try:
            corner_enum = Qt.Corner.TopRightCorner  # PyQt6
        except AttributeError:
            corner_enum = Qt.TopRightCorner  # PyQt5
        self.tabs.layout = QVBoxLayout()
        self.tabs.setCornerWidget(panel, corner_enum)

    def apply_params(self):
        try:
            self.params = HestonParams(
                S0=float(self.S0.value()),
                v0=float(self.v0.value()),
                kappa=float(self.kappa.value()),
                theta=float(self.theta.value()),
                xi=float(self.xi.value()),
                rho=float(self.rho.value()),
                r=float(self.r.value()),
                q=float(self.q.value()),
            )
            self.model = HestonModel(self.params)
            QMessageBox.information(self, "Params", "Parameters updated successfully.", QMessageBox.StandardButton.Ok if QT_API=="PyQt6" else QMessageBox.Ok)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update parameters:\n{e}")

    # ------------------------
    # Pricing tab
    # ------------------------
    def _init_pricing_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)

        # Controls
        ctrl = QGroupBox("Pricing Controls")
        form = QFormLayout(ctrl)
        self.T_price = _spinbox(tab, 1.0, 1e-6, 50.0, 0.05, 6)
        self.K_price = _spinbox(tab, 100.0, 1e-6, 1e6, 1.0, 4)
        self.opt_type = QComboBox(tab)
        self.opt_type.addItems(["call", "put"])
        self.btn_price = QPushButton("Compute Price & IV")
        self.btn_price.clicked.connect(self._compute_price)
        form.addRow("Maturity T (years):", self.T_price)
        form.addRow("Strike K:", self.K_price)
        form.addRow("Option Type:", self.opt_type)
        form.addRow(self.btn_price)

        # Output
        out = QGroupBox("Results")
        out_layout = QFormLayout(out)
        self.lbl_price = QLabel("—")
        self.lbl_iv = QLabel("—")
        out_layout.addRow("Price:", self.lbl_price)
        out_layout.addRow("Implied Vol (BS):", self.lbl_iv)

        # Plot
        self.canvas_price = MplCanvas(tab, width=5.5, height=4.0, dpi=120)
        self.toolbar_price = NavigationToolbar(self.canvas_price, tab)

        left = QVBoxLayout()
        left.addWidget(ctrl)
        left.addWidget(out)
        right = QVBoxLayout()
        right.addWidget(self.toolbar_price)
        right.addWidget(self.canvas_price)

        grid.addLayout(left, 0, 0, 1, 1)
        grid.addLayout(right, 0, 1, 1, 2)

        self.tabs.addTab(tab, "Pricing")

    def _compute_price(self):
        try:
            T = float(self.T_price.value())
            K = float(self.K_price.value())
            flag = str(self.opt_type.currentText())
            price = self.model.price_call_put(K, T, flag)
            iv = self.model.implied_vol(K, T, flag)

            self.lbl_price.setText(f"{price:.6f}")
            self.lbl_iv.setText("NaN" if (iv is None or np.isnan(iv)) else f"{iv:.6f}")

            # Integrand decay preview
            self.canvas_price.ax.clear()
            us = np.linspace(1e-6, 200.0, 1000)
            lnK = math.log(K)
            def integrand_j(j:int, u:float) -> float:
                u_c = complex(u, 0.0)
                i = 1j
                denom = self.params.S0 * math.exp((self.params.r - self.params.q)*T) if j==1 else 1.0
                if j==1:
                    phi = self.model.charfunc(u_c - i, T)
                else:
                    phi = self.model.charfunc(u_c, T)
                val = np.exp(-1j*u*lnK) * phi / (1j*u)
                return (val.real / denom)

            y1 = [integrand_j(1,u) for u in us]
            y2 = [integrand_j(2,u) for u in us]
            self.canvas_price.ax.plot(us, y1, label="P1 integrand (real)")
            self.canvas_price.ax.plot(us, y2, label="P2 integrand (real)")
            self.canvas_price.ax.set_xlabel("u")
            self.canvas_price.ax.set_ylabel("integrand")
            self.canvas_price.ax.set_title("Integrand Decay Preview")
            self.canvas_price.ax.legend()
            self.canvas_price.fig.tight_layout()
            self.canvas_price.draw()
        except Exception as e:
            QMessageBox.critical(self, "Pricing Error", f"{e}")

    # ------------------------
    # Simulation tab
    # ------------------------
    def _init_sim_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)

        # Controls
        ctrl = QGroupBox("Simulation Controls")
        form = QFormLayout(ctrl)
        self.T_sim = _spinbox(tab, 1.0, 1e-6, 50.0, 0.05, 6)
        self.steps_sim = _intbox(tab, 252, 1, 20000, 1)
        self.paths_sim = _intbox(tab, 1000, 1, 200000, 100)
        self.scheme_sim = QComboBox(tab)
        self.scheme_sim.addItems(["QE", "Euler"])
        self.antithetic_sim = QComboBox(tab)
        self.antithetic_sim.addItems(["True", "False"])
        self.seed_sim = _intbox(tab, 42, 0, 10**9-1, 1)
        self.btn_run_sim = QPushButton("Run Simulation")
        self.btn_run_sim.clicked.connect(self._run_simulation)
        self.btn_export_sim = QPushButton("Export CSV (paths)")
        self.btn_export_sim.clicked.connect(self._export_sim_csv)

        form.addRow("T (years):", self.T_sim)
        form.addRow("# Steps:", self.steps_sim)
        form.addRow("# Paths:", self.paths_sim)
        form.addRow("Scheme:", self.scheme_sim)
        form.addRow("Antithetic:", self.antithetic_sim)
        form.addRow("Seed:", self.seed_sim)
        form.addRow(self.btn_run_sim)
        form.addRow(self.btn_export_sim)

        # Outputs
        out = QGroupBox("Summary")
        out_layout = QFormLayout(out)
        self.lbl_S_mean = QLabel("—")
        self.lbl_S_std = QLabel("—")
        self.lbl_v_mean = QLabel("—")
        self.lbl_v_std = QLabel("—")
        out_layout.addRow("E[S_T]:", self.lbl_S_mean)
        out_layout.addRow("Std[S_T]:", self.lbl_S_std)
        out_layout.addRow("E[v_T]:", self.lbl_v_mean)
        out_layout.addRow("Std[v_T]:", self.lbl_v_std)

        # Plots
        self.canvas_sim_S = MplCanvas(tab, width=5.5, height=3.2, dpi=120)
        self.toolbar_sim_S = NavigationToolbar(self.canvas_sim_S, tab)
        self.canvas_sim_v = MplCanvas(tab, width=5.5, height=3.2, dpi=120)
        self.toolbar_sim_v = NavigationToolbar(self.canvas_sim_v, tab)

        left = QVBoxLayout()
        left.addWidget(ctrl)
        left.addWidget(out)
        right = QVBoxLayout()
        right.addWidget(self.toolbar_sim_S)
        right.addWidget(self.canvas_sim_S)
        right.addWidget(self.toolbar_sim_v)
        right.addWidget(self.canvas_sim_v)

        grid.addLayout(left, 0, 0, 2, 1)
        grid.addLayout(right, 0, 1, 2, 2)

        self._sim_cache = None  # (t, S, v)
        self.tabs.addTab(tab, "Simulation")

    def _run_simulation(self):
        try:
            T = float(self.T_sim.value())
            steps = int(self.steps_sim.value())
            paths = int(self.paths_sim.value())
            scheme = str(self.scheme_sim.currentText())
            antithetic = True if self.antithetic_sim.currentText() == "True" else False
            seed = int(self.seed_sim.value())

            # Optional: keep UI responsive on huge path runs (Qt5/6)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor if hasattr(Qt, "CursorShape") else Qt.WaitCursor)
            try:
                t, S, v = self.model.simulate_paths(T, steps, paths, scheme=scheme, antithetic=antithetic, seed=seed)
            finally:
                QApplication.restoreOverrideCursor()

            self._sim_cache = (t, S, v)

            # Summary
            ST = S[:, -1]
            vT = v[:, -1]
            self.lbl_S_mean.setText(f"{np.mean(ST):.6f}")
            self.lbl_S_std.setText(f"{np.std(ST, ddof=1):.6f}")
            self.lbl_v_mean.setText(f"{np.mean(vT):.6f}")
            self.lbl_v_std.setText(f"{np.std(vT, ddof=1):.6f}")

            # Plot few sample paths
            self.canvas_sim_S.ax.clear()
            n_plot = min(25, S.shape[0])
            for i in range(n_plot):
                self.canvas_sim_S.ax.plot(t, S[i], lw=0.8)
            self.canvas_sim_S.ax.set_title("Sample Price Paths")
            self.canvas_sim_S.ax.set_xlabel("Time (years)")
            self.canvas_sim_S.ax.set_ylabel("S")
            self.canvas_sim_S.fig.tight_layout()
            self.canvas_sim_S.draw()

            self.canvas_sim_v.ax.clear()
            for i in range(n_plot):
                self.canvas_sim_v.ax.plot(t, v[i], lw=0.8)
            self.canvas_sim_v.ax.set_title("Sample Variance Paths")
            self.canvas_sim_v.ax.set_xlabel("Time (years)")
            self.canvas_sim_v.ax.set_ylabel("v")
            self.canvas_sim_v.fig.tight_layout()
            self.canvas_sim_v.draw()

        except Exception as e:
            QMessageBox.critical(self, "Simulation Error", f"{e}")

    def _export_sim_csv(self):
        if not self._sim_cache:
            QMessageBox.warning(self, "Export", "Run a simulation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "heston_paths.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            t, S, v = self._sim_cache
            n, m = S.shape
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                header = ["path_id", "time_index", "t", "S", "v"]
                w.writerow(header)
                for i in range(n):
                    for j in range(m):
                        w.writerow([i, j, float(t[j]), float(S[i, j]), float(v[i, j])])
            QMessageBox.information(self, "Export", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"{e}")

    # ------------------------
    # Smile tab
    # ------------------------
    def _init_smile_tab(self):
        tab = QWidget()
        grid = QGridLayout(tab)

        ctrl = QGroupBox("Smile Controls")
        form = QFormLayout(ctrl)
        self.T_smile = _spinbox(tab, 1.0, 1e-6, 50.0, 0.05, 6)
        self.centerK_smile = _spinbox(tab, 100.0, 1e-6, 1e7, 1.0, 4)
        self.range_smile = _spinbox(tab, 0.5, 0.01, 5.0, 0.05, 3)  # ± as fraction of K
        self.npts_smile = _intbox(tab, 21, 3, 401, 2)
        self.flag_smile = QComboBox(tab)
        self.flag_smile.addItems(["call", "put"])
        self.btn_smile = QPushButton("Generate Smile")
        self.btn_smile.clicked.connect(self._gen_smile)
        self.btn_smile_export = QPushButton("Export CSV (smile)")
        self.btn_smile_export.clicked.connect(self._export_smile_csv)

        form.addRow("T (years):", self.T_smile)
        form.addRow("Center K:", self.centerK_smile)
        form.addRow("± Range (×K):", self.range_smile)
        form.addRow("# Points:", self.npts_smile)
        form.addRow("Type:", self.flag_smile)
        form.addRow(self.btn_smile)
        form.addRow(self.btn_smile_export)

        self.canvas_smile = MplCanvas(tab, width=6.5, height=4.5, dpi=120)
        self.toolbar_smile = NavigationToolbar(self.canvas_smile, tab)

        left = QVBoxLayout()
        left.addWidget(ctrl)
        right = QVBoxLayout()
        right.addWidget(self.toolbar_smile)
        right.addWidget(self.canvas_smile)

        grid.addLayout(left, 0, 0, 1, 1)
        grid.addLayout(right, 0, 1, 1, 2)

        self._smile_cache = None  # (Ks, IVs)
        self.tabs.addTab(tab, "IV Smile")

    def _gen_smile(self):
        try:
            T = float(self.T_smile.value())
            Kc = float(self.centerK_smile.value())
            rng = float(self.range_smile.value())
            npts = int(self.npts_smile.value())
            flag = str(self.flag_smile.currentText())
            Ks = np.linspace(Kc * (1 - rng), Kc * (1 + rng), npts)
            ivs = self.model.iv_smile(Ks, T, flag)
            self._smile_cache = (Ks, ivs, T, flag)

            self.canvas_smile.ax.clear()
            self.canvas_smile.ax.plot(Ks, ivs, marker="o", lw=1.2)
            self.canvas_smile.ax.set_title(f"Implied Vol Smile (T={T:.3f}, {flag})")
            self.canvas_smile.ax.set_xlabel("Strike K")
            self.canvas_smile.ax.set_ylabel("Implied Volatility")
            self.canvas_smile.fig.tight_layout()
            self.canvas_smile.draw()
        except Exception as e:
            QMessageBox.critical(self, "Smile Error", f"{e}")

    def _export_smile_csv(self):
        if not self._smile_cache:
            QMessageBox.warning(self, "Export", "Generate a smile first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "heston_smile.csv", "CSV Files (*.csv)")
        if not path:
            return
        try:
            Ks, ivs, T, flag = self._smile_cache
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["K", "IV", "T", "flag"])
                for K, iv in zip(Ks, ivs):
                    w.writerow([float(K), float(iv), float(T), flag])
            QMessageBox.information(self, "Export", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"{e}")

def main():
    app = QApplication(sys.argv)
    win = HestonGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
