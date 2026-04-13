#!/usr/bin/env python3
"""
SCT Coherence Field Simulator
==============================
Selective Coherence Theory — Interactive Field Visualization
Tomasz Adam Markiewicz, 2026
www.sctheory.info

Requires: pip install numpy matplotlib scipy

Controls:
  Left click      — add coherence source
  Right click     — remove nearest source
  Scroll wheel    — zoom in/out
  D slider        — change field exponent
  Space           — pause/resume
  R               — reset to default sources
  S               — save current frame as PNG
  Q / Esc         — quit
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import gaussian_filter
import time, sys

# ── PARAMETERS ──
GRID_W, GRID_H = 600, 400      # field resolution
FPS_TARGET      = 30
D_DEFAULT       = 1.77          # = √π
D_MIN, D_MAX    = 0.3, 4.0
OMEGA           = 1.2           # angular frequency of coherence waves
K_WAVE          = 0.025         # spatial wave number
N_SOURCES_MAX   = 10

# ── COHERENCE FIELD ──
class CoherenceSource:
    def __init__(self, x, y, r0=80, phase=0.0, color='cyan'):
        self.x     = x
        self.y     = y
        self.r0    = r0
        self.phase = phase
        self.color = color

    def amplitude(self, X, Y, D):
        dr = np.sqrt((X - self.x)**2 + (Y - self.y)**2)
        return 1.0 / (1.0 + np.power(np.maximum(dr, 0.1) / self.r0, D))

    def complex_field(self, X, Y, D, t):
        dr     = np.sqrt((X - self.x)**2 + (Y - self.y)**2)
        amp    = 1.0 / (1.0 + np.power(np.maximum(dr, 0.1) / self.r0, D))
        phase  = self.phase - dr * K_WAVE + OMEGA * t
        return amp * np.exp(1j * phase)


def compute_field(sources, X, Y, D, t, mode='magnitude'):
    """Compute coherence field on grid."""
    if not sources:
        return np.zeros((GRID_H, GRID_W))

    if mode == 'magnitude':
        C = np.zeros((GRID_H, GRID_W))
        for s in sources:
            C += s.amplitude(X, Y, D)
        return np.clip(C, 0, 2.5)

    elif mode == 'complex':
        C = np.zeros((GRID_H, GRID_W), dtype=complex)
        for s in sources:
            C += s.complex_field(X, Y, D, t)
        return C

    elif mode == 'interference':
        C = np.zeros((GRID_H, GRID_W), dtype=complex)
        for s in sources:
            C += s.complex_field(X, Y, D, t)
        return np.abs(C)


def field_to_rgba(field_complex, mode='phase_magnitude'):
    """Map complex field to RGBA image."""
    mag   = np.abs(field_complex)
    phase = np.angle(field_complex)   # -π to π

    mag_norm = np.clip(mag / (mag.max() + 1e-8), 0, 1)

    if mode == 'phase_magnitude':
        # Hue = phase, Value = magnitude, Saturation = high
        hue = (phase + np.pi) / (2 * np.pi)       # 0–1
        sat = np.ones_like(hue) * 0.85
        val = mag_norm ** 0.6

        hsv = np.stack([hue, sat, val], axis=-1)
        rgb = hsv_to_rgb(hsv)

        # Fade low-magnitude regions to black
        alpha = mag_norm ** 0.4
        rgba  = np.dstack([rgb, alpha])

    elif mode == 'thermal':
        # Blue → cyan → white (like the website)
        r = np.clip(mag_norm * 0.15, 0, 1)
        g = np.clip(mag_norm * 0.55 + 0.1, 0, 1)
        b = np.clip(mag_norm * 0.8 + 0.2, 0, 1)
        alpha = mag_norm ** 0.5
        rgba  = np.dstack([r, g, b, alpha])

    return rgba


# ── MAIN APP ──
class SCTSimulator:
    def __init__(self):
        self.D       = D_DEFAULT
        self.t       = 0.0
        self.paused  = False
        self.mode    = 'interference'   # 'magnitude' | 'interference' | 'phase_magnitude'
        self.colormap= 'phase_magnitude'

        # Grid
        x = np.linspace(0, GRID_W, GRID_W)
        y = np.linspace(0, GRID_H, GRID_H)
        self.X, self.Y = np.meshgrid(x, y)

        # Default sources
        self.sources = [
            CoherenceSource(GRID_W*0.28, GRID_H*0.45, r0=90,  phase=0.0),
            CoherenceSource(GRID_W*0.72, GRID_H*0.55, r0=80,  phase=np.pi),
            CoherenceSource(GRID_W*0.50, GRID_H*0.28, r0=70,  phase=np.pi/2),
        ]

        self._build_ui()
        self._connect_events()
        self._last_time = time.time()

    def _build_ui(self):
        self.fig = plt.figure(
            figsize=(14, 8),
            facecolor='#04050a',
            num='SCT Coherence Field Simulator'
        )
        self.fig.patch.set_facecolor('#04050a')

        # Main field axis
        self.ax = self.fig.add_axes([0.02, 0.15, 0.65, 0.82])
        self.ax.set_facecolor('#04050a')
        self.ax.set_xticks([]); self.ax.set_yticks([])
        for sp in self.ax.spines.values():
            sp.set_color('#1a8fd1'); sp.set_linewidth(0.5)

        # Title
        self.fig.text(0.35, 0.97,
            'SCT Coherence Field  ·  C(r) = 1/(1+(r/r₀)^D)',
            ha='center', va='top', color='#4fc3f7',
            fontsize=11, fontfamily='monospace')

        # Initial image
        blank = np.zeros((GRID_H, GRID_W, 4))
        self.im = self.ax.imshow(blank, origin='lower',
            extent=[0, GRID_W, 0, GRID_H], aspect='auto', interpolation='bilinear')

        # Source markers
        self.source_dots, = self.ax.plot([], [], 'o',
            color='#4fc3f7', ms=8, zorder=5, alpha=0.9)
        self.source_rings = []

        # ── RIGHT PANEL ──
        panel_x = 0.70

        # D slider
        ax_d = self.fig.add_axes([panel_x, 0.78, 0.25, 0.03], facecolor='#0d0d1a')
        self.sl_d = widgets.Slider(ax_d, 'D', D_MIN, D_MAX,
            valinit=D_DEFAULT, color='#1a8fd1', initcolor='#4fc3f7')
        self.sl_d.label.set_color('#4fc3f7')
        self.sl_d.valtext.set_color('#e8b84b')
        self.sl_d.on_changed(self._on_d_change)

        # r₀ slider
        ax_r = self.fig.add_axes([panel_x, 0.72, 0.25, 0.03], facecolor='#0d0d1a')
        self.sl_r = widgets.Slider(ax_r, 'r₀', 20, 200,
            valinit=80, color='#1a8fd1', initcolor='#4fc3f7')
        self.sl_r.label.set_color('#4fc3f7')
        self.sl_r.valtext.set_color('#e8b84b')

        # Speed slider
        ax_s = self.fig.add_axes([panel_x, 0.66, 0.25, 0.03], facecolor='#0d0d1a')
        self.sl_spd = widgets.Slider(ax_s, 'Speed', 0.1, 3.0,
            valinit=1.0, color='#1a8fd1', initcolor='#4fc3f7')
        self.sl_spd.label.set_color('#4fc3f7')
        self.sl_spd.valtext.set_color('#e8b84b')

        # Mode buttons
        ax_mode = self.fig.add_axes([panel_x, 0.54, 0.25, 0.09], facecolor='#04050a')
        ax_mode.set_title('Render mode', color='#4fc3f7', fontsize=9, pad=4)
        ax_mode.set_xticks([]); ax_mode.set_yticks([])
        self.radio_mode = widgets.RadioButtons(
            ax_mode, ['Interference', 'Magnitude', 'Phase/magnitude'],
            activecolor='#4fc3f7')
        for lbl in self.radio_mode.labels:
            lbl.set_color('#94a3b8'); lbl.set_fontsize(9)
        self.radio_mode.on_clicked(self._on_mode_change)

        # Action buttons
        ax_pause = self.fig.add_axes([panel_x,      0.46, 0.11, 0.05], facecolor='#0d0d1a')
        ax_reset = self.fig.add_axes([panel_x+0.14, 0.46, 0.11, 0.05], facecolor='#0d0d1a')
        ax_save  = self.fig.add_axes([panel_x,      0.39, 0.11, 0.05], facecolor='#0d0d1a')
        ax_clear = self.fig.add_axes([panel_x+0.14, 0.39, 0.11, 0.05], facecolor='#0d0d1a')

        self.btn_pause = widgets.Button(ax_pause, 'Pause', color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_reset = widgets.Button(ax_reset, 'Reset', color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_save  = widgets.Button(ax_save,  'Save PNG', color='#0d0d1a', hovercolor='#1a1a2e')
        self.btn_clear = widgets.Button(ax_clear, 'Clear', color='#0d0d1a', hovercolor='#1a1a2e')

        for btn in [self.btn_pause, self.btn_reset, self.btn_save, self.btn_clear]:
            btn.label.set_color('#4fc3f7')
            btn.label.set_fontsize(9)

        self.btn_pause.on_clicked(self._toggle_pause)
        self.btn_reset.on_clicked(self._reset)
        self.btn_save.on_clicked(self._save)
        self.btn_clear.on_clicked(self._clear)

        # Info panel
        self.info_ax = self.fig.add_axes([panel_x, 0.15, 0.25, 0.22], facecolor='#080c14')
        self.info_ax.set_xticks([]); self.info_ax.set_yticks([])
        for sp in self.info_ax.spines.values():
            sp.set_color('#1a8fd1'); sp.set_linewidth(0.5)
        self.info_text = self.info_ax.text(
            0.05, 0.95, '', transform=self.info_ax.transAxes,
            va='top', ha='left', color='#94a3b8', fontsize=8,
            fontfamily='monospace', linespacing=1.8)

        # Instructions
        self.fig.text(0.02, 0.10,
            'Left click: add source  ·  Right click: remove  ·  Space: pause  ·  S: save  ·  R: reset  ·  Q: quit',
            color='#3d3d5c', fontsize=8, fontfamily='monospace')

        # √π annotation
        self.fig.text(panel_x, 0.84,
            f'√π = {np.sqrt(np.pi):.6f}',
            color='#e8b84b', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1400', edgecolor='#e8b84b', alpha=0.5))

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event',  self._on_click)
        self.fig.canvas.mpl_connect('key_press_event',     self._on_key)
        self.fig.canvas.mpl_connect('scroll_event',        self._on_scroll)

    def _on_d_change(self, val):
        self.D = val

    def _on_mode_change(self, label):
        modes = {'Interference': 'interference',
                 'Magnitude': 'magnitude',
                 'Phase/magnitude': 'phase_magnitude'}
        self.colormap = modes.get(label, 'interference')

    def _toggle_pause(self, event=None):
        self.paused = not self.paused
        self.btn_pause.label.set_text('Resume' if self.paused else 'Pause')

    def _reset(self, event=None):
        self.sources = [
            CoherenceSource(GRID_W*0.28, GRID_H*0.45, r0=90,  phase=0.0),
            CoherenceSource(GRID_W*0.72, GRID_H*0.55, r0=80,  phase=np.pi),
            CoherenceSource(GRID_W*0.50, GRID_H*0.28, r0=70,  phase=np.pi/2),
        ]
        self.D = D_DEFAULT
        self.sl_d.set_val(D_DEFAULT)

    def _clear(self, event=None):
        self.sources = []

    def _save(self, event=None):
        fname = f'SCT_field_D{self.D:.3f}_t{self.t:.1f}.png'
        self.fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='#04050a')
        print(f'Saved: {fname}')

    def _on_click(self, event):
        if event.inaxes != self.ax: return
        x, y = event.xdata, event.ydata
        if x is None: return
        r0 = self.sl_r.val

        if event.button == 1:  # left — add
            if len(self.sources) < N_SOURCES_MAX:
                self.sources.append(CoherenceSource(x, y, r0=r0,
                    phase=np.random.uniform(0, 2*np.pi)))
        elif event.button == 3:  # right — remove nearest
            if self.sources:
                dists = [np.sqrt((s.x-x)**2 + (s.y-y)**2) for s in self.sources]
                self.sources.pop(int(np.argmin(dists)))

    def _on_key(self, event):
        if event.key == ' ':           self._toggle_pause()
        elif event.key == 'r':         self._reset()
        elif event.key == 's':         self._save()
        elif event.key in ('q','escape'): plt.close('all'); sys.exit()

    def _on_scroll(self, event):
        # Adjust r0 of nearest source
        if event.inaxes != self.ax or not self.sources: return
        x, y = event.xdata, event.ydata
        if x is None: return
        dists = [np.sqrt((s.x-x)**2 + (s.y-y)**2) for s in self.sources]
        nearest = self.sources[int(np.argmin(dists))]
        nearest.r0 = np.clip(nearest.r0 + event.step * 5, 10, 250)

    def _update_info(self, fps):
        n = len(self.sources)
        d_str = f'{self.D:.4f}'
        sqrtpi = f'{np.sqrt(np.pi):.4f}'
        diff   = abs(self.D - np.sqrt(np.pi))
        lines  = [
            f'D      = {d_str}',
            f'√π     = {sqrtpi}',
            f'|D-√π| = {diff:.4f}',
            f'σ      = {3-self.D:.4f}',
            f'η      = {self.D-1:.4f}',
            '',
            f'Sources: {n}/{N_SOURCES_MAX}',
            f'Time:    {self.t:.1f}',
            f'FPS:     {fps:.0f}',
            '',
            'sctheory.info',
        ]
        self.info_text.set_text('\n'.join(lines))

    def run(self):
        plt.ion()
        plt.show(block=False)

        dt_target = 1.0 / FPS_TARGET

        while plt.fignum_exists(self.fig.number):
            t0 = time.time()

            if not self.paused:
                self.t += dt_target * self.sl_spd.val

                # Compute field
                C_complex = np.zeros((GRID_H, GRID_W), dtype=complex)
                for s in self.sources:
                    if self.colormap in ('interference', 'phase_magnitude'):
                        C_complex += s.complex_field(self.X, self.Y, self.D, self.t)
                    else:
                        C_complex += s.amplitude(self.X, self.Y, self.D)

                # Render
                if self.colormap == 'magnitude':
                    mag = np.abs(C_complex)
                    mag = np.clip(mag / (mag.max()+1e-8), 0, 1)
                    r   = mag * 0.15
                    g   = mag * 0.55 + 0.1 * mag
                    b   = mag * 0.8  + 0.2 * mag
                    rgba = np.dstack([np.clip(r,0,1), np.clip(g,0,1), np.clip(b,0,1), mag**0.5])
                else:
                    rgba = field_to_rgba(C_complex, self.colormap)

                self.im.set_data(rgba)

                # Source markers
                if self.sources:
                    xs = [s.x for s in self.sources]
                    ys = [s.y for s in self.sources]
                    self.source_dots.set_data(xs, ys)
                else:
                    self.source_dots.set_data([], [])

            elapsed = time.time() - t0
            fps = 1.0 / max(elapsed, 0.001)
            self._update_info(fps)

            try:
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except Exception:
                break

            sleep_t = max(0, dt_target - (time.time() - t0))
            time.sleep(sleep_t)


if __name__ == '__main__':
    print("SCT Coherence Field Simulator")
    print(f"D = √π = {np.sqrt(np.pi):.6f}")
    print("Left click to add sources, right click to remove")
    print()
    app = SCTSimulator()
    app.run()
