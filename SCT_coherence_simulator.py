"""
SCT Coherence Field Simulator
==============================
Tomasz Adam Markiewicz · Selective Coherence Theory · 2026

Interactive simulation of the coherence field C(r) from the
non-local Ginzburg-Landau equation of SCT.

Usage:
    python SCT_coherence_simulator.py

Requirements:
    pip install numpy matplotlib

Controls:
    Click        — Add a coherence source
    Right-click  — Remove nearest source
    Scroll       — Resize nearest source (r₀)
    D slider     — Change decay exponent D
    r₀ slider    — Change default source radius
    Speed slider — Animation speed
    Mode buttons — Switch render mode
    Save         — Export current frame as PNG
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import hsv_to_rgb
import os, datetime

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════
GRID_W, GRID_H = 400, 300
D_INIT = np.sqrt(np.pi)  # √π ≈ 1.7725
R0_INIT = 60.0
SPEED_INIT = 1.0
FPS = 30

# ═══════════════════════════════════════════
# COHERENCE FIELD ENGINE
# ═══════════════════════════════════════════
class CoherenceSource:
    def __init__(self, x, y, r0=R0_INIT, phase=0.0):
        self.x = x
        self.y = y
        self.r0 = r0
        self.phase = phase

class SCTSimulator:
    def __init__(self):
        self.sources = []
        self.D = D_INIT
        self.r0_default = R0_INIT
        self.speed = SPEED_INIT
        self.time = 0.0
        self.mode = 'magnitude'  # magnitude, phase, combined

        # Pre-compute coordinate grids
        self.y_grid, self.x_grid = np.mgrid[0:GRID_H, 0:GRID_W]

        # Add two default sources
        self.sources.append(CoherenceSource(GRID_W * 0.35, GRID_H * 0.5, R0_INIT, 0))
        self.sources.append(CoherenceSource(GRID_W * 0.65, GRID_H * 0.5, R0_INIT, np.pi))

    def compute_field(self):
        """Compute complex coherence field C(r) · e^(iφ) with wave interference."""
        C_real = np.zeros((GRID_H, GRID_W))
        C_imag = np.zeros((GRID_H, GRID_W))

        for s in self.sources:
            # Distance from source
            dx = self.x_grid - s.x
            dy = self.y_grid - s.y
            dr = np.sqrt(dx**2 + dy**2) + 1e-6

            # Coherence amplitude: C(r) = 1 / (1 + (r/r₀)^D)
            amplitude = 1.0 / (1.0 + np.power(dr / s.r0, self.D))

            # Phase: radial waves + source phase + time evolution
            wave_phase = (dr / s.r0) * 2.0 * np.pi + s.phase + self.time * self.speed * 0.5

            # Complex field with interference
            C_real += amplitude * np.cos(wave_phase)
            C_imag += amplitude * np.sin(wave_phase)

        return C_real, C_imag

    def render(self):
        """Render the field as an RGB image."""
        C_real, C_imag = self.compute_field()
        magnitude = np.sqrt(C_real**2 + C_imag**2)
        phase = np.arctan2(C_imag, C_real)

        # Normalise magnitude to [0, 1]
        mag_max = np.max(magnitude) + 1e-8
        mag_norm = np.clip(magnitude / mag_max, 0, 1)

        if self.mode == 'magnitude':
            # Blue-cyan colour map based on magnitude
            r = mag_norm * 0.1
            g = mag_norm * 0.45 + 0.05
            b = 0.5 + mag_norm * 0.5
            img = np.stack([r, g, b], axis=-1)

        elif self.mode == 'phase':
            # HSV: hue = phase, saturation = 0.8, value = magnitude
            hue = (phase + np.pi) / (2 * np.pi)  # [0, 1]
            sat = np.full_like(hue, 0.85)
            val = mag_norm * 0.9 + 0.1
            hsv = np.stack([hue, sat, val], axis=-1)
            img = hsv_to_rgb(hsv)

        elif self.mode == 'combined':
            # Phase as hue, magnitude as brightness, with blue tint
            hue = (phase + np.pi) / (2 * np.pi)
            # Shift hue to blue-purple range
            hue = (hue * 0.4 + 0.55) % 1.0
            sat = 0.6 + mag_norm * 0.3
            val = mag_norm * 0.85 + 0.05
            hsv = np.stack([hue, sat, val], axis=-1)
            img = hsv_to_rgb(hsv)

        return np.clip(img, 0, 1)

    def step(self, dt=1.0/FPS):
        """Advance time for animation."""
        self.time += dt

        # Gentle source drift
        for i, s in enumerate(self.sources):
            s.x += np.sin(self.time * 0.3 + i * 1.7) * 0.3 * self.speed
            s.y += np.cos(self.time * 0.2 + i * 1.1) * 0.2 * self.speed

# ═══════════════════════════════════════════
# INTERACTIVE VISUALISATION
# ═══════════════════════════════════════════
def main():
    sim = SCTSimulator()

    # Set up figure
    fig = plt.figure(figsize=(12, 8), facecolor='#04050a')
    fig.canvas.manager.set_window_title('SCT Coherence Field Simulator — D = √π')

    # Main plot area
    ax = fig.add_axes([0.05, 0.22, 0.65, 0.72])
    ax.set_facecolor('#04050a')
    ax.set_xlim(0, GRID_W)
    ax.set_ylim(0, GRID_H)
    ax.set_aspect('equal')
    ax.axis('off')

    # Initial render
    img_data = sim.render()
    img_plot = ax.imshow(img_data, origin='lower', extent=[0, GRID_W, 0, GRID_H], aspect='auto')

    # Source markers
    source_dots, = ax.plot([], [], 'o', color='#4fc3f7', markersize=5, alpha=0.8)

    # Title
    title_text = ax.set_title('', color='#4fc3f7', fontsize=10, fontfamily='monospace', pad=8)

    # ── INFO PANEL ──
    info_ax = fig.add_axes([0.72, 0.22, 0.26, 0.72])
    info_ax.set_facecolor('#080c14')
    info_ax.axis('off')
    for spine in info_ax.spines.values():
        spine.set_color('#1a3050')

    info_lines = [
        ('SCT COHERENCE FIELD', '#4fc3f7', 12, 'bold'),
        ('', '#444', 8, 'normal'),
        ('C(r) = 1 / (1 + (r/r₀)^D)', '#e8b84b', 10, 'normal'),
        ('', '#444', 6, 'normal'),
    ]

    y_pos = 0.95
    for text, color, size, weight in info_lines:
        info_ax.text(0.05, y_pos, text, transform=info_ax.transAxes,
                    color=color, fontsize=size, fontweight=weight, fontfamily='monospace')
        y_pos -= 0.06

    # Dynamic info text
    info_d = info_ax.text(0.05, 0.68, '', transform=info_ax.transAxes,
                         color='#f0f4ff', fontsize=11, fontfamily='monospace', fontweight='bold')
    info_sqrtpi = info_ax.text(0.05, 0.62, '', transform=info_ax.transAxes,
                              color='#e8b84b', fontsize=9, fontfamily='monospace')
    info_delta = info_ax.text(0.05, 0.56, '', transform=info_ax.transAxes,
                             color='#4ade80', fontsize=9, fontfamily='monospace')
    info_sigma = info_ax.text(0.05, 0.48, '', transform=info_ax.transAxes,
                             color='#a78bfa', fontsize=9, fontfamily='monospace')
    info_eta = info_ax.text(0.05, 0.42, '', transform=info_ax.transAxes,
                           color='#a78bfa', fontsize=9, fontfamily='monospace')
    info_sources = info_ax.text(0.05, 0.34, '', transform=info_ax.transAxes,
                               color='#888', fontsize=8, fontfamily='monospace')
    info_mode = info_ax.text(0.05, 0.28, '', transform=info_ax.transAxes,
                            color='#888', fontsize=8, fontfamily='monospace')

    # Controls label
    info_ax.text(0.05, 0.18, '─── CONTROLS ───', transform=info_ax.transAxes,
                color='#333', fontsize=7, fontfamily='monospace')
    controls_text = [
        'Click: add source',
        'Right-click: remove',
        'Scroll: resize nearest',
        '',
        'Sliders below to adjust',
        'D, r₀, and speed',
    ]
    for i, line in enumerate(controls_text):
        info_ax.text(0.05, 0.13 - i * 0.04, line, transform=info_ax.transAxes,
                    color='#555', fontsize=7, fontfamily='monospace')

    # ── SLIDERS ──
    slider_color = '#1a8fd1'
    slider_bg = '#0a1520'

    ax_d = fig.add_axes([0.08, 0.12, 0.35, 0.025], facecolor=slider_bg)
    s_d = Slider(ax_d, 'D', 0.5, 3.5, valinit=D_INIT, color=slider_color, valstep=0.01)
    s_d.label.set_color('#4fc3f7')
    s_d.label.set_fontsize(9)
    s_d.valtext.set_color('#e8b84b')

    ax_r0 = fig.add_axes([0.08, 0.08, 0.35, 0.025], facecolor=slider_bg)
    s_r0 = Slider(ax_r0, 'r₀', 10, 150, valinit=R0_INIT, color=slider_color, valstep=1)
    s_r0.label.set_color('#4fc3f7')
    s_r0.label.set_fontsize(9)
    s_r0.valtext.set_color('#e8b84b')

    ax_speed = fig.add_axes([0.08, 0.04, 0.35, 0.025], facecolor=slider_bg)
    s_speed = Slider(ax_speed, 'Speed', 0, 5, valinit=SPEED_INIT, color=slider_color, valstep=0.1)
    s_speed.label.set_color('#4fc3f7')
    s_speed.label.set_fontsize(9)
    s_speed.valtext.set_color('#e8b84b')

    # ── MODE BUTTONS ──
    ax_mode = fig.add_axes([0.52, 0.04, 0.12, 0.09], facecolor=slider_bg)
    radio = RadioButtons(ax_mode, ('magnitude', 'phase', 'combined'), active=0)
    for label in radio.labels:
        label.set_color('#4fc3f7')
        label.set_fontsize(8)
        label.set_fontfamily('monospace')
    for circle in radio.circles:
        circle.set_facecolor(slider_bg)
        circle.set_edgecolor('#1a8fd1')

    # ── SAVE BUTTON ──
    ax_save = fig.add_axes([0.67, 0.04, 0.08, 0.04])
    btn_save = Button(ax_save, 'Save PNG', color='#0a1520', hovercolor='#1a3050')
    btn_save.label.set_color('#4fc3f7')
    btn_save.label.set_fontsize(8)
    btn_save.label.set_fontfamily('monospace')

    # ── RESET BUTTON ──
    ax_reset = fig.add_axes([0.67, 0.09, 0.08, 0.04])
    btn_reset = Button(ax_reset, 'Reset', color='#0a1520', hovercolor='#1a3050')
    btn_reset.label.set_color('#fb923c')
    btn_reset.label.set_fontsize(8)
    btn_reset.label.set_fontfamily('monospace')

    # ═══════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════
    def on_slider_d(val):
        sim.D = val

    def on_slider_r0(val):
        sim.r0_default = val

    def on_slider_speed(val):
        sim.speed = val

    def on_mode(label):
        sim.mode = label

    def on_click(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # Left click: add source
            phase = np.random.uniform(0, 2 * np.pi)
            sim.sources.append(CoherenceSource(event.xdata, event.ydata, sim.r0_default, phase))
            if len(sim.sources) > 8:
                sim.sources.pop(0)
        elif event.button == 3:  # Right click: remove nearest
            if sim.sources:
                dists = [np.sqrt((s.x - event.xdata)**2 + (s.y - event.ydata)**2) for s in sim.sources]
                sim.sources.pop(np.argmin(dists))

    def on_scroll(event):
        if event.inaxes != ax or not sim.sources:
            return
        dists = [np.sqrt((s.x - event.xdata)**2 + (s.y - event.ydata)**2) for s in sim.sources]
        nearest = sim.sources[np.argmin(dists)]
        nearest.r0 = np.clip(nearest.r0 + event.step * 5, 10, 200)

    def on_save(event):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        fname = f'SCT_field_D{sim.D:.4f}_{timestamp}.png'
        fig.savefig(fname, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        print(f'Saved: {fname}')

    def on_reset(event):
        sim.sources.clear()
        sim.sources.append(CoherenceSource(GRID_W * 0.35, GRID_H * 0.5, sim.r0_default, 0))
        sim.sources.append(CoherenceSource(GRID_W * 0.65, GRID_H * 0.5, sim.r0_default, np.pi))
        sim.time = 0
        s_d.set_val(D_INIT)
        s_r0.set_val(R0_INIT)
        s_speed.set_val(SPEED_INIT)

    # Connect events
    s_d.on_changed(on_slider_d)
    s_r0.on_changed(on_slider_r0)
    s_speed.on_changed(on_slider_speed)
    radio.on_clicked(on_mode)
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    btn_save.on_clicked(on_save)
    btn_reset.on_clicked(on_reset)

    # ═══════════════════════════════════════
    # ANIMATION LOOP
    # ═══════════════════════════════════════
    def update(frame=None):
        sim.step()

        # Render field
        img_data = sim.render()
        img_plot.set_data(img_data)

        # Update source dots
        if sim.sources:
            sx = [s.x for s in sim.sources]
            sy = [s.y for s in sim.sources]
            source_dots.set_data(sx, sy)
        else:
            source_dots.set_data([], [])

        # Update info panel
        sqrtpi = np.sqrt(np.pi)
        delta = abs(sim.D - sqrtpi)
        sigma = 3 - sim.D
        eta = sim.D - 1  # D - (d-2) with d=3

        info_d.set_text(f'D = {sim.D:.4f}')
        info_sqrtpi.set_text(f'√π = {sqrtpi:.6f}')
        info_delta.set_text(f'|D − √π| = {delta:.6f}')
        info_sigma.set_text(f'σ = 3 − D = {sigma:.4f}')
        info_eta.set_text(f'η = D − 1 = {eta:.4f}')
        info_sources.set_text(f'Sources: {len(sim.sources)}')
        info_mode.set_text(f'Mode: {sim.mode}')

        # Title with live readout
        title_text.set_text(
            f'C(r) = 1/(1+(r/r₀)^{sim.D:.2f})  ·  '
            f'|D−√π| = {delta:.4f}  ·  '
            f't = {sim.time:.1f}'
        )

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # Use timer for animation
    timer = fig.canvas.new_timer(interval=1000 // FPS)
    timer.add_callback(update)
    timer.start()

    plt.show()


if __name__ == '__main__':
    print()
    print('  ╔═══════════════════════════════════════════════╗')
    print('  ║   SCT Coherence Field Simulator              ║')
    print('  ║   D = √π ≈ 1.7725                            ║')
    print('  ║   Tomasz Adam Markiewicz · 2026               ║')
    print('  ╚═══════════════════════════════════════════════╝')
    print()
    print('  Controls:')
    print('    Click       — add coherence source')
    print('    Right-click — remove nearest source')
    print('    Scroll      — resize nearest source')
    print('    Sliders     — D, r₀, speed')
    print('    Buttons     — render mode, save PNG, reset')
    print()
    main()
