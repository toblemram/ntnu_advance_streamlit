import math
import streamlit as st
from streamlit.components.v1 import html as st_html

st.set_page_config(
    page_title="TBM Cutterhead Dashboard",
    page_icon="🚇",
    layout="wide",
)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color_hex(c1: str, c2: str, t: float) -> str:
    """
    Linear interpolation between two colors in hex format (#RRGGBB).
    t in [0, 1].
    """
    t = clamp(t, 0.0, 1.0)
    c1_rgb = tuple(int(c1[i: i + 2], 16) for i in (1, 3, 5))
    c2_rgb = tuple(int(c2[i: i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(int(round(lerp(a, b, t))) for a, b in zip(c1_rgb, c2_rgb))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


# ---------------------------------------------------------
# MAIN ANIMATION: TBM CUTTERHEAD + MONSTER HEN (fixed window)
# ---------------------------------------------------------
def generate_tbm_html(
    diameter_m: float,
    rpm: float,
    cutters: int,
    mb_kN: float,
    rmc: float,
) -> str:
    """
    Visualize TBM cutterhead in front view with a monster hen as reference:
    - Fixed window for the cutterhead, only the disk scales
    - Rotation based on RPM
    - Number of cutters
    - Cutter color based on gross thrust per cutter (M_B)
    - r_mc ring
    - Hen is scaled to represent ~8 m height
    """
    cutters = max(1, min(cutters, 60))

    # Shared scale: pixels per meter (used both for cutterhead and hen)
    px_per_meter = 16.0

    # Monster hen: 8 m tall in the same scale
    hen_total_px = 8.0 * px_per_meter  # ~128 px

    # Fixed window for cutterhead card
    wrapper_size_px = 420

    # Disk size in pixels using same meter scale
    disc_size_px = diameter_m * px_per_meter
    # Make sure the largest diameter (20 m) still fits well inside the window
    disc_size_px = min(disc_size_px, wrapper_size_px - 60)

    # RPM → rotation speed (seconds per revolution)
    if rpm > 0:
        period_sec = 60.0 / rpm
        cutter_animation_style = (
            f"animation: tbm-spin {period_sec:.2f}s linear infinite;"
        )
    else:
        cutter_animation_style = "animation: none;"

    # M_B → cutter color (low thrust = yellow, high = orange/red-ish)
    MB_MIN, MB_MAX = 200.0, 400.0
    mb_norm = clamp((mb_kN - MB_MIN) / (MB_MAX - MB_MIN), 0.0, 1.0)
    cutter_fill_color = lerp_color_hex("#facc15", "#f97316", mb_norm)
    cutter_stroke_color = lerp_color_hex("#facc15", "#ea580c", mb_norm)

    # Geometry in SVG space
    outer_radius = 45.0
    rmc_radius = outer_radius * clamp(rmc, 0.0, 1.0)

    # Spokes (ribs)
    spokes = []
    for i in range(6):
        angle = 2 * math.pi * i / 6
        x = 50 + 32 * math.cos(angle)
        y = 50 + 32 * math.sin(angle)
        spokes.append(
            f'<line x1="50" y1="50" x2="{x:.2f}" y2="{y:.2f}" '
            f'stroke="#111827" stroke-width="3" />'
        )

    # Cutters along outer ring
    cutter_circles = []
    for i in range(cutters):
        angle = 2 * math.pi * i / cutters
        x = 50 + 36 * math.cos(angle)
        y = 50 + 36 * math.sin(angle)
        cutter_circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" '
            f'fill="{cutter_fill_color}" stroke="{cutter_stroke_color}" stroke-width="1.4" />'
        )

    style = f"""
<style>
.tbm-ui-root {{
  background: radial-gradient(circle at 10% 20%, #1f2937 0, #020617 60%);
  border-radius: 24px;
  padding: 24px 28px;
  color: #e5e7eb;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.85);
  border: 1px solid rgba(148, 163, 184, 0.4);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
}}
.tbm-header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 18px;
}}
.tbm-header-title {{
  font-size: 1.1rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}}
.tbm-header-sub {{
  font-size: 0.8rem;
  opacity: 0.75;
}}
.tbm-visual-row {{
  display: flex;
  align-items: flex-end;
  justify-content: center;
  gap: 48px;
  margin-top: 8px;
}}

/* MONSTER HEN */
.tbm-hen {{
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}}
.tbm-hen-body-wrapper {{
  position: relative;
  width: 90px;
  height: {hen_total_px}px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}}
.tbm-hen-body {{
  position: relative;
  width: 70px;
  height: 55px;
  border-radius: 60px 60px 50px 45px;
  background: linear-gradient(180deg, #e5e7eb, #9ca3af);
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.9);
}}
.tbm-hen-wing {{
  position: absolute;
  right: 8px;
  top: 18px;
  width: 32px;
  height: 26px;
  border-radius: 40px 40px 30px 30px;
  background: radial-gradient(circle at 30% 20%, #f9fafb, #cbd5f5);
  opacity: 0.9;
}}
.tbm-hen-head {{
  position: absolute;
  right: -8px;
  top: calc(100% - 85px);  /* head moved down to sit on the body */
  width: 30px;
  height: 30px;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 20%, #f9fafb, #d1d5db);
  box-shadow: 0 0 10px rgba(248, 250, 252, 0.7);
}}
.tbm-hen-eye {{
  position: absolute;
  right: 8px;
  top: 9px;
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #0f172a;
  box-shadow: 0 0 4px rgba(15, 23, 42, 0.8);
}}
.tbm-hen-beak {{
  position: absolute;
  right: -4px;
  top: 12px;
  width: 0;
  height: 0;
  border-top: 5px solid transparent;
  border-bottom: 5px solid transparent;
  border-left: 9px solid #f97316;
}}
.tbm-hen-comb {{
  position: absolute;
  left: 3px;
  top: -6px;
  width: 16px;
  height: 10px;
  border-radius: 999px;
  background: #ef4444;
  box-shadow: 0 0 6px rgba(248, 113, 113, 0.9);
}}
.tbm-hen-legs {{
  position: absolute;
  bottom: -8px;
  left: 18px;
  display: flex;
  gap: 8px;
}}
.tbm-hen-leg {{
  width: 2px;
  height: 10px;
  background: #f97316;
  box-shadow: 0 0 6px rgba(249, 115, 22, 0.9);
}}
.tbm-hen-label {{
  font-size: 0.75rem;
  opacity: 0.9;
}}
.tbm-hen-caption {{
  font-size: 0.7rem;
  opacity: 0.7;
}}

/* CUTTERHEAD */
.tbm-disk-wrapper {{
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(circle at 30% 20%, rgba(56, 189, 248, 0.16), transparent 65%);
  border-radius: 999px;
  padding: 18px;
  width: {wrapper_size_px}px;
  height: {wrapper_size_px}px;
}}
.tbm-disk-size {{
  display: flex;
  align-items: center;
  justify-content: center;
}}
.tbm-disk-svg {{
  filter: drop-shadow(0 0 18px rgba(59, 130, 246, 0.7));
}}
.tbm-disk-ring {{
  fill: url(#tbmDiskGradient);
  stroke: #1e293b;
  stroke-width: 3;
}}
.tbm-disk-rim {{
  fill: none;
  stroke: #38bdf8;
  stroke-width: 1.4;
  stroke-dasharray: 3 3;
  opacity: 0.9;
}}
.tbm-disk-center {{
  fill: #020617;
  stroke: #38bdf8;
  stroke-width: 2;
}}
.tbm-disk-bolt {{
  fill: #94a3b8;
}}
.tbm-disk-cutout {{
  fill: rgba(15, 23, 42, 0.65);
}}
.tbm-rmc-ring {{
  fill: none;
  stroke: #a855f7;
  stroke-width: 1.5;
  stroke-dasharray: 4 3;
  opacity: 0.95;
  filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.6));
}}
.tbm-disk-gauge {{
  font-size: 0.7rem;
  opacity: 0.7;
  margin-top: 6px;
  text-align: center;
  line-height: 1.3;
}}
.tbm-disk-gauge span {{
  font-weight: 600;
  color: #e5e7eb;
}}
.tbm-legend {{
  display: flex;
  justify-content: center;
  gap: 18px;
  margin-top: 6px;
  font-size: 0.7rem;
  opacity: 0.75;
  flex-wrap: wrap;
}}
.tbm-dot {{
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  margin-right: 4px;
}}
.tbm-dot-cutter {{
  background: #fbbf24;
  box-shadow: 0 0 8px rgba(251, 191, 36, 0.9);
}}
.tbm-dot-spoke {{
  background: #38bdf8;
}}
.tbm-dot-rim {{
  background: #64748b;
}}
.tbm-dot-rmc {{
  background: #a855f7;
}}
.tbm-cutterhead {{
  transform-origin: 50% 50%;
  transform-box: fill-box;
  {cutter_animation_style}
}}
@keyframes tbm-spin {{
  from {{ transform: rotate(0deg); }}
  to {{ transform: rotate(360deg); }}
}}
</style>
"""

    html = f"""
{style}
<div class="tbm-ui-root">
  <div class="tbm-header">
    <div class="tbm-header-title">TBM cutterhead – front view</div>
    <div class="tbm-header-sub">
      Fixed window · cutterhead scales · spins with RPM
    </div>
  </div>
  <div class="tbm-visual-row">
    <div class="tbm-hen">
      <div class="tbm-hen-body-wrapper">
        <div class="tbm-hen-body">
          <div class="tbm-hen-wing"></div>
        </div>
        <div class="tbm-hen-head">
          <div class="tbm-hen-eye"></div>
          <div class="tbm-hen-beak"></div>
          <div class="tbm-hen-comb"></div>
        </div>
        <div class="tbm-hen-legs">
          <div class="tbm-hen-leg"></div>
          <div class="tbm-hen-leg"></div>
        </div>
      </div>
      <div class="tbm-hen-label">Reference monster hen</div>
      <div class="tbm-hen-caption">≈ 8 m tall in this scale</div>
    </div>
    <div class="tbm-disk-wrapper">
      <div class="tbm-disk-size" style="width: {disc_size_px:.1f}px; height: {disc_size_px:.1f}px;">
        <svg viewBox="0 0 100 100" class="tbm-disk-svg">
          <defs>
            <radialGradient id="tbmDiskGradient" cx="30%" cy="25%" r="70%">
              <stop offset="0%" stop-color="#1f2937" />
              <stop offset="45%" stop-color="#020617" />
              <stop offset="100%" stop-color="#0b1120" />
            </radialGradient>
          </defs>
          <g class="tbm-cutterhead">
            <circle cx="50" cy="50" r="{outer_radius:.1f}" class="tbm-disk-ring" />
            <circle cx="50" cy="50" r="43" class="tbm-disk-rim" />
            {"".join(spokes)}
            <circle cx="50" cy="50" r="18" class="tbm-disk-cutout" />
            <circle cx="50" cy="50" r="10.5" class="tbm-disk-center" />
            <circle cx="50" cy="50" r="3" class="tbm-disk-bolt" />
            <circle cx="50" cy="50" r="{rmc_radius:.2f}" class="tbm-rmc-ring" />
            {"".join(cutter_circles)}
          </g>
        </svg>
      </div>
    </div>
  </div>
  <div class="tbm-disk-gauge">
    <div>
      <span>{diameter_m:.1f} m</span> cutterhead diameter ·
      <span>{rpm:.1f} rpm</span> ·
      <span>{cutters}</span> cutters
    </div>
    <div>
      <span>{mb_kN:.0f} kN/c</span> gross thrust per cutter ·
      <span>r_mc = {rmc:.2f}</span>
    </div>
  </div>
  <div class="tbm-legend">
    <div><span class="tbm-dot tbm-dot-cutter"></span>Cutters</div>
    <div><span class="tbm-dot tbm-dot-spoke"></span>Ribs</div>
    <div><span class="tbm-dot tbm-dot-rim"></span>Outer ring</div>
    <div><span class="tbm-dot tbm-dot-rmc"></span>r_mc ring</div>
  </div>
</div>
"""
    return html


# ---------------------------------------------------------
# POWER METER FOR P_tbm
# ---------------------------------------------------------
def generate_power_html(power_kw: float, rpm: float) -> str:
    max_kw = 8000.0
    ratio = clamp(power_kw / max_kw, 0.0, 1.0)
    rpm_norm = clamp(rpm / 12.0, 0.0, 1.0)
    est_load_pct = clamp((ratio * 0.6 + rpm_norm * 0.4) * 100, 0, 100)

    style = """
<style>
.tbm-power-card {
  background: radial-gradient(circle at 0% 0%, rgba(52, 211, 153, 0.2), rgba(15, 23, 42, 0.9));
  border-radius: 18px;
  padding: 14px 16px 16px;
  border: 1px solid rgba(52, 211, 153, 0.6);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.7);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 0.8rem;
}
.tbm-power-main {
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 4px;
}
.tbm-power-sub {
  opacity: 0.8;
  margin-bottom: 8px;
}
.tbm-power-bar {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  overflow: hidden;
  background: linear-gradient(90deg, #020617, #020617);
  border: 1px solid rgba(148, 163, 184, 0.7);
}
.tbm-power-bar-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e, #eab308, #ef4444);
  box-shadow: 0 0 12px rgba(52, 211, 153, 0.8);
}
.tbm-power-footer {
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  opacity: 0.8;
}
</style>
"""
    html = f"""
{style}
<div class="tbm-power-card">
  <div class="tbm-power-main">{power_kw:.0f} kW</div>
  <div class="tbm-power-sub">Installed power for cutterhead</div>
  <div class="tbm-power-bar">
    <div class="tbm-power-bar-fill" style="width: {ratio*100:.1f}%;"></div>
  </div>
  <div class="tbm-power-footer">
    <span>0 – {max_kw:.0f} kW</span>
    <span>Estimated load: {est_load_pct:.0f}%</span>
  </div>
</div>
"""
    return html


# ---------------------------------------------------------
# CUTTER ZOOM FOR d_c
# ---------------------------------------------------------
def generate_cutter_zoom_html(dc_inch: float) -> str:
    dc_mm = dc_inch * 25.4

    # Normalize 10–25" → radius 20–45 in SVG units
    inch_min, inch_max = 10.0, 25.0
    t = clamp((dc_inch - inch_min) / (inch_max - inch_min), 0.0, 1.0)
    radius = 20 + t * 25

    style = """
<style>
.tbm-cutter-card {
  background: radial-gradient(circle at 100% 0%, rgba(96, 165, 250, 0.24), rgba(15, 23, 42, 0.95));
  border-radius: 18px;
  padding: 12px 14px 14px;
  border: 1px solid rgba(59, 130, 246, 0.7);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.7);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 0.78rem;
}
.tbm-cutter-title {
  font-weight: 600;
  margin-bottom: 4px;
}
.tbm-cutter-svg {
  display: block;
  margin: 4px auto 2px;
}
.tbm-cutter-caption {
  opacity: 0.8;
  text-align: center;
  margin-top: 4px;
}
</style>
"""
    svg = f"""
{style}
<div class="tbm-cutter-card">
  <div class="tbm-cutter-title">Cutter diameter</div>
  <svg viewBox="0 0 120 120" class="tbm-cutter-svg" width="120" height="120">
    <defs>
      <radialGradient id="tbmCutterGrad" cx="30%" cy="25%" r="70%">
        <stop offset="0%" stop-color="#111827" />
        <stop offset="55%" stop-color="#020617" />
        <stop offset="100%" stop-color="#020617" />
      </radialGradient>
    </defs>

    <circle cx="60" cy="60" r="{radius:.1f}" fill="url(#tbmCutterGrad)"
            stroke="#1f2937" stroke-width="2" />
    <circle cx="60" cy="60" r="{radius*0.3:.1f}" fill="#020617"
            stroke="#38bdf8" stroke-width="1.5" />
    <circle cx="60" cy="60" r="2.5" fill="#e5e7eb" />

    <circle cx="{60 + radius*0.55:.1f}" cy="60" r="2" fill="#64748b" />
    <circle cx="{60 - radius*0.55:.1f}" cy="60" r="2" fill="#64748b" />

    <line x1="25" y1="95" x2="95" y2="95" stroke="#e5e7eb" stroke-width="1" />
    <line x1="25" y1="92" x2="25" y2="98" stroke="#e5e7eb" stroke-width="1" />
    <line x1="95" y1="92" x2="95" y2="98" stroke="#e5e7eb" stroke-width="1" />
    <text x="60" y="89" text-anchor="middle" fill="#e5e7eb" font-size="8">
      {dc_inch:.1f}"  (~{dc_mm:.0f} mm)
    </text>
  </svg>
  <div class="tbm-cutter-caption">
    Scaled cutter in front view. The line shows the cutter diameter.
  </div>
</div>
"""
    return svg


# ---------------------------------------------------------
# STROKE ANIMATION FOR l_s
# ---------------------------------------------------------
def generate_stroke_html(ls_m: float) -> str:
    ls_min, ls_max = 0.5, 3.0
    t = clamp((ls_m - ls_min) / (ls_max - ls_min), 0.0, 1.0)
    track_width = 140 + t * 80  # 140–220 px

    style = """
<style>
.tbm-stroke-card {
  background: radial-gradient(circle at 0% 100%, rgba(248, 250, 252, 0.06), rgba(15, 23, 42, 0.96));
  border-radius: 18px;
  padding: 12px 14px 14px;
  border: 1px solid rgba(148, 163, 184, 0.8);
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.7);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 0.78rem;
}
.tbm-stroke-title {
  font-weight: 600;
  margin-bottom: 6px;
}
.tbm-stroke-track-outer {
  display: flex;
  justify-content: center;
  margin-bottom: 4px;
}
.tbm-stroke-track-inner {
  height: 16px;
  background: linear-gradient(90deg, #020617, #020617);
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.8);
  position: relative;
  overflow: hidden;
}
.tbm-stroke-block {
  width: 28px;
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e, #0ea5e9);
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.9);
  position: absolute;
  top: 1px;
  left: 0;
  animation: tbm-stroke-move 1.7s ease-in-out infinite;
}
.tbm-stroke-caption {
  opacity: 0.8;
  text-align: center;
  font-size: 0.7rem;
}
@keyframes tbm-stroke-move {
  0%   { transform: translateX(0); }
  50%  { transform: translateX(100%); }
  100% { transform: translateX(0); }
}
</style>
"""
    html = f"""
{style}
<div class="tbm-stroke-card">
  <div class="tbm-stroke-title">Stroke length</div>
  <div class="tbm-stroke-track-outer">
    <div class="tbm-stroke-track-inner" style="width: {track_width:.1f}px;">
      <div class="tbm-stroke-block"></div>
    </div>
  </div>
  <div class="tbm-stroke-caption">
    Stroke: {ls_m:.2f} m · relative length: {t*100:.0f} %
  </div>
</div>
"""
    return html


# ---------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------
def main():
    st.markdown(
        "<h1 style='text-align:center;'>🚇 Tunnel Boring Machine – Cutterhead input dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Adjust TBM parameters. The cutterhead, monster hen, power meter, cutter zoom and stroke animation all update live."
    )

    col_controls, col_visual = st.columns([1, 2])

    # ----------- LEFT: INPUTS ----------
    with col_controls:
        st.markdown("### Parameters")

        diameter = st.slider(
            "Cutterhead diameter d_tbm [m]",
            min_value=6.0,
            max_value=20.0,
            value=7.1,
            step=0.1,
        )

        rpm = st.slider(
            "Cutterhead RPM [rpm]",
            min_value=0.0,
            max_value=12.0,
            value=6.0,
            step=0.1,
        )

        cutters = st.slider(
            "Number of cutters N_tbm [-]",
            min_value=20,
            max_value=60,
            value=45,
            step=1,
        )

        mb = st.slider(
            "Gross thrust per cutter M_B [kN/c]",
            min_value=200,
            max_value=400,
            value=300,
            step=10,
        )

        p_tbm = st.slider(
            "Installed power P_tbm [kW]",
            min_value=1000,
            max_value=8000,
            value=3500,
            step=100,
        )

        dc_inch = st.slider(
            "Cutter diameter d_c [inch]",
            min_value=10.0,
            max_value=25.0,
            value=19.0,
            step=0.5,
        )

        ls = st.slider(
            "Stroke length l_s [m]",
            min_value=0.5,
            max_value=3.0,
            value=1.8,
            step=0.1,
        )

        rmc = st.slider(
            "Relative cutter position r_mc [-]",
            min_value=0.40,
            max_value=0.90,
            value=0.59,
            step=0.01,
        )

        # Derived: total thrust
        total_thrust = mb * cutters  # kN
        total_thrust_str = f"{total_thrust:,.0f}".replace(",", " ")

        st.markdown("### Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Cutterhead diameter", f"{diameter:.1f} m")
            st.metric("Cutters", f"{cutters}")
        with c2:
            st.metric("RPM", f"{rpm:.1f}")
            if rpm > 0:
                period = 60.0 / rpm
                st.metric("Rotation period", f"{period:.1f} s")
            else:
                st.metric("Rotation period", "∞")
        with c3:
            st.metric("M_B", f"{mb} kN/c")
            st.metric("Total thrust", f"{total_thrust_str} kN")

        c4, c5, c6 = st.columns(3)
        with c4:
            st.metric("P_tbm", f"{p_tbm} kW")
        with c5:
            st.metric("d_c", f"{dc_inch:.1f} in")
        with c6:
            st.metric("l_s", f"{ls:.2f} m")

    # ----------- RIGHT: VISUALS ----------
    with col_visual:
        st.markdown("### Cutterhead and monster hen")
        st_html(generate_tbm_html(diameter, rpm, cutters, mb, rmc), height=650)

        st.markdown("### Power, cutter and stroke")
        pcol, ccol, scol = st.columns(3)

        with pcol:
            st_html(generate_power_html(p_tbm, rpm), height=220)

        with ccol:
            st_html(generate_cutter_zoom_html(dc_inch), height=260)

        with scol:
            st_html(generate_stroke_html(ls), height=240)


if __name__ == "__main__":
    main()
