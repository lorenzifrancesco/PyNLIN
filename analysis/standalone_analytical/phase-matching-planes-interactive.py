"""Interactive, rotatable version of ``phase-matching-planes.py``.

Exports the same geometry -- the two XPM planes, the tangent plane, the exact
beta4-bent sheet, the notebook slice nu_4 = 0 and the WDM channel cubes -- into a
single self-contained HTML file with a small canvas renderer.  No plotting
library is involved, so it runs with the project's existing dependencies and the
result opens in any browser.

Drag to rotate, wheel to zoom, and use the checkboxes to toggle layers.

    python analysis/standalone_analytical/phase-matching-planes-interactive.py

See phase_matching_planes.md for the mathematics.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT_HTML = Path("media/standalone_analytical/phase_matching_planes.html")


def load_base():
    """Import the sibling script, whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(
        "phase_matching_planes_base", HERE / "phase-matching-planes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def sheet_triangles(base, delta, half, scale, resolution=49):
    """Triangulate the exact sheet, keeping only quads fully inside the box."""
    nu1, nu2, nu3 = base.fwm_sheet(delta, half, resolution=resolution)

    points = np.stack([nu1, nu2, nu3], axis=-1) * scale
    limit = half * scale

    valid = np.all(np.isfinite(points), axis=-1) & np.all(
        np.abs(points) <= limit, axis=-1
    )

    triangles = []

    for i in range(resolution - 1):
        for j in range(resolution - 1):
            if not (valid[i, j] and valid[i + 1, j]
                    and valid[i, j + 1] and valid[i + 1, j + 1]):
                continue

            a = points[i, j]
            b = points[i + 1, j]
            c = points[i + 1, j + 1]
            d = points[i, j + 1]

            triangles.append([a.tolist(), b.tolist(), c.tolist()])
            triangles.append([a.tolist(), c.tolist(), d.tolist()])

    return triangles


def build_scene():
    base = load_base()

    delta = base.TWOPI * (base.F_COI_THZ - base.F_REF_THZ) * base.THZ
    half = base.TWOPI * base.SPAN_THZ * base.THZ
    spacing = base.TWOPI * base.CHANNEL_SPACING_THZ * base.THZ
    side = base.TWOPI * base.SYMBOL_RATE_THZ * base.THZ

    scale = 1.0 / (base.TWOPI * base.THZ)
    limit = half * scale

    nu_zdf = base.zero_dispersion_offset(delta)

    layers = []

    # ---- the two XPM planes ---------------------------------------
    for normal, colour, label in (
        ([1.0, -1.0, 0.0], "#fc8d62", "ν₁ = ν₂ — XPM plane"),
        ([0.0, 1.0, -1.0], "#8da0cb", "ν₂ = ν₃ — XPM plane"),
    ):
        polygon = base.plane_box_polygon(np.array(normal), 0.0, half) * scale

        layers.append({
            "name": label,
            "colour": colour,
            "opacity": 0.30,
            "polygons": [polygon.tolist()],
            "lines": [],
            "lineWidth": 1.5,
        })

    # ---- tangent plane at p = q = 0 -------------------------------
    tangent = base.plane_box_polygon(
        np.array([1.0, 0.0, 1.0]), 2.0 * nu_zdf[0], half
    ) * scale

    layers.append({
        "name": "ν₁ + ν₃ = 2ν_ZDF — tangent plane",
        "colour": "#66c2a5",
        "opacity": 0.30,
        "polygons": [tangent.tolist()] if len(tangent) else [],
        "lines": [],
        "lineWidth": 1.5,
    })

    # ---- the exact beta4-bent sheet -------------------------------
    layers.append({
        "name": "exact sheet — β₄ bent",
        "colour": "#1b7837",
        "opacity": 0.80,
        "polygons": sheet_triangles(base, delta, half, scale),
        "lines": [],
        "lineWidth": 0.0,
    })

    # ---- the notebook slice nu4 = 0 -------------------------------
    slice_polygon = base.plane_box_polygon(
        np.array([1.0, -1.0, 1.0]), 0.0, half
    ) * scale

    slice_lines = []
    for index in range(len(slice_polygon)):
        slice_lines.append([
            slice_polygon[index].tolist(),
            slice_polygon[(index + 1) % len(slice_polygon)].tolist(),
        ])

    layers.append({
        "name": "notebook slice ν₄ = 0",
        "colour": "#111111",
        "opacity": 0.10,
        "polygons": [slice_polygon.tolist()],
        "lines": slice_lines,
        "lineWidth": 2.2,
    })

    # ---- channel cubes crossed by the sheet -----------------------
    def sheet_function(nu):
        u = nu[:, 0] - nu[:, 1]
        v = nu[:, 1] - nu[:, 2]
        s = delta + 0.5 * (nu[:, 0] + nu[:, 2])
        return base.local_beta2(s) + base.BETA4 * (u**2 + v**2) / 24.0

    cube_lines = []
    cube_count = 0

    for n1 in base.CHANNEL_INDICES:
        for n2 in base.CHANNEL_INDICES:
            for n3 in base.CHANNEL_INDICES:
                centre = spacing * np.array([n1, n2, n3], dtype=float)
                corners = base.box_corners(0.5 * side) + centre

                if np.max(np.abs(corners)) > half:
                    continue

                values = sheet_function(corners)
                if values.min() > 0.0 or values.max() < 0.0:
                    continue

                cube_count += 1
                for a, b in base.channel_cube_edges(n1, n2, n3, spacing, side):
                    cube_lines.append([(a * scale).tolist(), (b * scale).tolist()])

    layers.append({
        "name": f"channel cubes on the sheet — {cube_count}",
        "colour": "#333333",
        "opacity": 0.0,
        "polygons": [],
        "lines": cube_lines,
        "lineWidth": 1.0,
    })

    # ---- bounding box ---------------------------------------------
    corners = base.box_corners(limit)
    box_lines = [
        [corners[i].tolist(), corners[j].tolist()]
        for i, j in base.BOX_EDGES
    ]

    layers.append({
        "name": "bounding box",
        "colour": "#9a9a9a",
        "opacity": 0.0,
        "polygons": [],
        "lines": box_lines,
        "lineWidth": 1.0,
    })

    print(f"channel cubes crossed by the sheet: {cube_count}")
    print(f"sheet triangles: {len(layers[3]['polygons'])}")

    return {
        "layers": layers,
        "limit": limit,
        "axisLabels": ["ν\u2081/2π [THz]", "ν\u2082/2π [THz]", "ν\u2083/2π [THz]"],
        "readout": [
            ["ω_COI/2π", f"{base.F_COI_THZ:.1f} THz"],
            ["ω_ref/2π", f"{base.F_REF_THZ:.1f} THz"],
            ["ν_ZDF/2π", f"{nu_zdf[0] * scale:+.2f} THz"],
            ["β₂", f"{base.BETA2 * 1e27:+.3f} ps²/km"],
            ["β₃", f"{base.BETA3 * 1e39:+.4f} ps³/km"],
            ["β₄", f"{base.BETA4 * 1e51:+.5f} ps⁴/km"],
            ["Δf", f"{base.CHANNEL_SPACING_THZ:.1f} THz"],
            ["R_s", f"{base.SYMBOL_RATE_THZ:.1f} THz"],
        ],
    }


HTML_TEMPLATE = """<title>FWM phase-matching surfaces</title>
<style>
  :root {
    color-scheme: light dark;
    --ground: #fafaf8;
    --panel: #f3f3ef;
    --ink: #191c1f;
    --muted: #6b7178;
    --hairline: #dedbd4;
    --accent: #2f6b76;
    --stage: #ffffff;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --ground: #101317; --panel: #171b20; --ink: #e4e7ea; --muted: #8b929a;
      --hairline: #272c33; --accent: #6fb3bd;
    }
  }
  :root[data-theme="dark"] {
    --ground: #101317; --panel: #171b20; --ink: #e4e7ea; --muted: #8b929a;
    --hairline: #272c33; --accent: #6fb3bd;
  }
  :root[data-theme="light"] {
    --ground: #fafaf8; --panel: #f3f3ef; --ink: #191c1f; --muted: #6b7178;
    --hairline: #dedbd4; --accent: #2f6b76; --stage: #ffffff;
  }

  body {
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font: 14px/1.5 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  }

  #wrap {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 268px;
    gap: 22px;
    align-items: start;
    max-width: 1180px;
    margin: 0 auto;
    padding: 26px 22px 34px;
  }
  @media (max-width: 820px) { #wrap { grid-template-columns: minmax(0, 1fr); } }

  header { grid-column: 1 / -1; border-bottom: 1px solid var(--hairline);
           padding-bottom: 14px; }
  h1 {
    font: 400 25px/1.2 "Iowan Old Style", "Charter", Georgia, serif;
    margin: 0 0 5px;
    text-wrap: balance;
    letter-spacing: -0.005em;
  }
  header p { margin: 0; color: var(--muted); max-width: 62ch; font-size: 13.5px; }

  #stage {
    background: var(--stage);
    border: 1px solid var(--hairline);
    border-radius: 3px;
    overflow: hidden;
  }
  #canvas { display: block; width: 100%; touch-action: none; cursor: grab; }
  #canvas.dragging { cursor: grabbing; }

  aside { display: flex; flex-direction: column; gap: 20px; }

  .eyebrow {
    font-size: 10.5px; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--muted);
    margin: 0 0 9px;
  }

  .eqn {
    background: var(--panel);
    border-left: 2px solid var(--accent);
    padding: 11px 13px;
    font: 15px/1.7 "Iowan Old Style", "Charter", Georgia, serif;
  }
  .eqn em { font-style: italic; }
  .eqn .note { display: block; margin-top: 7px; font: 12px/1.5 ui-sans-serif,
               system-ui, sans-serif; color: var(--muted); }

  .layer { display: flex; align-items: baseline; gap: 9px; padding: 4px 0;
           cursor: pointer; }
  .layer input { margin: 0; accent-color: var(--accent); flex: 0 0 auto;
                 position: relative; top: 2px; }
  .key { width: 15px; height: 9px; border-radius: 1px; flex: 0 0 auto;
         position: relative; top: 3px; }
  .layer span.name { font-size: 13px; }
  .layer input:focus-visible { outline: 2px solid var(--accent);
                               outline-offset: 2px; }

  table { border-collapse: collapse; width: 100%; }
  td {
    font: 12px/1.6 ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
    font-variant-numeric: tabular-nums;
    padding: 2px 0;
    border-bottom: 1px solid var(--hairline);
  }
  td:first-child { color: var(--muted); }
  td:last-child { text-align: right; }

  .views { display: flex; flex-wrap: wrap; gap: 6px; }
  button {
    font: 12px/1 ui-sans-serif, system-ui, sans-serif;
    padding: 6px 10px;
    color: var(--ink);
    background: var(--panel);
    border: 1px solid var(--hairline);
    border-radius: 2px;
    cursor: pointer;
  }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  .hint { color: var(--muted); font-size: 12px; margin: 0; }
</style>

<div id="wrap">
  <header>
    <h1>Phase-matching surfaces in the FWM frequency space</h1>
    <p>
      The locus &Delta;&beta;&nbsp;=&nbsp;0 in the offset coordinates
      (&nu;<sub>1</sub>,&nbsp;&nu;<sub>2</sub>,&nbsp;&nu;<sub>3</sub>), with
      &nu;<sub>4</sub>&nbsp;=&nbsp;&nu;<sub>1</sub>&nbsp;&minus;&nbsp;&nu;<sub>2</sub>&nbsp;+&nbsp;&nu;<sub>3</sub>
      eliminated by energy conservation. Two exact planes, one
      &beta;<sub>4</sub>-bent sheet.
    </p>
  </header>

  <div id="stage"><canvas id="canvas"></canvas></div>

  <aside>
    <div>
      <p class="eyebrow">Governing factorisation</p>
      <div class="eqn">
        &Delta;<em>&beta;</em> = (<em>p</em>&sup2; &minus; <em>q</em>&sup2;)
        [ <em>&beta;</em>&#772;<sub>2</sub>(<em>S</em>)
        + <em>&beta;</em><sub>4</sub>(<em>p</em>&sup2; + <em>q</em>&sup2;)/12 ]
        <span class="note">
          <em>S</em> is the common mean of the two pairs; <em>p</em>, <em>q</em>
          their half-splittings. The first factor gives the XPM planes, the
          bracket the FWM sheet.
        </span>
      </div>
    </div>

    <div>
      <p class="eyebrow">Layers</p>
      <div id="toggles"></div>
    </div>

    <div>
      <p class="eyebrow">Parameters</p>
      <table id="readout"></table>
    </div>

    <div>
      <p class="eyebrow">View</p>
      <div class="views">
        <button data-view="iso">Iso</button>
        <button data-view="n1">down &nu;<sub>1</sub></button>
        <button data-view="n2">down &nu;<sub>2</sub></button>
        <button data-view="n3">down &nu;<sub>3</sub></button>
      </div>
      <p class="hint" style="margin-top:9px">
        Drag to rotate &middot; wheel to zoom &middot; shift-drag to pan.
      </p>
    </div>
  </aside>
</div>

<script>
const SCENE = __SCENE__;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const readout = document.getElementById('readout');
for (const [label, value] of SCENE.readout) {
  const row = readout.insertRow();
  row.insertCell().textContent = label;
  row.insertCell().textContent = value;
}

const VIEWS = {
  iso: { azim: -52 * Math.PI / 180, elev: 22 * Math.PI / 180 },
  n1:  { azim: 0, elev: 0 },
  n2:  { azim: Math.PI / 2, elev: 0 },
  n3:  { azim: 0, elev: Math.PI / 2 - 1e-4 },
};

let view = { azim: VIEWS.iso.azim, elev: VIEWS.iso.elev,
             zoom: 1, panX: 0, panY: 0 };

const enabled = SCENE.layers.map(() => true);

const toggles = document.getElementById('toggles');
SCENE.layers.forEach((layer, index) => {
  const label = document.createElement('label');
  label.className = 'layer';

  const box = document.createElement('input');
  box.type = 'checkbox';
  box.checked = true;
  box.onchange = () => { enabled[index] = box.checked; draw(); };

  const key = document.createElement('span');
  key.className = 'key';
  key.style.background = layer.colour;

  const name = document.createElement('span');
  name.className = 'name';
  name.textContent = layer.name;

  label.append(box, key, name);
  toggles.append(label);
});

for (const button of document.querySelectorAll('[data-view]')) {
  button.onclick = () => {
    const preset = VIEWS[button.dataset.view];
    view.azim = preset.azim;
    view.elev = preset.elev;
    view.zoom = 1;
    view.panX = 0;
    view.panY = 0;
    draw();
  };
}

// Orthographic camera: yaw about the nu3 axis, then tilt.  The screen basis
// (sx, sy, depth) below is orthonormal, so there is no perspective shortening.
function project(point) {
  const ca = Math.cos(view.azim), sa = Math.sin(view.azim);
  const ce = Math.cos(view.elev), se = Math.sin(view.elev);

  const x = point[0] * ca + point[1] * sa;
  const y = -point[0] * sa + point[1] * ca;
  const z = point[2];

  return [y, z * ce - x * se, x * ce + z * se];
}

let cssWidth = 0, cssHeight = 0;

function draw() {
  const ratio = window.devicePixelRatio || 1;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const unit = view.zoom * Math.min(cssWidth, cssHeight) / (2.75 * SCENE.limit);
  const cx = cssWidth / 2 + view.panX, cy = cssHeight / 2 + view.panY;

  const toScreen = (point) => {
    const p = project(point);
    return [cx + p[0] * unit, cy - p[1] * unit, p[2]];
  };

  const items = [];

  SCENE.layers.forEach((layer, index) => {
    if (!enabled[index]) return;

    for (const polygon of layer.polygons) {
      if (polygon.length < 3) continue;

      const pts = polygon.map(toScreen);
      let depth = 0;
      for (const p of pts) depth += p[2];

      items.push({ kind: 'poly', pts, depth: depth / pts.length, layer });
    }

    for (const segment of layer.lines) {
      const pts = segment.map(toScreen);
      items.push({ kind: 'line', pts, depth: (pts[0][2] + pts[1][2]) / 2, layer });
    }
  });

  // Painter's algorithm: farthest first.
  items.sort((a, b) => a.depth - b.depth);

  for (const item of items) {
    ctx.beginPath();
    ctx.moveTo(item.pts[0][0], item.pts[0][1]);
    for (let i = 1; i < item.pts.length; i++) {
      ctx.lineTo(item.pts[i][0], item.pts[i][1]);
    }

    if (item.kind === 'poly') {
      ctx.closePath();

      if (item.layer.opacity > 0) {
        ctx.globalAlpha = item.layer.opacity;
        ctx.fillStyle = item.layer.colour;
        ctx.fill();
      }
      if (item.layer.lineWidth > 0) {
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = item.layer.colour;
        ctx.lineWidth = item.layer.lineWidth;
        ctx.stroke();
      }
    } else {
      ctx.globalAlpha = 0.75;
      ctx.strokeStyle = item.layer.colour;
      ctx.lineWidth = item.layer.lineWidth || 1;
      ctx.stroke();
    }
  }

  // Axis labels just outside the positive ends.
  ctx.globalAlpha = 1;
  ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillStyle = '#4a5058';   // the stage is paper-white in both themes
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((axis, index) => {
    const end = axis.map((component) => component * SCENE.limit * 1.2);
    const p = toScreen(end);
    ctx.fillText(SCENE.axisLabels[index], p[0], p[1]);
  });
}

let dragging = false, panning = false, lastX = 0, lastY = 0;

canvas.addEventListener('pointerdown', (event) => {
  dragging = true;
  panning = event.shiftKey;
  lastX = event.clientX;
  lastY = event.clientY;
  canvas.classList.add('dragging');
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener('pointermove', (event) => {
  if (!dragging) return;

  const dx = event.clientX - lastX, dy = event.clientY - lastY;
  lastX = event.clientX;
  lastY = event.clientY;

  if (panning) {
    view.panX += dx;
    view.panY += dy;
  } else {
    view.azim += dx * 0.008;
    view.elev = Math.max(-Math.PI / 2 + 0.02,
                Math.min(Math.PI / 2 - 0.02, view.elev + dy * 0.008));
  }
  draw();
});

for (const name of ['pointerup', 'pointercancel']) {
  canvas.addEventListener(name, () => {
    dragging = false;
    canvas.classList.remove('dragging');
  });
}

canvas.addEventListener('wheel', (event) => {
  event.preventDefault();
  view.zoom = Math.max(0.25, Math.min(8,
    view.zoom * Math.exp(-event.deltaY * 0.0012)));
  draw();
}, { passive: false });

function fit() {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();

  cssWidth = rect.width;
  cssHeight = Math.max(420, Math.min(680, rect.width * 0.82));

  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  canvas.style.height = cssHeight + 'px';

  draw();
}

window.addEventListener('resize', fit);
new MutationObserver(draw).observe(document.documentElement,
  { attributes: true, attributeFilter: ['data-theme'] });
fit();
</script>
"""


def main():
    scene = build_scene()

    html = HTML_TEMPLATE.replace("__SCENE__", json.dumps(scene))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")

    size_kb = OUT_HTML.stat().st_size / 1024
    print(f"wrote {OUT_HTML} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    main()
