#!/usr/bin/env python3

# ============================================================
# FWM frequency matching condition (angular frequencies):
#
#   omega_1 - omega_2 + omega_3 = omega_4    (absolute)
#   nu_1   - nu_2   + nu_3   = nu_4          (offsets from COI,
#                                              nu_j = omega_j - omega_COI)
#   |nu_4| <= B / 2                            (support / bandwidth)
#
# All variables are angular frequencies.
# Here B is the full angular passband width 2 pi R_s. Axes are normalised by
# Omega_0 = 2 pi Delta_f, the base inter-channel spacing, so
# spacing_ratio = Omega_0 / B = Delta_f / R_s.
#
# The plotted integer family q_idx = n1 - n2 + n3 uses the target as index
# zero. Its carrier residual in lorenzi_fast_method.md is
#
#   d = 2 pi q_idx (Omega_0 / B) = 2 pi q_idx spacing_ratio.
#
# A q_idx family has nonzero support iff
# |q_idx| * spacing_ratio < 2. Therefore q_idx is restricted to -1, 0, 1
# when 1 <= spacing_ratio < 2; sub-Nyquist spacing can admit |q_idx| >= 2.
#
# This q_idx is not the quadratic phase coefficient q_j and not the
# spacing-to-baud ratio called q in some XPM plots.
# ============================================================

import base64
import io
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, callback
from itertools import combinations, product
from scipy.spatial import ConvexHull

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.fwm.plot_wdm_quadruplet import draw_wdm_quadruplet

B = 1.0
h = B / 2
tol = 1e-9

INDEX_VALUES = (0, 1)
PLANE_NORMAL = np.array([1.0, -1.0, 1.0])


def clipped_region_vertices(n1, n2, n3, spacing_ratio):
    omega = spacing_ratio * B
    center = omega * np.array([n1, n2, n3], dtype=float)
    lower = center - h
    upper = center + h

    A = np.array([
        [ 1,  0,  0], [-1,  0,  0],
        [ 0,  1,  0], [ 0, -1,  0],
        [ 0,  0,  1], [ 0,  0, -1],
        [ 1, -1,  1], [-1,  1, -1],
    ], dtype=float)

    b = np.array([
        upper[0], -lower[0],
        upper[1], -lower[1],
        upper[2], -lower[2],
        h, h,
    ])

    vertices = []

    for selected_planes in combinations(range(len(A)), 3):
        matrix = A[list(selected_planes)]
        rhs = b[list(selected_planes)]

        if abs(np.linalg.det(matrix)) < tol:
            continue

        point = np.linalg.solve(matrix, rhs)

        if np.all(A @ point <= b + 1e-8):
            if not any(
                np.linalg.norm(point - old_point) < 1e-8
                for old_point in vertices
            ):
                vertices.append(point)

    return np.asarray(vertices)


def true_hull_edges(vertices):
    hull = ConvexHull(vertices)
    edge_to_faces = {}

    for face_index, triangle in enumerate(hull.simplices):
        for i, j in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(i), int(j))))
            edge_to_faces.setdefault(edge, []).append(face_index)

    normals = hull.equations[:, :3]
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    edges = []

    for edge, adjacent_faces in edge_to_faces.items():
        if len(adjacent_faces) == 1:
            edges.append(edge)
            continue

        normal_1 = normals[adjacent_faces[0]]
        normal_2 = normals[adjacent_faces[1]]

        if abs(normal_1 @ normal_2) < 1 - 1e-7:
            edges.append(edge)

    return hull, edges


def edges_as_lines(vertices, edges):
    x, y, z = [], [], []

    for i, j in edges:
        p = vertices[i]
        q = vertices[j]

        x.extend([p[0], q[0], None])
        y.extend([p[1], q[1], None])
        z.extend([p[2], q[2], None])

    return x, y, z


def shifted_cube_section(n1, n2, n3, spacing_ratio, slice_value):
    omega = spacing_ratio * B
    center = omega * np.array([n1, n2, n3], dtype=float)
    lower = center - h
    upper = center + h

    cube_vertices = np.array([
        [x, y, z]
        for x in (lower[0], upper[0])
        for y in (lower[1], upper[1])
        for z in (lower[2], upper[2])
    ])

    cube_edges = []

    for i, p in enumerate(cube_vertices):
        for j, q in enumerate(cube_vertices):
            if i >= j:
                continue
            if np.count_nonzero(np.abs(p - q) > tol) == 1:
                cube_edges.append((p, q))

    points = []

    for p, q in cube_edges:
        fp = PLANE_NORMAL @ p - slice_value
        fq = PLANE_NORMAL @ q - slice_value

        if abs(fp) < tol and abs(fq) < tol:
            points.extend([p, q])
            continue

        if fp * fq <= 0 and abs(fp - fq) > tol:
            t = fp / (fp - fq)
            if -tol <= t <= 1 + tol:
                points.append(p + t * (q - p))

    unique_points = []

    for point in points:
        if not any(
            np.linalg.norm(point - old_point) < 1e-8
            for old_point in unique_points
        ):
            unique_points.append(point)

    points = np.asarray(unique_points)

    if len(points) < 3:
        return np.empty((0, 3))

    normal = PLANE_NORMAL / np.linalg.norm(PLANE_NORMAL)
    reference = np.array([1.0, 0.0, 0.0])

    if abs(normal @ reference) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    projected = np.column_stack((points @ e1, points @ e2))
    polygon_hull = ConvexHull(projected)

    return points[polygon_hull.vertices]


def derive_quadruplet(nu1, nu3, spacing_ratio, slice_value):
    """Derive nu2 from the frequency-matching condition nu1-nu2+nu3=nu4,
    with nu4 fixed by the requested output frequency (slice_value). nu1 and
    nu3 are user-controlled (nu/Omega_0 units); nu2 is whatever value makes
    the triplet land exactly on the COI's output frequency."""
    Omega_0 = spacing_ratio * B
    nu4 = slice_value / Omega_0
    nu2 = nu1 + nu3 - nu4
    return nu1, nu2, nu3, nu4


def quadruplet_image_src(nu1, nu3, spacing_ratio, slice_value):
    """Render the (nu1, nu2, nu3, nu4) quadruplet (nu2 derived, nu4 fixed by
    slice_value) with plot_wdm_quadruplet.draw_wdm_quadruplet, base64-encoded
    for an html.Img src. Passes the channel bandwidth-over-spacing
    (B/Omega_0 = 1/spacing_ratio), not just the frequencies, so the WDM comb
    is drawn to scale with the 3D admissible-region figure."""
    quadruplet = derive_quadruplet(nu1, nu3, spacing_ratio, slice_value)
    fig = draw_wdm_quadruplet(quadruplet, bandwidth_norm=1.0 / spacing_ratio)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_figure(spacing_ratio=1.10, slice_value=0.0, marker_point=None):
    Omega_0 = spacing_ratio * B
    s_norm = slice_value / Omega_0

    fig = go.Figure()
    regions = []

    for n1, n2, n3 in product(INDEX_VALUES, repeat=3):
        vertices = clipped_region_vertices(
            n1=n1,
            n2=n2,
            n3=n3,
            spacing_ratio=spacing_ratio,
        )
        vertices = vertices / Omega_0

        if len(vertices) >= 4:
            regions.append({
                "indices": (n1, n2, n3),
                "q": n1 - n2 + n3,
                "vertices": vertices,
            })

    q_colors = {
        -1: "#4daf4a",
         0: "#377eb8",
         1: "#ff7f00",
         2: "#e41a1c",
    }

    legend_shown = set()
    active_q_values = set()
    number_of_sections = 0

    for region in regions:
        n1, n2, n3 = region["indices"]
        q = region["q"]
        vertices = region["vertices"]

        active_q_values.add(q)
        group_name = f"q={q:+d}"

        hull, edges = true_hull_edges(vertices)
        edge_x, edge_y, edge_z = edges_as_lines(vertices, edges)

        hover_label = (
            f"(n₁,n₂,n₃)=({n1},{n2},{n3})"
            f"<br>q=n₁−n₂+n₃={q:+d}"
        )

        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=hull.simplices[:, 0],
            j=hull.simplices[:, 1],
            k=hull.simplices[:, 2],
            color=q_colors.get(q, "royalblue"),
            opacity=0.18,
            flatshading=True,
            name=group_name,
            legendgroup=group_name,
            showlegend=q not in legend_shown,
            text=[hover_label] * len(vertices),
            hovertemplate=(
                "%{text}<br>"
                "ν₁/Ω₀=%{x:.3f}<br>"
                "ν₂/Ω₀=%{y:.3f}<br>"
                "ν₃/Ω₀=%{z:.3f}<extra></extra>"
            ),
        ))

        legend_shown.add(q)

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="black", width=5),
            opacity=0.80,
            name=group_name,
            legendgroup=group_name,
            showlegend=False,
            hoverinfo="skip",
        ))

        section = shifted_cube_section(
            n1=n1,
            n2=n2,
            n3=n3,
            spacing_ratio=spacing_ratio,
            slice_value=slice_value,
        )
        section = section / Omega_0

        if len(section) < 3:
            continue

        number_of_sections += 1
        centroid = section.mean(axis=0)
        mesh_vertices = np.vstack([centroid, section])

        n_boundary = len(section)
        triangle_i = np.zeros(n_boundary, dtype=int)
        triangle_j = np.arange(1, n_boundary + 1)
        triangle_k = np.roll(triangle_j, -1)

        section_label = (
            f"(n₁,n₂,n₃)=({n1},{n2},{n3})"
            f"<br>q={q:+d}"
            f"<br>ν₄/Ω₀={s_norm:+.3f}"
        )

        fig.add_trace(go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=triangle_i,
            j=triangle_j,
            k=triangle_k,
            color="crimson",
            opacity=0.82,
            flatshading=True,
            name=group_name,
            legendgroup=group_name,
            showlegend=False,
            text=[section_label] * len(mesh_vertices),
            hovertemplate=(
                "%{text}<br>"
                "ν₁/Ω₀=%{x:.3f}<br>"
                "ν₂/Ω₀=%{y:.3f}<br>"
                "ν₃/Ω₀=%{z:.3f}<extra></extra>"
            ),
        ))

        closed_section = np.vstack([section, section[0]])

        fig.add_trace(go.Scatter3d(
            x=closed_section[:, 0],
            y=closed_section[:, 1],
            z=closed_section[:, 2],
            mode="lines+markers",
            line=dict(color="darkred", width=4),
            marker=dict(color="darkred", size=2),
            name=group_name,
            legendgroup=group_name,
            showlegend=False,
            hoverinfo="skip",
        ))

    if marker_point is not None:
        mx, my, mz = marker_point
        fig.add_trace(go.Scatter3d(
            x=[mx], y=[my], z=[mz],
            mode="markers",
            marker=dict(size=8, color="#ffff00", symbol="diamond",
                        line=dict(color="black", width=1.5)),
            name="selected (ν₁,ν₂,ν₃)",
            showlegend=True,
            hovertemplate=(
                "selected point<br>"
                f"ν₁/Ω₀={mx:.3f}<br>ν₂/Ω₀={my:.3f}<br>ν₃/Ω₀={mz:.3f}"
                "<extra></extra>"
            ),
        ))

    plot_min = -0.65 / Omega_0
    plot_max = 1 + 0.65 / Omega_0
    q_text = ", ".join(f"{q:+d}" for q in sorted(active_q_values))

    fig.update_layout(
        font=dict(size=16),
        title=(
            f"Channels 0 and 1 only: Ω₀/B={spacing_ratio:.2f}, "
            f"ν₄/Ω₀={s_norm:+.3f}"
            f"<br><sup>{len(regions)} admissible regions; "
            f"{number_of_sections} slice polygons; "
            f"q∈{{{q_text}}}</sup>"
            f"<br><sup>FWM: ω₁−ω₂+ω₃=ω₄; "
            f"ν₁−ν₂+ν₃=ν₄; |ν₄|≤πB</sup>"
        ),
        uirevision="domain-camera",
        legend_uirevision="q-family-visibility",
        scene=dict(
            # Pinned so slider-driven re-renders never reset user pan/zoom/
            # camera; only an actual uirevision *change* would reset it, and
            # this string never changes.
            uirevision="domain-camera",
            xaxis=dict(
                title="ν₁/Ω₀",
                range=[plot_min, plot_max],
                showbackground=False,
                zeroline=True,
            ),
            yaxis=dict(
                title="ν₂/Ω₀",
                range=[plot_min, plot_max],
                showbackground=False,
                zeroline=True,
            ),
            zaxis=dict(
                title="ν₃/Ω₀",
                range=[plot_min, plot_max],
                showbackground=False,
                zeroline=True,
            ),
            aspectmode="cube",
            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=1.55, y=-1.55, z=1.55),
                up=dict(x=0, y=0, z=1),
            ),
        ),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1.0,
                y=0.85,
                xanchor="right",
                yanchor="top",
                # Each preset pins both eye and up explicitly (not just eye)
                # so the projection always comes out with the two visible
                # axes oriented like a normal 2D plot (increasing right and
                # up), regardless of which button was clicked previously.
                buttons=[
                    dict(
                        label="Iso",
                        method="relayout",
                        args=[{
                            "scene.camera.eye": dict(x=1.55, y=-1.55, z=1.55),
                            "scene.camera.up": dict(x=0, y=0, z=1),
                        }],
                    ),
                    dict(
                        label="ν₁",
                        method="relayout",
                        args=[{
                            # looking along -x: y increases right, z increases up
                            "scene.camera.eye": dict(x=1, y=0, z=0),
                            "scene.camera.up": dict(x=0, y=0, z=1),
                        }],
                    ),
                    dict(
                        label="ν₂",
                        method="relayout",
                        args=[{
                            # looking along +y (eye on the -y side): x increases right, z increases up
                            "scene.camera.eye": dict(x=0, y=-1, z=0),
                            "scene.camera.up": dict(x=0, y=0, z=1),
                        }],
                    ),
                    dict(
                        label="ν₃",
                        method="relayout",
                        args=[{
                            # looking along -z (from above): x increases right, y increases up
                            "scene.camera.eye": dict(x=0, y=0, z=1),
                            "scene.camera.up": dict(x=0, y=1, z=0),
                        }],
                    ),
                ],
            )
        ],
        legend=dict(
            x=0.82,
            y=0.98,
            groupclick="togglegroup",
        ),
        margin=dict(l=0, r=0, b=0, t=115),
        height=760,
    )

    return fig


def _slider_block(label, slider_id, **slider_kwargs):
    """Compact label+slider pair for the control grid."""
    return html.Div(
        [
            html.Label(
                label,
                htmlFor=slider_id,
                style={
                    "display": "block",
                    "fontFamily": "sans-serif",
                    "fontWeight": "bold",
                    "fontSize": "13px",
                    "marginBottom": "2px",
                },
            ),
            dcc.Slider(
                id=slider_id,
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="mouseup",
                **slider_kwargs,
            ),
        ],
    )


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H3(
            "Frequency-domain regions: channels 0 and 1",
            style={"fontFamily": "sans-serif", "marginBottom": "4px"},
        ),
        html.P(
            "q_idx = n₁ − n₂ + n₃ labels each admissible channel-index "
            "family. In the normalized variables of lorenzi_fast_method.md, "
            "its carrier residual is d = 2π q_idx (Ω₀/B). This q_idx is not "
            "the quadratic phase coefficient q_j. Nonzero support requires "
            "|q_idx|(Ω₀/B) < 2, so q_idx ∈ {−1,0,1} for 1 ≤ Ω₀/B < 2.",
            style={
                "fontFamily": "sans-serif",
                "fontSize": "13px",
                "color": "#555",
                "marginTop": "0px",
                "marginBottom": "14px",
            },
        ),

        html.Div(
            [
                _slider_block(
                    "Ω₀/B (spacing)",
                    "spacing-slider",
                    min=0.50,
                    max=2.20,
                    step=0.05,
                    value=1.10,
                    marks={0.50: "0.50", 1.10: "1.10", 2.20: "2.20"},
                ),
                _slider_block(
                    "ν₄/Ω₀ (output)",
                    "slice-slider",
                    min=-0.5,
                    max=0.5,
                    step=0.025,
                    value=0.0,
                    marks={-0.5: "−0.50", 0.0: "0", 0.5: "+0.50"},
                ),
                _slider_block(
                    "ν₁/Ω₀",
                    "nu1-slider",
                    min=-0.75,
                    max=1.75,
                    step=0.05,
                    value=1.0,
                ),
                _slider_block(
                    "ν₃/Ω₀",
                    "nu3-slider",
                    min=-0.75,
                    max=1.75,
                    step=0.05,
                    value=0.0,
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(2, minmax(220px, 1fr))",
                "columnGap": "32px",
                "rowGap": "16px",
                "marginBottom": "18px",
            },
        ),

        html.Div(
            [
                html.Div(
                    dcc.Loading(
                        dcc.Graph(
                            id="domain-graph",
                            figure=build_figure(
                                spacing_ratio=1.10,
                                slice_value=0.0,
                                marker_point=(1.0, 1.0, 0.0),
                            ),
                            config={
                                "displaylogo": False,
                                "scrollZoom": True,
                                "responsive": True,
                            },
                            style={"height": "72vh"},
                        ),
                        type="circle",
                    ),
                    style={"flex": "1 1 58%", "minWidth": "0"},
                ),
                html.Div(
                    [
                        html.H4(
                            "Selected quadruplet (ν₁, ν₃ controlled; ν₂ derived)",
                            style={"fontFamily": "sans-serif", "marginTop": "0px"},
                        ),
                        html.Img(
                            id="quadruplet-image",
                            src=quadruplet_image_src(1.0, 0.0, 1.10, 0.0),
                            style={"maxWidth": "100%"},
                        ),
                    ],
                    style={"flex": "1 1 38%", "minWidth": "320px"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "alignItems": "flex-start",
                "gap": "24px",
            },
        ),
    ],
    style={
        "maxWidth": "1600px",
        "margin": "0 auto",
        "padding": "20px",
    },
)


@callback(
    Output("domain-graph", "figure"),
    Output("quadruplet-image", "src"),
    Input("spacing-slider", "value"),
    Input("slice-slider", "value"),
    Input("nu1-slider", "value"),
    Input("nu3-slider", "value"),
)
def update_domain_figure(spacing_ratio, slice_value, nu1, nu3):
    spacing_ratio = float(spacing_ratio)
    slice_value = float(slice_value)
    nu1 = float(nu1)
    nu3 = float(nu3)

    _, nu2, _, _ = derive_quadruplet(nu1, nu3, spacing_ratio, slice_value)
    figure = build_figure(
        spacing_ratio=spacing_ratio,
        slice_value=slice_value,
        marker_point=(nu1, nu2, nu3),
    )
    image_src = quadruplet_image_src(nu1, nu3, spacing_ratio, slice_value)
    return figure, image_src


if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=8050,
    )
