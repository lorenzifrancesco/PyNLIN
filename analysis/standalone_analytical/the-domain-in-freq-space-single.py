import numpy as np
import plotly.graph_objects as go

from scipy.spatial import ConvexHull, HalfspaceIntersection


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
# Omega_0 = 2 pi Delta_f, the base inter-channel spacing. For a shifted cube
# with integer family q_idx, the support shift of lorenzi_fast_method.md is
# d = 2 pi q_idx (Omega_0 / B). This single-cube view uses q_idx = 0.
# In general, nonzero support requires |q_idx| (Omega_0 / B) < 2.
# ============================================================

B = 1.0
h = B / 2
tol = 1e-10
Omega_0 = 2.0  # angular channel spacing (change as needed)
h_norm = h / Omega_0


# ============================================================
# Admissible region
#
# |nu1|, |nu2|, |nu3| <= B/2
# |nu1 - nu2 + nu3| <= B/2
# ============================================================

halfspaces = np.array([
    [ 1,  0,  0, -h],
    [-1,  0,  0, -h],
    [ 0,  1,  0, -h],
    [ 0, -1,  0, -h],
    [ 0,  0,  1, -h],
    [ 0,  0, -1, -h],
    [ 1, -1,  1, -h],
    [-1,  1, -1, -h],
])

intersection = HalfspaceIntersection(
    halfspaces,
    interior_point=np.zeros(3),
)

region_vertices = intersection.intersections
region_vertices_norm = region_vertices / Omega_0
region_hull = ConvexHull(region_vertices_norm)


# ============================================================
# Utility: extract unique edges from triangular hull faces
# ============================================================

def hull_edges(simplices):
    edges = set()

    for triangle in simplices:
        for a, b in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ]:
            edges.add(tuple(sorted((a, b))))

    return sorted(edges)


region_edges = hull_edges(region_hull.simplices)


def edges_to_lines(vertices, edges):
    x, y, z = [], [], []

    for i, j in edges:
        p = vertices[i]
        q = vertices[j]

        x.extend([p[0], q[0], None])
        y.extend([p[1], q[1], None])
        z.extend([p[2], q[2], None])

    return x, y, z


region_edge_x, region_edge_y, region_edge_z = edges_to_lines(
    region_vertices_norm,
    region_edges,
)


# ============================================================
# Original cube and its edges
# ============================================================

cube_vertices = np.array([
    [x, y, z]
    for x in (-h, h)
    for y in (-h, h)
    for z in (-h, h)
])
cube_vertices_norm = cube_vertices / Omega_0

cube_edges = []

for i, p in enumerate(cube_vertices):
    for j, q in enumerate(cube_vertices):
        if i >= j:
            continue

        different_coordinates = np.count_nonzero(
            np.abs(p - q) > tol
        )

        if different_coordinates == 1:
            cube_edges.append((i, j))


cube_edge_x, cube_edge_y, cube_edge_z = edges_to_lines(
    cube_vertices_norm,
    cube_edges,
)


# ============================================================
# Plane section
#
# nu1 - nu2 + nu3 = s
# ============================================================

plane_normal = np.array([1.0, -1.0, 1.0])


def plane_cube_section(s):
    """Section of the cube by the plane nu1 - nu2 + nu3 = s.

    All coordinates are in normalised units (nu_j / Omega_0).
    The parameter s is also normalised.
    """
    points = []

    for i, j in cube_edges:
        p = cube_vertices_norm[i]
        q = cube_vertices_norm[j]

        fp = plane_normal @ p - s
        fq = plane_normal @ q - s

        if abs(fp) < tol and abs(fq) < tol:
            points.extend([p, q])
            continue

        if fp * fq <= 0 and abs(fp - fq) > tol:
            t = fp / (fp - fq)

            if -tol <= t <= 1 + tol:
                points.append(p + t * (q - p))

    unique_points = []

    for p in points:
        if not any(
            np.linalg.norm(p - q) < tol
            for q in unique_points
        ):
            unique_points.append(p)

    points = np.asarray(unique_points)

    if len(points) < 3:
        return points

    normal = plane_normal / np.linalg.norm(plane_normal)

    reference = np.array([1.0, 0.0, 0.0])

    if abs(normal @ reference) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(normal, reference)
    e1 /= np.linalg.norm(e1)

    e2 = np.cross(normal, e1)

    projected = np.column_stack([
        points @ e1,
        points @ e2,
    ])

    hull_2d = ConvexHull(projected)

    return points[hull_2d.vertices]


# ============================================================
# Figure
# ============================================================

fig = go.Figure()


# Transparent admissible polyhedron
fig.add_trace(
    go.Mesh3d(
        x=region_vertices[:, 0],
        y=region_vertices[:, 1],
        z=region_vertices[:, 2],
        i=region_hull.simplices[:, 0],
        j=region_hull.simplices[:, 1],
        k=region_hull.simplices[:, 2],
        color="royalblue",
        opacity=0.16,
        flatshading=True,
        name="Admissible region",
        hoverinfo="skip",
    )
)


# Strong admissible-region edges
fig.add_trace(
    go.Scatter3d(
        x=region_edge_x,
        y=region_edge_y,
        z=region_edge_z,
        mode="lines",
        line=dict(
            color="midnightblue",
            width=7,
        ),
        name="Region edges",
        hoverinfo="skip",
    )
)


# Original cube as a lighter reference
fig.add_trace(
    go.Scatter3d(
        x=cube_edge_x,
        y=cube_edge_y,
        z=cube_edge_z,
        mode="lines",
        line=dict(
            color="gray",
            width=3,
            dash="dash",
        ),
        opacity=0.45,
        name="Original cube",
        hoverinfo="skip",
    )
)


# ============================================================
# Moving sections
# ============================================================

s_values = np.linspace(-h_norm, h_norm, 31)
first_dynamic_trace = len(fig.data)

for index, s in enumerate(s_values):
    section = plane_cube_section(s)

    centroid = section.mean(axis=0)
    mesh_vertices = np.vstack([centroid, section])

    n_boundary = len(section)

    triangle_i = np.zeros(n_boundary, dtype=int)
    triangle_j = np.arange(1, n_boundary + 1)
    triangle_k = np.roll(triangle_j, -1)

    visible = index == len(s_values) // 2

    # Filled section
    fig.add_trace(
        go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=triangle_i,
            j=triangle_j,
            k=triangle_k,
            color="crimson",
            opacity=0.65,
            visible=visible,
            name=f"Section s={s:.3f}",
            hovertemplate=(
                "ν₁/Ω₀ = %{x:.3f}<br>"
                "ν₂/Ω₀ = %{y:.3f}<br>"
                "ν₃/Ω₀ = %{z:.3f}<br>"
                f"ν₁/Ω₀ − ν₂/Ω₀ + ν₃/Ω₀ = {s:.3f}"
                "<extra></extra>"
            ),
        )
    )

    # Thick section boundary
    closed_section = np.vstack([section, section[0]])

    fig.add_trace(
        go.Scatter3d(
            x=closed_section[:, 0],
            y=closed_section[:, 1],
            z=closed_section[:, 2],
            mode="lines+markers",
            line=dict(
                color="darkred",
                width=10,
            ),
            marker=dict(
                color="darkred",
                size=4,
            ),
            visible=visible,
            name="Section boundary",
            hoverinfo="skip",
            showlegend=False,
        )
    )


# ============================================================
# Slider
# ============================================================

steps = []
n_static_traces = first_dynamic_trace
n_total_traces = len(fig.data)

for index, s in enumerate(s_values):
    visibility = [True] * n_static_traces
    visibility += [False] * (n_total_traces - n_static_traces)

    visibility[first_dynamic_trace + 2 * index] = True
    visibility[first_dynamic_trace + 2 * index + 1] = True

    steps.append(
        dict(
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": (
                        "Axonometric section: "
                        f"ν₁/Ω₀ − ν₂/Ω₀ + ν₃/Ω₀ = {s:.3f}"
                    )
                },
            ],
            label=f"{s:.2f}",
        )
    )


# ============================================================
# Orthographic / axonometric camera
# ============================================================

fig.update_layout(
    font=dict(size=18),
    title="Axonometric section: ν₁/Ω₀ − ν₂/Ω₀ + ν₃/Ω₀ = 0",
    scene=dict(
        xaxis=dict(
            title="ν₁/Ω₀",
            range=[-h_norm, h_norm],
            showbackground=False,
            showgrid=True,
            zeroline=True,
        ),
        yaxis=dict(
            title="ν₂/Ω₀",
            range=[-h_norm, h_norm],
            showbackground=False,
            showgrid=True,
            zeroline=True,
        ),
        zaxis=dict(
            title="ν₃/Ω₀",
            range=[-h_norm, h_norm],
            showbackground=False,
            showgrid=True,
            zeroline=True,
        ),
        aspectmode="cube",

        # Orthographic projection: no perspective shortening
        camera=dict(
            projection=dict(type="orthographic"),

            # Isometric-like viewing direction
            eye=dict(
                x=1.55,
                y=1.55,
                z=1.25,
            ),
            up=dict(
                x=0,
                y=0,
                z=1,
            ),
        ),
    ),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(
                    label="Iso",
                    method="relayout",
                    args=[{"scene.camera.eye": dict(x=1.55, y=1.55, z=1.25)}],
                ),
                dict(
                    label="ν₁",
                    method="relayout",
                    args=[{"scene.camera.eye": dict(x=1, y=0, z=0)}],
                ),
                dict(
                    label="ν₂",
                    method="relayout",
                    args=[{"scene.camera.eye": dict(x=0, y=1, z=0)}],
                ),
                dict(
                    label="ν₃",
                    method="relayout",
                    args=[{"scene.camera.eye": dict(x=0, y=0, z=1)}],
                ),
            ],
        )
    ],
    sliders=[
        dict(
            active=len(s_values) // 2,
            currentvalue=dict(
                prefix="s/Ω₀ = ν₁/Ω₀ − ν₂/Ω₀ + ν₃/Ω₀ = ",
            ),
            pad=dict(t=50),
            steps=steps,
        )
    ],
    annotations=[
        dict(
            text=(
                "FWM:  ω₁ − ω₂ + ω₃ = ω₄ ;  "
                "ν₁ − ν₂ + ν₃ = ν₄ ;  "
                "|ν₄| ≤ πB"
            ),
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.05,
            xanchor="center",
            yanchor="top",
            font=dict(size=18),
        )
    ],
    margin=dict(l=0, r=0, b=40, t=60),
    legend=dict(
        x=0.02,
        y=0.98,
    ),
)

fig.show()
