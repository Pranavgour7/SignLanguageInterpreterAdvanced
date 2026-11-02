import pickle
import json
import numpy as np

# Load original training dataset
d = pickle.load(open('data.pickle', 'rb'))
labels = d['labels']
data = [x[:42] for x in d['data']]

connections = [
    [0,1],[1,2],[2,3],[3,4],       # Thumb
    [0,5],[5,6],[6,7],[7,8],       # Index
    [0,9],[9,10],[10,11],[11,12],  # Middle
    [0,13],[13,14],[14,15],[15,16],# Ring
    [0,17],[17,18],[18,19],[19,20],# Pinky
    [5,9],[9,13],[13,17]           # Palm transversal
]

svg_dict = {}
poses_3d = {}

for label in sorted(list(set(labels))):
    samples = np.array([data[i] for i, l in enumerate(labels) if l == label])
    avg = np.mean(samples, axis=0) # 42 floats
    xs = avg[0::2].copy()
    ys = avg[1::2].copy()

    # ─────────────────────────────────────────────────────────────
    # 1. 2D SVG GENERATION (Direct ground truth from training data)
    # ─────────────────────────────────────────────────────────────
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)
    cx_2d = (min_x + max_x) / 2.0
    cy_2d = (min_y + max_y) / 2.0
    span_2d = max(max_x - min_x, max_y - min_y, 1e-4)
    target_span = 240.0
    scale_2d = target_span / span_2d

    # SVG coords (viewBox 0 0 320 320, centered at 160, 160)
    svg_xs = 160.0 + (xs - cx_2d) * scale_2d
    svg_ys = 160.0 + (ys - cy_2d) * scale_2d

    svg_parts = []
    svg_parts.append('<svg viewBox="0 0 320 320" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">')
    svg_parts.append('<defs>')
    svg_parts.append('  <linearGradient id="palmGrad" x1="0" y1="1" x2="0" y2="0"><stop offset="0%" stop-color="#1e293b" stop-opacity="0.25"/><stop offset="100%" stop-color="#3b82f6" stop-opacity="0.12"/></linearGradient>')
    svg_parts.append('</defs>')

    # Palm polygon fill
    palm_pts = f"{svg_xs[0]:.1f},{svg_ys[0]:.1f} {svg_xs[1]:.1f},{svg_ys[1]:.1f} {svg_xs[5]:.1f},{svg_ys[5]:.1f} {svg_xs[9]:.1f},{svg_ys[9]:.1f} {svg_xs[13]:.1f},{svg_ys[13]:.1f} {svg_xs[17]:.1f},{svg_ys[17]:.1f}"
    svg_parts.append(f'<polygon points="{palm_pts}" fill="url(#palmGrad)" stroke="rgba(59,130,246,0.3)" stroke-width="1.5" stroke-dasharray="3,3" />')

    # Draw bones
    for b_idx, (p1, p2) in enumerate(connections):
        stroke_color = '#3b82f6'
        stroke_w = 7
        if p2 <= 4:
            stroke_color = '#f59e0b' # Thumb (Amber)
            stroke_w = 9
        elif p2 <= 8:
            stroke_color = '#2563eb' # Index (Sapphire Blue)
            stroke_w = 8
        elif p2 <= 12:
            stroke_color = '#6366f1' # Middle (Indigo)
            stroke_w = 8
        elif p2 <= 16:
            stroke_color = '#10b981' # Ring (Emerald)
            stroke_w = 7
        elif p2 <= 20:
            stroke_color = '#ec4899' # Pinky (Rose)
            stroke_w = 6.5
        else: # Palm bridge
            stroke_color = '#64748b'
            stroke_w = 4.5

        svg_parts.append(f'<line x1="{svg_xs[p1]:.1f}" y1="{svg_ys[p1]:.1f}" x2="{svg_xs[p2]:.1f}" y2="{svg_ys[p2]:.1f}" stroke="{stroke_color}" stroke-width="{stroke_w}" stroke-linecap="round" />')

    # Draw joint nodes
    for i in range(21):
        is_tip = (i in [4, 8, 12, 16, 20])
        is_wrist = (i == 0)
        r = 6.5 if is_wrist else (5.5 if is_tip else 4.0)
        fill = '#10b981' if is_tip else ('#0f172a' if is_wrist else '#f8fafc')
        stroke = '#ffffff' if is_tip or is_wrist else '#475569'
        svg_parts.append(f'<circle cx="{svg_xs[i]:.1f}" cy="{svg_ys[i]:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2" />')

    svg_parts.append('</svg>')
    svg_dict[label] = ''.join(svg_parts)

    # ─────────────────────────────────────────────────────────────
    # 2. 3D POSE GENERATION (Synchronized 1:1 with 2D coordinates)
    # ─────────────────────────────────────────────────────────────
    # Invert Y for 3D coordinates (Y up)
    ys_3d = -ys
    cx_3d = (np.min(xs) + np.max(xs)) / 2.0
    cy_3d = (np.min(ys_3d) + np.max(ys_3d)) / 2.0
    xs_3d = xs - cx_3d
    ys_3d = ys_3d - cy_3d

    span_3d = max(np.max(xs_3d) - np.min(xs_3d), np.max(ys_3d) - np.min(ys_3d), 1e-4)
    scale_3d = 2.2 / span_3d
    xs_3d = xs_3d * scale_3d
    ys_3d = ys_3d * scale_3d

    # Calculate anatomical depth Z matching true hand kinematics
    zs_3d = np.zeros(21)
    zs_3d[0] = 0.0     # Wrist
    zs_3d[1] = 0.05    # Thumb CMC
    zs_3d[5] = 0.02    # Index MCP
    zs_3d[9] = 0.04    # Middle MCP
    zs_3d[13] = 0.02   # Ring MCP
    zs_3d[17] = -0.02  # Pinky MCP

    # Thumb extension
    thumb_dist = np.linalg.norm([xs_3d[4]-xs_3d[1], ys_3d[4]-ys_3d[1]])
    if thumb_dist > 0.75:
        zs_3d[2] = 0.10
        zs_3d[3] = 0.14
        zs_3d[4] = 0.12
    else:
        zs_3d[2] = 0.16
        zs_3d[3] = 0.25
        zs_3d[4] = 0.22

    # 4 fingers curl check
    for mcp, pip, dip, tip in [(5,6,7,8), (9,10,11,12), (13,14,15,16), (17,18,19,20)]:
        dist = np.linalg.norm([xs_3d[tip]-xs_3d[mcp], ys_3d[tip]-ys_3d[mcp]])
        if dist > 0.85: # Extended upright
            zs_3d[pip] = zs_3d[mcp] + 0.04
            zs_3d[dip] = zs_3d[mcp] + 0.06
            zs_3d[tip] = zs_3d[mcp] + 0.08
        elif dist < 0.45: # Curled tight fist
            zs_3d[pip] = zs_3d[mcp] + 0.32
            zs_3d[dip] = zs_3d[mcp] + 0.36
            zs_3d[tip] = zs_3d[mcp] + 0.14
        else: # Hook / Arc
            zs_3d[pip] = zs_3d[mcp] + 0.28
            zs_3d[dip] = zs_3d[mcp] + 0.32
            zs_3d[tip] = zs_3d[mcp] + 0.22

    cz_3d = np.mean(zs_3d)
    zs_3d = zs_3d - cz_3d

    points_3d = [[float(round(xs_3d[i], 3)), float(round(ys_3d[i], 3)), float(round(zs_3d[i], 3))] for i in range(21)]
    poses_3d[label] = points_3d

# Save 2D SVGs
with open('web/signs_2d_svg.json', 'w') as f:
    json.dump(svg_dict, f)

# Save 3D Poses
with open('web/poses_3d.json', 'w') as f:
    json.dump(poses_3d, f, indent=2)

print(f"Generated 2D and 3D files for all {len(svg_dict)} classes directly from training data!")

