#!/usr/bin/env python

from pathlib import Path


OUT = Path("docs/assets/simamba_architecture.svg")


def box(x, y, w, h, label, fill="#f8fafc", stroke="#334155"):
    lines = label.split("\n")
    text = []
    start = y + h / 2 - (len(lines) - 1) * 8
    for i, line in enumerate(lines):
        text.append(
            f'<text x="{x + w / 2}" y="{start + 16 * i}" '
            f'text-anchor="middle" dominant-baseline="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="13" fill="#0f172a">{line}</text>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        + "".join(text)
    )


def line(x1, y1, x2, y2, label=None):
    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    text = ""
    if label:
        text = (
            f'<text x="{midx}" y="{midy - 7}" text-anchor="middle" '
            f'font-family="Inter, Arial, sans-serif" font-size="11" fill="#475569">{label}</text>'
        )
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#475569" stroke-width="1.5" marker-end="url(#arrow)"/>'
        + text
    )


def main():
    width, height = 1180, 760
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#475569"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="590" y="36" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="24" font-weight="700" fill="#0f172a">Simamba Language Model Block</text>',
        '<text x="590" y="62" text-anchor="middle" font-family="Inter, Arial, sans-serif" font-size="13" fill="#475569">Training path used in the HPML experiments; local convolution is optional and disabled by default.</text>',
    ]

    svg += [
        box(40, 120, 150, 58, "Token IDs", "#eef2ff", "#4338ca"),
        box(240, 120, 170, 58, "Embedding\n(B, L, d_model)", "#eef2ff", "#4338ca"),
        box(460, 120, 180, 58, "Residual block\nAdd + RMSNorm", "#f8fafc", "#334155"),
        box(700, 120, 190, 58, "Simamba mixer", "#ecfeff", "#0891b2"),
        box(940, 120, 170, 58, "Out projection\nLM head", "#eef2ff", "#4338ca"),
        line(190, 149, 240, 149),
        line(410, 149, 460, 149),
        line(640, 149, 700, 149),
        line(890, 149, 940, 149),
    ]

    svg += [
        box(70, 250, 160, 58, "Hidden states\nu", "#f1f5f9", "#475569"),
        box(280, 250, 190, 58, "Linear in_proj", "#f1f5f9", "#475569"),
        box(520, 225, 170, 58, "z gate", "#fff7ed", "#c2410c"),
        box(520, 300, 170, 58, "x / B / C stream", "#ecfeff", "#0891b2"),
        box(520, 375, 170, 58, "dt, A, coeffs,\nrotary angles", "#fefce8", "#a16207"),
        line(230, 279, 280, 279),
        line(470, 279, 520, 254),
        line(470, 279, 520, 329),
        line(470, 279, 520, 404),
    ]

    svg += [
        box(745, 300, 210, 58, "Optional depthwise\ncausal Conv1d over x/B/C", "#dcfce7", "#15803d"),
        box(1000, 300, 130, 58, "SiLU", "#dcfce7", "#15803d"),
        line(690, 329, 745, 329, "d_conv=4"),
        line(955, 329, 1000, 329),
    ]

    svg += [
        box(170, 515, 190, 70, "Normalize B and C\nRMSNormGated", "#f1f5f9", "#475569"),
        box(430, 500, 210, 100, "SISO recurrence\nSimpson or trapezoid\nwith rotary state", "#e0f2fe", "#0369a1"),
        box(710, 515, 190, 70, "D skip + optional\nz gate / out norm", "#fff7ed", "#c2410c"),
        box(970, 515, 150, 70, "Mixer output", "#f1f5f9", "#475569"),
        line(1065, 358, 265, 515, "x, B, C"),
        line(690, 404, 520, 500, "dt, A, coeffs, angles"),
        line(690, 254, 805, 515, "z"),
        line(360, 550, 430, 550),
        line(640, 550, 710, 550),
        line(900, 550, 970, 550),
    ]

    svg += [
        '<text x="60" y="685" font-family="Inter, Arial, sans-serif" font-size="13" font-weight="700" fill="#0f172a">Key implementation details</text>',
        '<text x="60" y="710" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">1. d_conv=0 exactly preserves the original Simamba mixer path.</text>',
        '<text x="60" y="730" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">2. d_conv=4 matches Mamba2-style local mixing over x/B/C before the SSM recurrence.</text>',
        '<text x="620" y="710" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">3. Simpson uses a learned correction coefficient with a negative lag-2 term.</text>',
        '<text x="620" y="730" font-family="Inter, Arial, sans-serif" font-size="12" fill="#334155">4. The matched trapezoid baseline uses the same projections and output path.</text>',
        "</svg>",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(svg) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
