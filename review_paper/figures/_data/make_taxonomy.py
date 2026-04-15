"""Compression-family taxonomy: clean tree.
Design: Single horizontal "trunk" at the top with a centered root label.
Seven branches descend to colored family headers; under each header,
methods are listed as plain text on individual lines (no nested boxes).
"""
import os
import matplotlib.pyplot as plt, matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "figure.dpi": 150,
})

FAMILIES = [
    ("Backbone\nSubstitution", "#1f77b4",
     ["MobileStereoNet", "Separable-Stereo", "MABNet"]),
    ("Cost-Volume\nCompression", "#2ca02c",
     ["BGNet", "DeepPruner", "Cascade-CV", "ADStereo", "PCVNet"]),
    ("Iterative-Loop\nCompression", "#d62728",
     ["Pip-Stereo (PIP)", "RT-IGEV", "IINet"]),
    ("Knowledge\nDistillation", "#9467bd",
     ["DTP", "Fast-FoundationStereo", "MPT (Pip-Stereo)"]),
    ("Architectural\nCompression", "#ff7f0e",
     ["BANet", "LightStereo", "GGEV", "LiteAnyStereo"]),
    ("Adaptive\nCompute", "#8c564b",
     ["AnyNet", "HITNet", "ADStereo", "DeepPruner"]),
    ("NAS for\nEfficiency", "#17becf",
     ["LEAStereo", "EASNet", "AutoDispNet"]),
]

n = len(FAMILIES)
fig, ax = plt.subplots(figsize=(11.0, 5.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Root: centered text, no box (just bold label)
root_x, root_y = 50, 92
ax.text(root_x, root_y, "Compression of Deep Stereo Networks",
        ha='center', va='center', fontsize=12, color='#2c3e50',
        fontweight='bold')
# Trunk: short horizontal line below the root
ax.plot([15, 85], [88, 88], color='#2c3e50', linewidth=1.4)
ax.plot([root_x, root_x], [root_y - 2.5, 88], color='#2c3e50', linewidth=1.4)

# Branches: family header at top of branch, methods listed below as plain text
xs = [10 + 80 * i / (n - 1) for i in range(n)]
header_y = 82
TEXT_LINE_H = 4.8
for x, (name, color, methods) in zip(xs, FAMILIES):
    # Branch line from trunk to header
    ax.plot([x, x], [88, header_y + 1.5], color=color, linewidth=1.2, alpha=0.85)
    # Family header: just text in color, no box
    ax.text(x, header_y, name, ha='center', va='top',
            fontsize=10, color=color, fontweight='bold')
    # Vertical bar to the left of the methods (color rail)
    methods_top = header_y - 7
    methods_bot = methods_top - len(methods) * TEXT_LINE_H
    ax.plot([x, x], [methods_top + 1.5, methods_bot - 0.5], color=color,
            linewidth=2.2, alpha=0.65)
    # Method names: plain text, one per line
    for j, m in enumerate(methods):
        my = methods_top - (j + 0.5) * TEXT_LINE_H
        ax.text(x + 1.5, my, m, ha='left', va='center',
                fontsize=8.2, color='black')

# Footer caption-like note
ax.text(50, 4,
        "Each leaf is a representative method. Many methods belong to "
        "multiple families (e.g., Pip-Stereo combines Iterative-Loop "
        "Compression and Knowledge Distillation).",
        ha='center', va='bottom', fontsize=7.8, style='italic', color='gray')

plt.tight_layout()
out_pdf = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'fig_taxonomy.pdf')
plt.savefig(out_pdf, bbox_inches='tight', pad_inches=0.18)
plt.savefig(out_pdf.replace('.pdf', '.png'), bbox_inches='tight', pad_inches=0.18, dpi=200)
print("saved", out_pdf)
