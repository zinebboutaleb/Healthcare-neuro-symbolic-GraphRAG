"""Render ontology/ontology_diagram.png from src/ontology.py.   python ontology/make_diagram.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.ontology import SCHEMA

POS = {"RiskFactor": (0, 6), "Symptom": (0, 4), "DiagnosticTest": (0, 2), "Disease": (4, 4),
       "Medication": (8, 4), "Treatment": (8, 6.2), "SideEffect": (12, 5.8), "PatientCondition": (12, 1),
       "Contraindication": (8, 1), "ClinicalGuideline": (4, 0.2)}
COLOR = {"Disease": "#f4b6b6", "Symptom": "#fde3a7", "RiskFactor": "#fde3a7", "DiagnosticTest": "#b8dff0",
         "Medication": "#c5e8c0", "Treatment": "#c5e8c0", "SideEffect": "#e0cdf0", "PatientCondition": "#f0d0b0",
         "Contraindication": "#f0d0b0", "ClinicalGuideline": "#d8d8d8"}
fig, ax = plt.subplots(figsize=(15, 8)); ax.set_xlim(-1.5, 13.5); ax.set_ylim(-0.8, 7.4); ax.axis("off")
for n, (x, y) in POS.items():
    ax.add_patch(FancyBboxPatch((x - 1.15, y - 0.35), 2.3, 0.7, boxstyle="round,pad=0.05", fc=COLOR[n], ec="#444"))
    ax.text(x, y, n, ha="center", va="center", fontsize=10, weight="bold")
offset = {}
selfloops = {}
for rel, pairs in SCHEMA.items():
    for a, b in pairs:
        if a == b:  # self-relation: draw a small loop above the node
            x, y = POS[a]; n = selfloops.get(a, 0); selfloops[a] = n + 1
            ax.annotate("", xy=(x + 0.5, y + 0.4), xytext=(x - 0.5, y + 0.4), arrowprops=dict(arrowstyle="->", color="#555", lw=1.1, connectionstyle="arc3,rad=-1.6"))
            ax.text(x, y + 0.95 + 0.25 * n, rel + " (self)", fontsize=7, ha="center", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
            continue
        (x1, y1), (x2, y2) = POS[a], POS[b]
        k = tuple(sorted((a, b))); offset[k] = offset.get(k, 0) + 1
        rad = 0.0 if a != b and offset[k] == 1 else 0.25 * offset[k]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#555", lw=1.1,
                    connectionstyle=f"arc3,rad={rad}", shrinkA=26, shrinkB=26))
        t = 0.5 - 0.06 * (offset[k] - 1)
        ax.text(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t + 0.12 + 0.18 * (offset[k] - 1), rel, fontsize=7, ha="center",
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
ax.set_title("Clinical knowledge-graph ontology (classes and allowed relations)", fontsize=13)
plt.tight_layout(); plt.savefig(Path(__file__).parent / "ontology_diagram.png", dpi=130)
print("saved")
