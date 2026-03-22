"""Inject academic CSS and figures into pandoc HTML output."""
import re

HTML_FILE = "preprint_draft.html"

ACADEMIC_CSS = """  <style>
    @page { size: A4; margin: 2.5cm; }
    @media print {
      body { font-size: 10pt; }
      h1 { font-size: 16pt; }
      h2 { font-size: 13pt; page-break-after: avoid; }
      h3 { font-size: 11pt; page-break-after: avoid; }
      table { page-break-inside: avoid; }
      .figure { page-break-inside: avoid; }
    }
    body {
      font-family: "Times New Roman", Georgia, serif;
      max-width: 750px;
      margin: 0 auto;
      padding: 2rem;
      line-height: 1.5;
      color: #1a1a1a;
      font-size: 11pt;
    }
    h1 {
      font-size: 18pt;
      text-align: center;
      margin-bottom: 0.5em;
      line-height: 1.3;
    }
    h1 + p, h1 + p + p, h1 + p + p + p {
      text-align: center;
    }
    h2 {
      font-size: 14pt;
      border-bottom: 1px solid #ccc;
      padding-bottom: 0.2em;
      margin-top: 1.8em;
    }
    h3 { font-size: 12pt; margin-top: 1.4em; }
    p { text-align: justify; margin: 0.6em 0; }
    blockquote {
      border-left: 3px solid #2980b9;
      background: #f0f5fa;
      padding: 0.8em 1.2em;
      margin: 1em 0;
      font-size: 10pt;
      color: #333;
    }
    blockquote p { text-align: left; }
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 1em 0;
      font-size: 9.5pt;
    }
    th, td {
      border: 1px solid #ccc;
      padding: 0.4em 0.6em;
      text-align: center;
    }
    th {
      background: #f0f0f5;
      font-weight: bold;
    }
    tr:nth-child(even) { background: #fafafa; }
    code { white-space: pre-wrap; font-size: 9pt; background: #f5f5f5; padding: 0.1em 0.3em; border-radius: 2px; }
    .display.math { display: block; text-align: center; margin: 0.5rem auto; font-family: monospace; background: #f8f8fc; padding: 0.5em; }
    hr { border: none; border-top: 1px solid #ddd; margin: 1.5em 0; }
    .figure {
      text-align: center;
      margin: 1.5em 0;
    }
    .figure img {
      max-width: 100%;
      height: auto;
    }
    .figure .caption {
      font-size: 9pt;
      font-style: italic;
      color: #555;
      margin-top: 0.5em;
      text-align: left;
      padding: 0 0.5em;
    }
    .figure-pair {
      display: flex;
      gap: 1em;
      margin: 1.5em 0;
      page-break-inside: avoid;
    }
    .figure-pair .figure {
      flex: 1;
      margin: 0;
    }
    .figure-pair .figure img {
      width: 100%;
    }
    .figure-pair .figure:only-child {
      max-width: 85%;
      margin: 0 auto;
    }
    ul, ol { margin: 0.5em 0; padding-left: 2em; }
    li { margin: 0.3em 0; }
  </style>"""

FIGURE_INSERTIONS = [
    ("well-captured by frequency-based ranking.",
     '<div class="figure-pair"><div class="figure"><img src="fig3_method_comparison.png" alt="Figure 3"><p class="caption"><strong>Figure 3.</strong> Method comparison across transductive and inductive evaluation paradigms. The gray bar includes treatment edge leakage (10.5 pp). TxGNN results from Huang et al. (2024).</p></div></div>'),

    ("rather than reviewing all 13,416\npredictions.",
     '<div class="figure-pair"><div class="figure"><img src="fig1_confidence_tiers.png" alt="Figure 1"><p class="caption"><strong>Figure 1.</strong> Confidence tier holdout precision. Error bars show standard error. GOLDEN and HIGH tiers (2,159 predictions combined) achieve &gt;83% precision.</p></div></div>'),

    ("rather than methodological refinement.",
     '<div class="figure-pair"><div class="figure"><img src="fig5_ceiling_analysis.png" alt="Figure 5"><p class="caption"><strong>Figure 5.</strong> Performance ceiling analysis. Our kNN method reaches 62% of the oracle ceiling (60%), establishing that remaining gains require external data sources beyond DRKG.</p></div></div>'),

    ("sufficient neighborhood signal to generate\npredictions.",
     '<div class="figure-pair"><div class="figure"><img src="fig4_bimodal_coverage.png" alt="Figure 4"><p class="caption"><strong>Figure 4.</strong> Bimodal performance distribution. 84.7% of diseases have kNN coverage (24.2% R@30); 15.3% have zero coverage.</p></div><div class="figure"><img src="fig2_rare_disease.png" alt="Figure 2"><p class="caption"><strong>Figure 2.</strong> Performance by disease rarity. Orphan diseases achieve 13.5% R@30 versus 32.2% for well-studied diseases, a 2.4x gap.</p></div></div>'),
]

with open(HTML_FILE, encoding="utf-8") as f:
    html = f.read()

# Replace pandoc default style
style_match = re.search(r"<style>.*?</style>", html, re.DOTALL)
if style_match:
    html = html[:style_match.start()] + ACADEMIC_CSS + html[style_match.end():]

# Remove empty stylesheet link
html = html.replace('<link rel="stylesheet" href="" />', '')

# Insert figures after anchor text
for anchor, figure_html in FIGURE_INSERTIONS:
    # Find the paragraph containing the anchor text
    anchor_clean = anchor.replace("\n", "\\s*")
    match = re.search(rf"{anchor_clean}\s*</p>", html)
    if match:
        insert_pos = match.end()
        html = html[:insert_pos] + "\n" + figure_html + "\n" + html[insert_pos:]
        print(f"  Inserted figure after: {anchor[:50]}...")
    else:
        print(f"  WARNING: Could not find anchor: {anchor[:50]}...")

with open(HTML_FILE, "w", encoding="utf-8") as f:
    f.write(html)

print("Done - HTML updated with CSS and figures")
