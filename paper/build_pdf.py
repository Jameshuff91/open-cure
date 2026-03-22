"""Build publication PDF from markdown + figures using fpdf2."""

import os
import re
from fpdf import FPDF

PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(PAPER_DIR, "preprint_draft.md")
OUTPUT = os.path.join(PAPER_DIR, "preprint_draft.pdf")

# Figure placement: after which section heading each figure appears
FIGURE_PLACEMENTS = {
    "3.2 Confidence Tier Performance": ("fig1_confidence_tiers.png", "Figure 1: Confidence tier holdout precision. Error bars show standard error. The GOLDEN and HIGH tiers (2,159 predictions combined) achieve >83% precision, providing actionable prioritization for experimental validation."),
    "3.4 Coverage and Rare Disease Performance": ("fig2_rare_disease.png", "Figure 2: Performance by disease rarity. Orphan diseases with a single known treatment achieve 13.5% R@30 versus 32.2% for well-studied diseases (6-10 treatments), a 2.4x gap inherent to collaborative filtering."),
    "3.1 Primary Performance": ("fig3_method_comparison.png", "Figure 3: Method comparison across transductive and inductive evaluation paradigms. The gray bar includes treatment edge leakage (10.5 pp). TxGNN results from Huang et al. (2024)."),
    "Bimodal": ("fig4_bimodal_coverage.png", "Figure 4: Bimodal performance distribution. 84.7% of diseases have kNN neighborhood coverage (24.2% R@30); 15.3% have zero coverage (0% R@30). The overall metric (26.06%) obscures this stark split."),
    "3.3 Performance Ceiling Analysis": ("fig5_ceiling_analysis.png", "Figure 5: Performance ceiling analysis. Our kNN method (36.59% with original embeddings) reaches 62% of the oracle ceiling (60%), establishing that the remaining gap requires external data sources beyond DRKG."),
}


class PaperPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, "Huff (2026) - Confidence-Stratified Drug Repurposing", align="L")
            self.cell(0, 10, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)

    def footer(self):
        pass


def parse_markdown(md_text: str) -> list[dict]:
    """Parse markdown into structured blocks."""
    blocks: list[dict] = []
    lines = md_text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]

        # Headings
        if line.startswith("# ") and not line.startswith("## "):
            blocks.append({"type": "h1", "text": line[2:].strip()})
            i += 1
            continue
        if line.startswith("## "):
            blocks.append({"type": "h2", "text": line[3:].strip()})
            i += 1
            continue
        if line.startswith("### "):
            blocks.append({"type": "h3", "text": line[4:].strip()})
            i += 1
            continue

        # Horizontal rules
        if line.strip() == "---":
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Block quotes
        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith("> "):
                quote_lines.append(lines[i][2:])
                i += 1
            blocks.append({"type": "quote", "text": " ".join(quote_lines)})
            continue

        # Tables
        if "|" in line and i + 1 < len(lines) and re.match(r"^\|[-\s|:]+\|$", lines[i + 1].strip()):
            table_lines = []
            while i < len(lines) and "|" in lines[i]:
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "lines": table_lines})
            continue

        # Math blocks ($$...$$)
        if line.strip().startswith("$$"):
            math_lines = [line]
            i += 1
            while i < len(lines) and not lines[i].strip().endswith("$$"):
                math_lines.append(lines[i])
                i += 1
            if i < len(lines):
                math_lines.append(lines[i])
                i += 1
            math_text = " ".join(l.strip().strip("$") for l in math_lines)
            blocks.append({"type": "math", "text": math_text})
            continue

        # Ordered lists
        if re.match(r"^\d+\.\s", line):
            list_items = []
            while i < len(lines) and re.match(r"^\d+\.\s", lines[i]):
                list_items.append(re.sub(r"^\d+\.\s", "", lines[i]))
                i += 1
            blocks.append({"type": "ol", "items": list_items})
            continue

        # Unordered lists
        if line.startswith("- "):
            list_items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("  ")):
                if lines[i].startswith("- "):
                    list_items.append(lines[i][2:])
                else:
                    if list_items:
                        list_items[-1] += " " + lines[i].strip()
                i += 1
            blocks.append({"type": "ul", "items": list_items})
            continue

        # Empty lines
        if line.strip() == "":
            i += 1
            continue

        # Regular paragraph — collect consecutive non-empty lines
        para_lines = []
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") and not lines[i].startswith("|") and not lines[i].startswith("- ") and not re.match(r"^\d+\.\s", lines[i]) and not lines[i].startswith(">") and lines[i].strip() != "---":
            para_lines.append(lines[i])
            i += 1
        if para_lines:
            blocks.append({"type": "para", "text": " ".join(para_lines)})

    return blocks


def clean_md(text: str) -> str:
    """Strip markdown formatting and replace unicode for latin-1 compat."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.+?)\*", r"\1", text)  # italic
    text = re.sub(r"`(.+?)`", r"\1", text)  # code
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # links
    # Unicode replacements for latin-1 compatibility
    text = text.replace("\u2014", "--")  # em dash
    text = text.replace("\u2013", "-")   # en dash
    text = text.replace("\u2019", "'")   # right single quote
    text = text.replace("\u2018", "'")   # left single quote
    text = text.replace("\u201c", '"')   # left double quote
    text = text.replace("\u201d", '"')   # right double quote
    text = text.replace("\u2022", "-")   # bullet
    text = text.replace("\u2026", "...")  # ellipsis
    text = text.replace("\u00d7", "x")   # multiplication sign
    text = text.replace("\u2265", ">=")  # >=
    text = text.replace("\u2264", "<=")  # <=
    text = text.replace("\u00b1", "+/-") # plus-minus
    text = text.replace("\u2192", "->")  # right arrow
    # Catch any remaining non-latin-1 chars
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def render_rich_text(pdf: PaperPDF, text: str, font_size: int = 10) -> None:
    """Render text with bold/italic markdown formatting."""
    # Split on bold markers
    parts = re.split(r"(\*\*.*?\*\*)", text)
    pdf.set_font("Helvetica", "", font_size)

    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            inner = clean_md(part[2:-2])
            pdf.set_font("Helvetica", "B", font_size)
            pdf.write(5, inner)
            pdf.set_font("Helvetica", "", font_size)
        else:
            cleaned = clean_md(part)
            pdf.write(5, cleaned)


def build_pdf() -> None:
    with open(MD_FILE, encoding="utf-8") as f:
        md_text = f.read()

    blocks = parse_markdown(md_text)

    pdf = PaperPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    placed_figures: set[str] = set()

    for block in blocks:
        btype = block["type"]

        if btype == "h1":
            pdf.set_font("Helvetica", "B", 16)
            pdf.multi_cell(0, 8, clean_md(block["text"]), align="C")
            pdf.ln(4)

        elif btype == "h2":
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, clean_md(block["text"]))
            pdf.ln(2)

            # Check for figure placement after this heading
            for key, (fig_file, caption) in FIGURE_PLACEMENTS.items():
                if key in block["text"] and fig_file not in placed_figures:
                    placed_figures.add(fig_file)
                    fig_path = os.path.join(PAPER_DIR, fig_file)
                    if os.path.exists(fig_path):
                        # Check if we need a new page for the figure
                        if pdf.get_y() > 180:
                            pdf.add_page()
                        pdf.image(fig_path, x=25, w=160)
                        pdf.ln(2)
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.multi_cell(0, 4, clean_md(caption))
                        pdf.ln(4)

        elif btype == "h3":
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, clean_md(block["text"]))
            pdf.ln(1)

            # Check for figure placement after subsection headings
            for key, (fig_file, caption) in FIGURE_PLACEMENTS.items():
                if key in block["text"] and fig_file not in placed_figures:
                    placed_figures.add(fig_file)
                    fig_path = os.path.join(PAPER_DIR, fig_file)
                    if os.path.exists(fig_path):
                        if pdf.get_y() > 180:
                            pdf.add_page()
                        pdf.image(fig_path, x=25, w=160)
                        pdf.ln(2)
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.multi_cell(0, 4, clean_md(caption))
                        pdf.ln(4)

        elif btype == "hr":
            pdf.ln(2)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(2)

        elif btype == "quote":
            pdf.set_fill_color(240, 240, 245)
            pdf.set_font("Helvetica", "I", 9)
            x = pdf.get_x()
            pdf.set_x(x + 10)
            quote_text = clean_md(block["text"])
            pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 20, 5,
                           quote_text, fill=True)
            pdf.ln(3)

        elif btype == "para":
            pdf.set_font("Helvetica", "", 10)
            render_rich_text(pdf, block["text"])
            pdf.ln(6)

        elif btype == "math":
            pdf.set_font("Courier", "", 9)
            pdf.set_fill_color(245, 245, 250)
            pdf.multi_cell(0, 5, clean_md(block["text"]), fill=True, align="C")
            pdf.ln(3)

        elif btype == "ol":
            pdf.set_font("Helvetica", "", 10)
            for idx, item in enumerate(block["items"], 1):
                pdf.set_x(pdf.l_margin + 5)
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 10, 5,
                               f"{idx}. {clean_md(item)}")
                pdf.ln(1)
            pdf.ln(2)

        elif btype == "ul":
            pdf.set_font("Helvetica", "", 10)
            for item in block["items"]:
                pdf.set_x(pdf.l_margin + 5)
                text = clean_md(item)
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 10, 5,
                               f"- {text}")
                pdf.ln(1)
            pdf.ln(2)

        elif btype == "table":
            table_lines = block["lines"]
            # Parse header and data rows
            rows = []
            for tl in table_lines:
                if re.match(r"^\|[-\s|:]+\|$", tl.strip()):
                    continue
                cells = [c.strip() for c in tl.strip().strip("|").split("|")]
                rows.append(cells)

            if not rows:
                continue

            num_cols = len(rows[0])
            usable_w = pdf.w - pdf.l_margin - pdf.r_margin
            col_w = usable_w / num_cols

            # Header row
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(230, 230, 235)
            for cell in rows[0]:
                pdf.cell(col_w, 6, clean_md(cell)[:40], border=1, fill=True, align="C")
            pdf.ln()

            # Data rows
            pdf.set_font("Helvetica", "", 8)
            for row in rows[1:]:
                max_h = 6
                for j, cell in enumerate(row):
                    text = clean_md(cell)[:50]
                    pdf.cell(col_w, max_h, text, border=1, align="C")
                pdf.ln()
            pdf.ln(3)

    # Place any remaining figures at the end
    for key, (fig_file, caption) in FIGURE_PLACEMENTS.items():
        if fig_file not in placed_figures:
            fig_path = os.path.join(PAPER_DIR, fig_file)
            if os.path.exists(fig_path):
                pdf.add_page()
                pdf.image(fig_path, x=25, w=160)
                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 8)
                pdf.multi_cell(0, 4, clean_md(caption))
                placed_figures.add(fig_file)

    pdf.output(OUTPUT)
    print(f"PDF saved to: {OUTPUT}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    build_pdf()
