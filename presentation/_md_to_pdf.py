"""Convert V5_Presentation_Script.md to a sibling PDF.

Pipeline: markdown -> styled HTML -> LibreOffice headless -> PDF.

We avoid pandoc / weasyprint because neither is installed on this
machine; LibreOffice is already used to convert the .pptx decks so it
is the safe dependency.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "script" / "V5_Presentation_Script.md"
OUT_PDF = ROOT / "script" / "V5_Presentation_Script.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body  { font-family: 'Liberation Serif', 'DejaVu Serif', serif;
        font-size: 11pt; line-height: 1.45; color: #111; }
h1    { font-size: 22pt; color: #14385C; margin: 0 0 8pt 0; }
h2    { font-size: 15pt; color: #14385C; margin: 18pt 0 6pt 0;
        border-bottom: 1pt solid #C24A1C; padding-bottom: 2pt; }
h3    { font-size: 13pt; color: #14385C; margin: 12pt 0 4pt 0; }
h4    { font-size: 11pt; color: #C24A1C; margin: 8pt 0 2pt 0;
        text-transform: uppercase; letter-spacing: 0.4pt; }
p     { margin: 4pt 0; }
strong { color: #14385C; }
em    { color: #5A5550; }
hr    { border: 0; border-top: 0.5pt solid #BFBAB0; margin: 10pt 0; }
blockquote { background: #FBF7EE; border-left: 2pt solid #C24A1C;
             padding: 6pt 10pt; margin: 6pt 0; color: #1A1A1F;
             font-size: 11pt; line-height: 1.5; }
ul    { margin: 4pt 0 4pt 20pt; padding: 0; }
li    { margin: 2pt 0; }
table { border-collapse: collapse; margin: 8pt 0;
        font-size: 10pt; width: auto; }
th, td { border: 0.5pt solid #BFBAB0; padding: 4pt 6pt;
         text-align: left; vertical-align: top; }
th    { background: #14385C; color: #FFF; font-weight: bold; }
tr:nth-child(even) td { background: #FBF7EE; }
code  { font-family: 'Liberation Mono', 'DejaVu Sans Mono', monospace;
        font-size: 10pt; color: #14385C;
        background: #F4EFE6; padding: 0 3pt; border-radius: 2pt; }
"""

# Wrap each "**Q.  ..."  paragraph (the question lines we use in the
# Q & A section) so it gets styled like a small accent header. We do
# this by post-processing the rendered HTML rather than using a
# markdown extension, to keep the markdown file readable.
def _emphasize_questions(html: str) -> str:
    import re
    return re.sub(
        r"<p><strong>(Q\.[^<]+?)</strong>",
        r'<p class="qa-q"><strong>\1</strong>',
        html,
    )


def _fix_schedule_table(html: str) -> str:
    """Inject a <colgroup> with explicit widths into the first <table>.
    LibreOffice's HTML import does not honour `white-space: nowrap` on
    table headers, so we set the column widths up front to stop the
    headers from wrapping. The first table in the doc is the schedule
    (5 columns: # | Slide | Time | Speaker | Words)."""
    # LibreOffice's HTML import does not honour CSS table-layout, but
    # it does respect the legacy HTML `width=` attribute on <col>.
    colgroup = (
        "<colgroup>"
        "<col width='40'>"
        "<col width='240'>"
        "<col width='80'>"
        "<col width='100'>"
        "<col width='80'>"
        "</colgroup>"
    )
    return html.replace(
        "<table>",
        f"<table width='540' cellspacing='0'>{colgroup}",
        1,
    )


def main() -> int:
    if not SRC.exists():
        print(f"missing source: {SRC}", file=sys.stderr)
        return 1

    md_text = SRC.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "sane_lists"],
        output_format="html5",
    )
    html_body = _emphasize_questions(html_body)
    html_body = _fix_schedule_table(html_body)

    css = CSS + """
.qa-q strong { color: #C24A1C; }
.qa-q { margin-top: 8pt; }
"""

    full_html = (
        "<!DOCTYPE html><html><head>"
        "<meta charset='utf-8'>"
        f"<title>{SRC.stem}</title>"
        f"<style>{css}</style>"
        "</head><body>"
        f"{html_body}"
        "</body></html>"
    )

    # LibreOffice converts HTML to PDF. We write the HTML into a temp
    # file with a stable basename so the resulting PDF has a clean
    # name we can rename to the final target afterwards.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        html_path = td / "V5_Presentation_Script.html"
        html_path.write_text(full_html, encoding="utf-8")

        soffice = shutil.which("libreoffice") or shutil.which("soffice")
        if soffice is None:
            print("libreoffice/soffice not on PATH", file=sys.stderr)
            return 2

        if OUT_PDF.exists():
            OUT_PDF.unlink()

        r = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf",
             str(html_path), "--outdir", str(td)],
            capture_output=True, text=True, timeout=180,
        )
        produced = td / "V5_Presentation_Script.pdf"
        if not produced.exists():
            msg = (r.stderr.strip() or r.stdout.strip()
                    or "no error message")
            print(f"PDF build FAILED: {msg[:400]}", file=sys.stderr)
            return 3
        shutil.move(str(produced), str(OUT_PDF))

    size_mb = OUT_PDF.stat().st_size / 1e6
    print(f"saved {OUT_PDF.name} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
