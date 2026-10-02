"""Builds docs/reference.docx: pandoc's default styles with heading and title colors removed (size and bold only)."""
import shutil
import subprocess
import tempfile
import zipfile
import re
from pathlib import Path
import pypandoc

# ===== CONFIG (user inputs) =====
OUT = Path(__file__).resolve().parent / "reference.docx"
STYLE_IDS = ("Heading1", "Heading2", "Heading3", "Heading4", "Title", "Subtitle", "Author", "Date", "Abstract")
HEADING_SIZES_HALF_POINTS = {"Heading1": 32, "Heading2": 28, "Heading3": 24, "Heading4": 22, "Title": 40}
# ===== END CONFIG =====


def strip_colors(styles_xml: str) -> str:
    def fix(match):
        block = match.group(0)
        sid = re.search(r'w:styleId="([^"]+)"', block).group(1)
        if sid not in STYLE_IDS:
            return block
        block = re.sub(r"<w:color [^>]*/>", "", block)
        block = re.sub(r"<w:themeColor [^>]*/>", "", block)
        if sid in HEADING_SIZES_HALF_POINTS:
            size = HEADING_SIZES_HALF_POINTS[sid]
            block = re.sub(r'<w:sz w:val="\d+"\s*/>', f'<w:sz w:val="{size}"/>', block)
            block = re.sub(r'<w:szCs w:val="\d+"\s*/>', f'<w:szCs w:val="{size}"/>', block)
        return block
    return re.sub(r"<w:style [^>]*>.*?</w:style>", fix, styles_xml, flags=re.S)


def main():
    work = Path(tempfile.mkdtemp())
    default = work / "default.docx"
    # pandoc prints its default reference.docx
    default.write_bytes(subprocess.run([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"],
                                       capture_output=True, check=True).stdout)
    with zipfile.ZipFile(default) as zi, zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zi.infolist():
            data = zi.read(item.filename)
            if item.filename == "word/styles.xml":
                data = strip_colors(data.decode("utf-8")).encode("utf-8")
            zo.writestr(item, data)
    with zipfile.ZipFile(OUT) as z:
        styles = z.read("word/styles.xml").decode("utf-8")
    for sid in STYLE_IDS:
        block = re.search(rf'<w:style [^>]*w:styleId="{sid}".*?</w:style>', styles, flags=re.S)
        assert block is None or "<w:color" not in block.group(0), f"color left in {sid}"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
