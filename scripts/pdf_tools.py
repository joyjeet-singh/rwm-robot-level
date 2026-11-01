"""PDF text, page count, metadata and font embedding.

Uses the poppler tools (pdftotext, pdfinfo, pdffonts) when all three are on PATH; otherwise pypdf
(text, page count, metadata and embedded fonts only). Added in revision round 1 (R0): this Mac has no poppler.
"""
import re, shutil, subprocess

POPPLER = all(shutil.which(t) for t in ("pdftotext", "pdfinfo", "pdffonts"))


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def _fonts(reader):
    """(name, embedded) for every font used on any page."""
    seen = {}
    for page in reader.pages:
        res = page.get("/Resources")
        res = res.get_object() if res is not None else {}
        fonts = res.get("/Font")
        for ref in (fonts.get_object().values() if fonts is not None else []):
            f = ref.get_object()
            desc = f.get("/FontDescriptor")
            if desc is None and "/DescendantFonts" in f:
                desc = f["/DescendantFonts"][0].get_object().get("/FontDescriptor")
            d = desc.get_object() if desc is not None else {}
            seen[str(f.get("/BaseFont"))] = any(k in d for k in ("/FontFile", "/FontFile2", "/FontFile3"))
    return seen


def pdf_info(path):
    """{'text', 'pages', 'meta' {Title, Author, Creator, Producer}, 'fonts_not_embedded' [names]}"""
    if POPPLER:
        text = _run(["pdftotext", "-layout", path, "-"])
        info = _run(["pdfinfo", path])
        meta = {k.strip(): v.strip() for k, v in (ln.split(":", 1) for ln in info.splitlines() if ":" in ln)}
        fonts = _run(["pdffonts", path]).splitlines()[2:]
        bad = [ln for ln in fonts if ln.split()[-5:-4] == ["no"]]
        return {"text": text, "pages": int(meta.get("Pages", 0)), "info": info,
                "meta": {k: meta.get(k, "") for k in ("Title", "Author", "Creator", "Producer")},
                "fonts_not_embedded": bad}
    from pypdf import PdfReader
    r = PdfReader(path)
    md = r.metadata or {}
    meta = {k: str(md.get("/" + k) or "") for k in ("Title", "Author", "Creator", "Producer")}
    text = "\n".join(p.extract_text() or "" for p in r.pages)
    return {"text": text, "pages": len(r.pages), "info": "\n".join(f"{k}: {v}" for k, v in meta.items()),
            "meta": meta, "fonts_not_embedded": [n for n, ok in _fonts(r).items() if not ok]}


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def nospace(text):
    return re.sub(r"\s+", "", text)
