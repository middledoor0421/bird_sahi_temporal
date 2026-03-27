from __future__ import annotations

import argparse
import struct
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NS = {
    "w": W_NS,
    "r": R_NS,
    "wp": WP_NS,
    "a": A_NS,
}

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("wp", WP_NS)
ET.register_namespace("a", A_NS)

EQUATION_TEXT_BY_RID = {
    "rId2": "d_t = D(x_t).",
    "rId3": "pi(h_t, B_t) = (a_t, R_t),",
    "rId4": "h_t = (d_t, z_t, m_t, f_t),",
    "rId5": "B_t = (B_t^(empty), B_t^(pos)),",
    "rId6": "max_pi J(pi) = U_disc(pi) - lambda_exp U_exp(pi) - lambda_cost U_cost(pi).",
    "rId7": "u_t_tilde = w_det r_det,t + w_track r_track,t + w_lost r_lost,t + w_motion r_motion,t + w_diff r_diff,t,",
    "rId8": "u_t = clip(u_t_tilde, 0, 1).",
    "rId9": "a_t = 1[u_t > tau(B_t, h_t)].",
    "rId10": "B_(t+1) = F(B_t, a_t, h_t),",
    "rId11": "R_t = TopK(C_t; q(.; h_t)),",
    "rId12": "q(g; h_t) = alpha Diff_t(g) + beta Motion_t(g) + gamma TrackOverlap_t(g),",
    "rId13": "u_t_tilde = 0.45 r_det + 0.45 r_track + 0.25 r_lost + 0.05 r_motion + 0.05 r_diff;  u_t = clip(u_t_tilde, 0, 1),",
}


def ensure_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag, NS)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def set_center_alignment(paragraph: ET.Element) -> None:
    ppr = ensure_child(paragraph, f"{{{W_NS}}}pPr")
    jc = ppr.find(f"{{{W_NS}}}jc")
    if jc is None:
        jc = ET.SubElement(ppr, f"{{{W_NS}}}jc")
    jc.set(f"{{{W_NS}}}val", "center")


def replace_paragraph_with_text(paragraph: ET.Element, text: str) -> None:
    ppr = ensure_child(paragraph, f"{{{W_NS}}}pPr")
    pstyle = ppr.find(f"{{{W_NS}}}pStyle")
    if pstyle is None:
        pstyle = ET.SubElement(ppr, f"{{{W_NS}}}pStyle")
    pstyle.set(f"{{{W_NS}}}val", "TextBody")
    set_center_alignment(paragraph)
    spacing = ppr.find(f"{{{W_NS}}}spacing")
    if spacing is None:
        spacing = ET.SubElement(ppr, f"{{{W_NS}}}spacing")
    spacing.set(f"{{{W_NS}}}before", "40")
    spacing.set(f"{{{W_NS}}}after", "40")

    for child in list(paragraph):
        if child.tag != f"{{{W_NS}}}pPr":
            paragraph.remove(child)

    run = ET.SubElement(paragraph, f"{{{W_NS}}}r")
    rpr = ET.SubElement(run, f"{{{W_NS}}}rPr")
    for tag in (f"{{{W_NS}}}sz", f"{{{W_NS}}}szCs"):
        sz = ET.SubElement(rpr, tag)
        sz.set(f"{{{W_NS}}}val", "22")
    ET.SubElement(rpr, f"{{{W_NS}}}i")
    t = ET.SubElement(run, f"{{{W_NS}}}t")
    t.text = text


def apply_table_borders(table: ET.Element) -> None:
    tbl_pr = ensure_child(table, f"{{{W_NS}}}tblPr")
    tbl_jc = tbl_pr.find(f"{{{W_NS}}}jc")
    if tbl_jc is None:
        tbl_jc = ET.SubElement(tbl_pr, f"{{{W_NS}}}jc")
    tbl_jc.set(f"{{{W_NS}}}val", "center")

    tbl_cell_mar = tbl_pr.find(f"{{{W_NS}}}tblCellMar")
    if tbl_cell_mar is None:
        tbl_cell_mar = ET.SubElement(tbl_pr, f"{{{W_NS}}}tblCellMar")
    for name, val in {"top": "45", "left": "80", "bottom": "45", "right": "80"}.items():
        edge = tbl_cell_mar.find(f"{{{W_NS}}}{name}")
        if edge is None:
            edge = ET.SubElement(tbl_cell_mar, f"{{{W_NS}}}{name}")
        edge.set(f"{{{W_NS}}}w", val)
        edge.set(f"{{{W_NS}}}type", "dxa")

    tbl_borders = tbl_pr.find(f"{{{W_NS}}}tblBorders")
    if tbl_borders is None:
        tbl_borders = ET.SubElement(tbl_pr, f"{{{W_NS}}}tblBorders")

    for name in ("top", "bottom"):
        edge = tbl_borders.find(f"{{{W_NS}}}{name}")
        if edge is None:
            edge = ET.SubElement(tbl_borders, f"{{{W_NS}}}{name}")
        edge.set(f"{{{W_NS}}}val", "single")
        edge.set(f"{{{W_NS}}}sz", "8")
        edge.set(f"{{{W_NS}}}space", "0")
        edge.set(f"{{{W_NS}}}color", "666666")

    edge = tbl_borders.find(f"{{{W_NS}}}insideH")
    if edge is None:
        edge = ET.SubElement(tbl_borders, f"{{{W_NS}}}insideH")
    edge.set(f"{{{W_NS}}}val", "single")
    edge.set(f"{{{W_NS}}}sz", "4")
    edge.set(f"{{{W_NS}}}space", "0")
    edge.set(f"{{{W_NS}}}color", "A6A6A6")

    for name in ("left", "right", "insideV"):
        edge = tbl_borders.find(f"{{{W_NS}}}{name}")
        if edge is None:
            edge = ET.SubElement(tbl_borders, f"{{{W_NS}}}{name}")
        edge.set(f"{{{W_NS}}}val", "nil")


def set_cell_width(cell: ET.Element, width: int) -> None:
    tc_pr = ensure_child(cell, f"{{{W_NS}}}tcPr")
    tc_w = tc_pr.find(f"{{{W_NS}}}tcW")
    if tc_w is None:
        tc_w = ET.SubElement(tc_pr, f"{{{W_NS}}}tcW")
    tc_w.set(f"{{{W_NS}}}w", str(width))
    tc_w.set(f"{{{W_NS}}}type", "dxa")


def set_no_wrap(cell: ET.Element) -> None:
    tc_pr = ensure_child(cell, f"{{{W_NS}}}tcPr")
    if tc_pr.find(f"{{{W_NS}}}noWrap") is None:
        ET.SubElement(tc_pr, f"{{{W_NS}}}noWrap")


def set_runs_size(cell: ET.Element, half_points: str) -> None:
    for run in cell.findall(f".//{{{W_NS}}}r"):
        rpr = run.find(f"./{{{W_NS}}}rPr")
        if rpr is None:
            rpr = ET.SubElement(run, f"{{{W_NS}}}rPr")
        for tag in (f"{{{W_NS}}}sz", f"{{{W_NS}}}szCs"):
            sz = rpr.find(tag)
            if sz is None:
                sz = ET.SubElement(rpr, tag)
            sz.set(f"{{{W_NS}}}val", half_points)


def apply_table_dimensions(table: ET.Element, table_idx: int) -> None:
    tbl_pr = ensure_child(table, f"{{{W_NS}}}tblPr")
    tbl_w = tbl_pr.find(f"{{{W_NS}}}tblW")
    if tbl_w is None:
        tbl_w = ET.SubElement(tbl_pr, f"{{{W_NS}}}tblW")

    rows = table.findall(f"./{{{W_NS}}}tr")
    if not rows:
        return

    if table_idx == 2:
        tbl_w.set(f"{{{W_NS}}}w", "6200")
        tbl_w.set(f"{{{W_NS}}}type", "dxa")
        widths = [2600, 2200, 1400]
        for row in rows:
            cells = row.findall(f"./{{{W_NS}}}tc")
            for cell, width in zip(cells, widths):
                set_cell_width(cell, width)
            if cells:
                set_no_wrap(cells[0])
    elif table_idx == 3:
        tbl_w.set(f"{{{W_NS}}}w", "9800")
        tbl_w.set(f"{{{W_NS}}}type", "dxa")
        widths = [3900, 1200, 1500, 1700, 1500]
        for row in rows:
            cells = row.findall(f"./{{{W_NS}}}tc")
            for cell, width in zip(cells, widths):
                set_cell_width(cell, width)


def polish_table_rows(table: ET.Element, table_idx: int) -> None:
    rows = table.findall(f"./{{{W_NS}}}tr")
    if not rows:
        return

    header = rows[0]
    tr_pr = header.find(f"./{{{W_NS}}}trPr")
    if tr_pr is None:
        tr_pr = ET.SubElement(header, f"{{{W_NS}}}trPr")
    tbl_header = tr_pr.find(f"{{{W_NS}}}tblHeader")
    if tbl_header is None:
        tbl_header = ET.SubElement(tr_pr, f"{{{W_NS}}}tblHeader")
    tbl_header.set(f"{{{W_NS}}}val", "true")

    for row_idx, row in enumerate(rows):
        cells = row.findall(f"./{{{W_NS}}}tc")
        for cell in cells:
            tc_pr = ensure_child(cell, f"{{{W_NS}}}tcPr")
            v_align = tc_pr.find(f"{{{W_NS}}}vAlign")
            if v_align is None:
                v_align = ET.SubElement(tc_pr, f"{{{W_NS}}}vAlign")
            v_align.set(f"{{{W_NS}}}val", "center")

            if row_idx == 0:
                set_runs_size(cell, "20")
                set_no_wrap(cell)
                for run in cell.findall(f".//{{{W_NS}}}r"):
                    rpr = run.find(f"./{{{W_NS}}}rPr")
                    if rpr is None:
                        rpr = ET.SubElement(run, f"{{{W_NS}}}rPr")
                    if rpr.find(f"{{{W_NS}}}b") is None:
                        ET.SubElement(rpr, f"{{{W_NS}}}b")
                    if rpr.find(f"{{{W_NS}}}bCs") is None:
                        ET.SubElement(rpr, f"{{{W_NS}}}bCs")


def png_size(png_bytes: bytes) -> tuple[int, int]:
    if png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Unsupported image format; expected PNG")
    width, height = struct.unpack(">II", png_bytes[16:24])
    return width, height


def set_extent(paragraph: ET.Element, width_emu: int, aspect_ratio: float | None = None) -> None:
    extent = paragraph.find(".//wp:extent", NS)
    if extent is None:
        return
    old_cx = int(extent.get("cx"))
    old_cy = int(extent.get("cy"))
    if old_cx <= 0 or old_cy <= 0:
        return
    new_cx = width_emu
    if aspect_ratio is None:
        new_cy = max(1, round(old_cy * new_cx / old_cx))
    else:
        new_cy = max(1, round(new_cx / aspect_ratio))
    extent.set("cx", str(new_cx))
    extent.set("cy", str(new_cy))

    xfrm_ext = paragraph.find(".//a:xfrm/a:ext", NS)
    if xfrm_ext is not None:
        xfrm_ext.set("cx", str(new_cx))
        xfrm_ext.set("cy", str(new_cy))


def patch_document_xml(document_xml: bytes, media_sizes: dict[str, tuple[int, int]]) -> bytes:
    text = document_xml.decode("utf-8")
    text = (
        text.replace("𝒥", "J")
        .replace("𝒞", "C")
        .replace("𝒰", "U")
    )
    root = ET.fromstring(text.encode("utf-8"))

    equation_ids = set(EQUATION_TEXT_BY_RID)
    main_figure_widths = {
        "rId14": 5_212_080,  # ~5.7 in
        "rId15": 4_663_440,  # ~5.1 in
        "rId16": 4_937_760,  # ~5.4 in
        "rId17": 4_937_760,  # ~5.4 in
    }

    for paragraph in root.findall(".//w:p", NS):
        rids = {
            blip.get(f"{{{R_NS}}}embed")
            for blip in paragraph.findall(".//a:blip", NS)
            if blip.get(f"{{{R_NS}}}embed")
        }
        if not rids:
            continue

        if rids <= equation_ids:
            rid = next(iter(rids))
            replace_paragraph_with_text(paragraph, EQUATION_TEXT_BY_RID[rid])
            continue

        if len(rids) == 1:
            rid = next(iter(rids))
            if rid in main_figure_widths:
                if rid in {"rId14", "rId15"}:
                    set_center_alignment(paragraph)
                img_w, img_h = media_sizes.get(rid, (0, 0))
                aspect_ratio = (img_w / img_h) if img_w and img_h else None
                set_extent(paragraph, main_figure_widths[rid], aspect_ratio=aspect_ratio)

    for rfonts in root.findall(".//w:rPr/w:rFonts", NS):
        attrs = list(rfonts.attrib.items())
        if any("monospace" in str(val).lower() for _, val in attrs):
            parent = None
            for rpr in root.findall(".//w:rPr", NS):
                if rfonts in list(rpr):
                    parent = rpr
                    break
            if parent is not None:
                parent.remove(rfonts)

    for table_idx, table in enumerate(root.findall(".//w:tbl", NS), start=1):
        apply_table_borders(table)
        apply_table_dimensions(table, table_idx)
        polish_table_rows(table, table_idx)

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def patch_styles_xml(styles_xml: bytes) -> bytes:
    root = ET.fromstring(styles_xml)
    for style_id in ("InternetLink", "VisitedInternetLink"):
        style = root.find(f".//{{{W_NS}}}style[@{{{W_NS}}}styleId='{style_id}']")
        if style is None:
            continue
        rpr = style.find(f"./{{{W_NS}}}rPr")
        if rpr is None:
            continue
        for child in list(rpr):
            if child.tag in {f"{{{W_NS}}}u", f"{{{W_NS}}}color"}:
                rpr.remove(child)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def patch_docx(docx_path: Path, backup: bool) -> None:
    if backup:
        backup_path = docx_path.with_name(
            f"{docx_path.stem}_before_layout_tune{docx_path.suffix}"
        )
        shutil.copy2(docx_path, backup_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / docx_path.name
        with ZipFile(docx_path) as zin, ZipFile(tmp_path, "w", ZIP_DEFLATED) as zout:
            media_sizes = {}
            for item in zin.infolist():
                if item.filename.startswith("word/media/") and item.filename.endswith(".png"):
                    rid = Path(item.filename).stem
                    media_sizes[rid] = png_size(zin.read(item.filename))
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename == "word/document.xml":
                    data = patch_document_xml(data, media_sizes)
                elif item.filename == "word/styles.xml":
                    data = patch_styles_xml(data)
                zout.writestr(item, data)
        shutil.copy2(tmp_path, docx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    patch_docx(args.docx, backup=not args.no_backup)


if __name__ == "__main__":
    main()
