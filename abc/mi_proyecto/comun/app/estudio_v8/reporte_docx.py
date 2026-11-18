"""Dibuja los bloques de `reporte.construir` como Word (.docx editable)."""
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (disponible para anexos apaisados)
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

VERDE = RGBColor(0x2E, 0x6B, 0x1A)
VERDE_CLARO = "E2EFDA"
GRIS = RGBColor(0x52, 0x51, 0x4E)


def _sombra(celda, color):
    tc = celda._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd")
    sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), color)
    tc.append(sh)


def _campo(parrafo, instr):
    """Campo de Word (p. ej. número de página) en un párrafo."""
    r = parrafo.add_run()
    for tipo, texto in (("begin", None), (None, instr), ("separate", None), ("end", None)):
        if tipo:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), tipo); r._r.append(e)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = texto; r._r.append(e)


def _segmentos(par, segs, tam=None, color=None):
    for texto, negrita in segs:
        r = par.add_run(texto)
        r.bold = negrita
        if tam:
            r.font.size = Pt(tam)
        if color is not None:
            r.font.color.rgb = color


def escribir(bloques, ruta, encabezado=""):
    doc = Document()
    sec = doc.sections[0]
    sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin, sec.bottom_margin = Cm(2.0), Cm(1.8)
    st = doc.styles["Normal"]
    st.font.name = "Calibri"; st.font.size = Pt(10.5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    st.paragraph_format.space_after = Pt(5)
    for n, tam in ((1, 15), (2, 12)):
        h = doc.styles[f"Heading {n}"]
        h.font.name = "Calibri"; h.font.size = Pt(tam); h.font.color.rgb = VERDE; h.font.bold = True
    if encabezado:
        p = sec.header.paragraphs[0]
        p.text = encabezado
        p.runs[0].font.size = Pt(8); p.runs[0].font.color.rgb = GRIS
    pie = sec.footer.paragraphs[0]
    pie.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    _campo(pie, "PAGE")

    for b in bloques:
        if b.tipo == "portada":
            m = b.meta
            for _ in range(6):
                doc.add_paragraph()
            p = doc.add_paragraph(); r = p.add_run(m["titulo"]); r.bold = True; r.font.size = Pt(22); r.font.color.rgb = VERDE
            p = doc.add_paragraph(); r = p.add_run(m["subtitulo"]); r.font.size = Pt(15); r.font.color.rgb = GRIS
            doc.add_paragraph()
            t = doc.add_table(rows=0, cols=2)
            for k, v in m["filas"]:
                c = t.add_row().cells
                c[0].text, c[1].text = k, str(v)
                c[0].paragraphs[0].runs[0].bold = True
                for cc in c:
                    cc.paragraphs[0].runs[0].font.size = Pt(10)
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif b.tipo == "titulo":
            doc.add_heading(b.texto, level=b.nivel)
        elif b.tipo == "parrafo":
            p = doc.add_paragraph(); _segmentos(p, b.texto)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif b.tipo == "vinetas":
            for par in b.texto:
                p = doc.add_paragraph(style="List Bullet"); _segmentos(p, par)
        elif b.tipo == "destacado":
            t = doc.add_table(rows=1, cols=1)
            c = t.cell(0, 0); _sombra(c, VERDE_CLARO)
            p = c.paragraphs[0]; _segmentos(p, b.texto, tam=11)
            p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(4)
            doc.add_paragraph()
        elif b.tipo == "tabla":
            df = b.tabla
            t = doc.add_table(rows=1, cols=df.shape[1])
            t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
            for j, col in enumerate(df.columns):
                c = t.rows[0].cells[j]; c.text = str(col); _sombra(c, "4EA72E")
                r = c.paragraphs[0].runs[0]; r.bold = True; r.font.size = Pt(8.5); r.font.color.rgb = RGBColor(255, 255, 255)
            for _, fila in df.iterrows():
                cells = t.add_row().cells
                for j, v in enumerate(fila):
                    cells[j].text = "" if v is None else str(v)
                    par = cells[j].paragraphs[0]
                    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT if (b.alinear[j] if j < len(b.alinear) else "l") == "r" \
                        else WD_ALIGN_PARAGRAPH.LEFT
                    if par.runs:
                        par.runs[0].font.size = Pt(8.5)
            if b.pie:
                p = doc.add_paragraph(); r = p.add_run(b.pie); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = GRIS
        elif b.tipo == "figura":
            if not b.ruta or not b.ruta.exists():
                continue
            doc.add_picture(str(b.ruta), width=Cm(b.ancho_cm))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            p = doc.add_paragraph(); r = p.add_run(b.pie); r.italic = True; r.font.size = Pt(8.5); r.font.color.rgb = GRIS
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif b.tipo == "salto":
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.save(str(ruta))
    return ruta
