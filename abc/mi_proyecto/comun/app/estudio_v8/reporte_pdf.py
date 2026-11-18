"""Dibuja los bloques de `reporte.construir` como PDF (reportlab, sin Word ni navegador).

Usa la DejaVu Sans que trae matplotlib, para tener tildes, ñ, ✔ y "−" también en el
servidor Linux.
"""
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

VERDE, VERDE_MEDIO, VERDE_CLARO = colors.HexColor("#2E6B1A"), colors.HexColor("#4EA72E"), colors.HexColor("#E2EFDA")
GRIS = colors.HexColor("#52514E")


def _fuentes():
    import matplotlib
    d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    if "DejaVu" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DejaVu", str(d / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(d / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("DejaVu-Oblique", str(d / "DejaVuSans-Oblique.ttf")))
        from reportlab.lib.fonts import addMapping
        addMapping("DejaVu", 0, 0, "DejaVu"); addMapping("DejaVu", 1, 0, "DejaVu-Bold")
        addMapping("DejaVu", 0, 1, "DejaVu-Oblique"); addMapping("DejaVu", 1, 1, "DejaVu-Bold")


def _estilos():
    base = dict(fontName="DejaVu", fontSize=9.2, leading=12.6)
    return {
        "normal": ParagraphStyle("n", alignment=TA_JUSTIFY, spaceAfter=5, **base),
        "vineta": ParagraphStyle("v", **base),
        "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=14, leading=18, textColor=VERDE, spaceBefore=8, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=11, leading=14, textColor=VERDE, spaceBefore=6, spaceAfter=4),
        "pie": ParagraphStyle("p", fontName="DejaVu-Oblique", fontSize=7.8, leading=10, textColor=GRIS, alignment=TA_CENTER,
                              spaceAfter=8),
        "celda": ParagraphStyle("c", fontName="DejaVu", fontSize=7.4, leading=9.2),
        "celda_r": ParagraphStyle("cr", fontName="DejaVu", fontSize=7.4, leading=9.2, alignment=TA_RIGHT),
        "cab": ParagraphStyle("h", fontName="DejaVu-Bold", fontSize=7.4, leading=9.2, textColor=colors.white),
        "dest": ParagraphStyle("d", fontName="DejaVu", fontSize=9.8, leading=13.5, alignment=TA_JUSTIFY),
        "port_t": ParagraphStyle("pt", fontName="DejaVu-Bold", fontSize=22, leading=27, textColor=VERDE),
        "port_s": ParagraphStyle("ps", fontName="DejaVu", fontSize=14, leading=18, textColor=GRIS),
    }


def _html(segs):
    return "".join(f"<b>{escape(t)}</b>" if n else escape(t) for t, n in segs)


def escribir(bloques, ruta, encabezado=""):
    _fuentes()
    E = _estilos()
    ancho = A4[0] - 4 * cm
    hist = []
    for b in bloques:
        if b.tipo == "portada":
            m = b.meta
            hist += [Spacer(1, 5 * cm), Paragraph(escape(m["titulo"]), E["port_t"]), Spacer(1, 0.3 * cm),
                     Paragraph(escape(m["subtitulo"]), E["port_s"]), Spacer(1, 1.2 * cm)]
            t = Table([[Paragraph(f"<b>{escape(k)}</b>", E["celda"]), Paragraph(escape(str(v)), E["celda"])]
                       for k, v in m["filas"]], colWidths=[5 * cm, ancho - 5 * cm])
            t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            hist += [t, PageBreak()]
        elif b.tipo == "titulo":
            hist.append(Paragraph(escape(b.texto), E["h1" if b.nivel == 1 else "h2"]))
        elif b.tipo == "parrafo":
            hist.append(Paragraph(_html(b.texto), E["normal"]))
        elif b.tipo == "vinetas":
            hist.append(ListFlowable([ListItem(Paragraph(_html(p), E["vineta"]), leftIndent=12) for p in b.texto],
                                     bulletType="bullet", start="•", leftIndent=12, bulletFontName="DejaVu",
                                     bulletFontSize=8, spaceAfter=6))
            hist.append(Spacer(1, 4))
        elif b.tipo == "destacado":
            t = Table([[Paragraph(_html(b.texto), E["dest"])]], colWidths=[ancho])
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), VERDE_CLARO),
                                   ("LINEBEFORE", (0, 0), (0, -1), 3, VERDE_MEDIO),
                                   ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9)]))
            hist += [t, Spacer(1, 8)]
        elif b.tipo == "tabla":
            df = b.tabla
            al = b.alinear + ["l"] * (df.shape[1] - len(b.alinear))
            datos = [[Paragraph(escape(str(c)), E["cab"]) for c in df.columns]]
            for _, fila in df.iterrows():
                datos.append([Paragraph(escape("" if v is None else str(v)), E["celda_r" if al[j] == "r" else "celda"])
                              for j, v in enumerate(fila)])
            # Ancho por columna: la palabra más larga del encabezado no se corta y el contenido
            # más largo pesa hasta 40 caracteres; mínimo 5.
            largos = []
            for j in range(df.shape[1]):
                palabra = max(len(w) for w in str(df.columns[j]).split() or [""])
                celda = max([len(str(v)) for v in df.iloc[:, j]] or [0])
                largos.append(min(max(palabra * 1.15 + 1, celda, 5), 40))
            cw = [ancho * l / sum(largos) for l in largos]
            t = Table(datos, colWidths=cw, repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), VERDE_MEDIO),
                                   ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F8EF")]),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d3c0")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
            hist.append(t)
            if b.pie:
                hist.append(Paragraph(escape(b.pie), E["pie"]))
        elif b.tipo == "figura":
            if not b.ruta or not b.ruta.exists():
                continue
            w, h = PILImage.open(b.ruta).size
            ww = min(b.ancho_cm * cm, ancho)
            hh = ww * h / w
            if hh > 21 * cm:
                hh = 21 * cm; ww = hh * w / h
            hist.append(KeepTogether([Image(str(b.ruta), width=ww, height=hh), Paragraph(escape(b.pie), E["pie"])]))
        elif b.tipo == "salto":
            hist.append(PageBreak())

    def pagina(canvas, doc_):
        if doc_.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DejaVu", 7.5); canvas.setFillColor(GRIS)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, encabezado)
        canvas.drawRightString(A4[0] - 2 * cm, 1.0 * cm, str(doc_.page))
        canvas.setStrokeColor(VERDE_MEDIO); canvas.setLineWidth(0.6)
        canvas.line(2 * cm, A4[1] - 1.35 * cm, A4[0] - 2 * cm, A4[1] - 1.35 * cm)
        canvas.restoreState()

    doc = SimpleDocTemplate(str(ruta), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm,
                            bottomMargin=1.7 * cm, title="Reporte de resiliencia", author="Herramienta de resiliencia v8")
    doc.build(hist, onFirstPage=pagina, onLaterPages=pagina)
    return ruta
