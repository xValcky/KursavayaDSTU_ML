"""Картинки телефонов.

Если в папке images/ рядом с данными лежит фото модели (<модель>.png/.jpg/.webp) — показываем его.
Иначе рисуем стилизованную иллюстрацию по характеристикам: пропорции экрана, раскладушки,
цвет бренда, блок камер. Работает офлайн и не требует чужих фото.
"""
import re

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap

from src.paths import APP_DIR, BUNDLE_DIR

BRAND_COLORS = {
    "Apple": ("#E9E4DC", "#B9B2A8"), "Samsung": ("#3E4A61", "#1F2633"), "Xiaomi": ("#F2F2F2", "#C9CDD4"),
    "POCO": ("#FFD34E", "#E0A800"), "Google": ("#C9E4DA", "#8DB7A6"), "OnePlus": ("#2B2B2E", "#0E0E10"),
    "Nothing": ("#F5F5F5", "#BDBDBD"), "Realme": ("#F3C969", "#C79A2B"), "Honor": ("#7CC3E8", "#3E86B8"),
    "Vivo": ("#B8C8F0", "#6E86C9"), "Oppo": ("#A8E0C8", "#54A586"), "Motorola": ("#4C5B8C", "#26304F"),
    "ASUS": ("#3A3A3A", "#121212"), "RedMagic": ("#3B1F1F", "#120707"), "Lenovo": ("#5A5A66", "#2A2A33"),
    "nubia": ("#E7E1D3", "#A79F8E"), "Fairphone": ("#9BC59D", "#5C8F60"), "Blackview": ("#5C5C4A", "#2E2E22"),
    "Ulefone": ("#6B6B57", "#34342A"),
}


def _image_path(model: str):
    name = re.sub(r"[^\w\-() +]", "", model).strip()
    for folder in (APP_DIR / "images", BUNDLE_DIR / "images"):
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            path = folder / f"{name}{ext}"
            if path.exists():
                return path
    return None


def _lens_count(p) -> int:
    if p["camera_mp"] >= 200 or p["price_rub"] >= 70000:
        return 3 if p["brand"] != "Apple" or "Pro" in p["model"] else 2
    if p["price_rub"] >= 25000:
        return 3
    return 2


def phone_pixmap(p, width: int = 140, height: int = 200) -> QPixmap:
    """p — строка телефона (pandas Series с brand, model, screen_in, camera_mp, price_rub)."""
    path = _image_path(str(p["model"]))
    if path:
        pix = QPixmap(str(path))
        if not pix.isNull():
            return pix.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    pix = QPixmap(width, height)
    pix.fill(Qt.transparent)
    qp = QPainter(pix)
    qp.setRenderHint(QPainter.Antialiasing)
    light, dark = BRAND_COLORS.get(p["brand"], ("#D8DCE6", "#9AA3B5"))
    model = str(p["model"]).lower()
    book = p["screen_in"] > 7.5
    flip = "flip" in model or "razr" in model

    h = height * (0.86 if not flip else 0.9)
    w = h * (0.46 if not book else 0.82) * (1 + (p["screen_in"] - 6.5) * 0.04 if not book else 1)
    w = min(w, width * 0.92)
    body = QRectF((width - w) / 2, (height - h) / 2, w, h)
    radius = w * (0.16 if not book else 0.08)

    # тень
    qp.setPen(Qt.NoPen)
    qp.setBrush(QColor(0, 0, 0, 28))
    qp.drawRoundedRect(body.translated(3, 5), radius, radius)
    # корпус (вид сзади)
    grad = QLinearGradient(body.topLeft(), body.bottomRight())
    grad.setColorAt(0, QColor(light))
    grad.setColorAt(1, QColor(dark))
    qp.setBrush(grad)
    qp.setPen(QPen(QColor(dark).darker(130), 1.2))
    qp.drawRoundedRect(body, radius, radius)

    if book:  # линия сгиба книжки
        qp.setPen(QPen(QColor(0, 0, 0, 60), 1.5))
        qp.drawLine(QPointF(body.center().x(), body.top() + 6), QPointF(body.center().x(), body.bottom() - 6))
    if flip:  # шарнир раскладушки и внешний экран
        qp.setPen(QPen(QColor(0, 0, 0, 70), 2))
        mid = body.top() + h / 2
        qp.drawLine(QPointF(body.left() + 4, mid), QPointF(body.right() - 4, mid))
        qp.setPen(Qt.NoPen)
        qp.setBrush(QColor("#151820"))
        qp.drawRoundedRect(QRectF(body.left() + w * 0.08, body.top() + h * 0.05, w * 0.84, h * 0.38), 10, 10)

    # блок камер
    lenses = _lens_count(p)
    area = QRectF(body.left() + w * (0.1 if not book else 0.56), body.top() + h * 0.05,
                  w * (0.5 if not book else 0.34), h * 0.24)
    style = {"Apple": "square", "Samsung": "column", "Google": "bar", "Motorola": "column"}.get(p["brand"], "island")
    if flip:
        style = "row"
    qp.setPen(Qt.NoPen)
    lens_r = min(area.width(), area.height()) * (0.24 if lenses == 3 else 0.28)
    centers = []
    if style == "bar":
        bar = QRectF(body.left() + 4, area.top() + area.height() * 0.15, w - 8, area.height() * 0.55)
        qp.setBrush(QColor(20, 22, 28, 220))
        qp.drawRoundedRect(bar, bar.height() / 2, bar.height() / 2)
        centers = [QPointF(bar.left() + bar.width() * (0.25 + 0.14 * k), bar.center().y()) for k in range(lenses)]
        lens_r = bar.height() * 0.32
    elif style == "column":
        centers = [QPointF(area.left() + lens_r * 1.3, area.top() + lens_r * (1.3 + 2.4 * k)) for k in range(lenses)]
    elif style == "row":
        centers = [QPointF(body.left() + w * 0.28 + k * lens_r * 2.6, body.top() + h * 0.12) for k in range(2)]
    else:
        island = QRectF(area.left(), area.top(), area.width(), area.width() * (0.95 if lenses >= 3 else 0.6))
        island.setHeight(min(island.height(), h * 0.3))
        qp.setBrush(QColor(0, 0, 0, 55))
        if style == "island" and p["brand"] in ("Xiaomi", "Honor", "Vivo", "OnePlus", "Oppo", "nubia"):
            r = min(island.width(), island.height())
            island = QRectF(island.left(), island.top(), r, r)
            qp.drawEllipse(island)
        else:
            qp.drawRoundedRect(island, island.width() * 0.22, island.width() * 0.22)
        cx, cy, s = island.center().x(), island.center().y(), island.width() * 0.22
        pts = [(-1, -1), (1, -1), (-1, 1), (1, 1)] if lenses >= 3 else [(0, -1), (0, 1)]
        centers = [QPointF(cx + dx * s, cy + dy * s) for dx, dy in pts[:lenses]]
        lens_r = s * 0.78
    for c in centers:
        qp.setBrush(QColor("#0E1014"))
        qp.drawEllipse(c, lens_r, lens_r)
        qp.setBrush(QColor(90, 120, 170, 140))
        qp.drawEllipse(c, lens_r * 0.45, lens_r * 0.45)
        qp.setBrush(QColor(255, 255, 255, 150))
        qp.drawEllipse(QPointF(c.x() - lens_r * 0.25, c.y() - lens_r * 0.25), lens_r * 0.12, lens_r * 0.12)
    if p["camera_mp"] >= 200:  # перископ
        qp.setBrush(QColor("#0E1014"))
        qp.drawRoundedRect(QRectF(area.right() - lens_r * 1.6, area.top() + 2, lens_r * 1.4, lens_r * 1.4), 3, 3)

    # подпись бренда
    qp.setPen(QColor(255, 255, 255, 170) if QColor(dark).lightness() < 120 else QColor(0, 0, 0, 110))
    f = QFont("Segoe UI", max(7, int(w * 0.09)))
    f.setBold(True)
    qp.setFont(f)
    qp.drawText(QRectF(body.left(), body.bottom() - h * 0.2, w, h * 0.1), Qt.AlignCenter, str(p["brand"]).upper())
    qp.end()
    return pix
