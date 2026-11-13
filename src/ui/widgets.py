"""Общие виджеты интерфейса: таблицы, карточки телефонов, «Похожие», «Дуэль», «Угадай класс», «Песочница»."""
import traceback

import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QCompleter, QDialog, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QPushButton, QScrollArea, QSlider, QSpinBox,
                               QSplitter, QTableView, QVBoxLayout, QWidget)

from src.ml.labeling import CLASS_NAMES
from src.ml.recommender import SIMILAR_BY, full_name, phone_row, similar_phones
from src.ui import charts
from src.ui.phone_art import phone_pixmap

FLAG_COLUMNS = ("5G", "NFC", "eSIM", "ИК")


# ---------- базовые помощники ----------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None):
        super().__init__()
        self.df = df if df is not None else pd.DataFrame()

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self.df = df.reset_index(drop=True)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.df)

    def columnCount(self, parent=QModelIndex()):
        return len(self.df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            v = self.df.iat[index.row(), index.column()]
            col = self.df.columns[index.column()]
            if col in FLAG_COLUMNS:
                return "✓" if v else "—"
            if isinstance(v, (float, np.floating)):
                if float(v).is_integer():
                    return f"{v:,.0f}".replace(",", " ")
                if abs(v) < 10:
                    return f"{v:.3f}".rstrip("0").rstrip(".")
                return f"{v:,.1f}".replace(",", " ")
            if isinstance(v, (int, np.integer)) and abs(v) >= 1000 and col != "Год":
                return f"{v:,}".replace(",", " ")
            return str(v)
        if role == Qt.TextAlignmentRole:
            numeric = pd.api.types.is_numeric_dtype(self.df.dtypes.iloc[index.column()])
            return int((Qt.AlignRight if numeric else Qt.AlignLeft) | Qt.AlignVCenter)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return str(self.df.columns[section])

    def sort(self, column, order=Qt.AscendingOrder):
        if self.df.empty:
            return
        self.set_df(self.df.sort_values(self.df.columns[column], ascending=order == Qt.AscendingOrder))


def make_table(model: PandasModel) -> QTableView:
    view = QTableView()
    view.setModel(model)
    view.setSortingEnabled(True)
    view.setSelectionBehavior(QTableView.SelectRows)
    view.setSelectionMode(QTableView.SingleSelection)
    view.setAlternatingRowColors(True)
    view.setShowGrid(False)
    view.verticalHeader().setVisible(False)
    view.verticalHeader().setDefaultSectionSize(30)
    view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    view.horizontalHeader().setStretchLastSection(True)
    view.horizontalHeader().setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    return view


def selected_id(view: QTableView, model: PandasModel) -> int | None:
    rows = view.selectionModel().selectedRows()
    if not rows or "id" not in model.df.columns:
        return None
    return int(model.df.iloc[rows[0].row()]["id"])


def page() -> tuple[QWidget, QVBoxLayout]:
    w = QWidget()
    w.setObjectName("page")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(16, 12, 16, 16)
    lay.setSpacing(12)
    return w, lay


def hint(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("hint")
    lbl.setWordWrap(True)
    return lbl


def split(orientation, *widgets, sizes=None) -> QSplitter:
    s = QSplitter(orientation)
    for w in widgets:
        s.addWidget(w)
    if sizes:
        s.setSizes(sizes)
    return s


def money(v) -> str:
    return f"{v:,.0f} ₽".replace(",", " ")


class _Signals(QObject):
    done = Signal(object)
    error = Signal(str)
    progress = Signal(str)


class Task(QRunnable):
    """Фоновая задача, чтобы обучение не замораживало окно."""

    def __init__(self, fn, *args, with_progress=False):
        super().__init__()
        self.fn, self.args, self.with_progress = fn, args, with_progress
        self.signals = _Signals()

    def run(self):
        try:
            kwargs = {"progress": self.signals.progress.emit} if self.with_progress else {}
            self.signals.done.emit(self.fn(*self.args, **kwargs))
        except Exception as e:
            self.signals.error.emit(f"{e}\n\n{traceback.format_exc()}")


# ---------- карточка телефона ----------
class PhoneCard(QFrame):
    """Карточка: картинка, название, цена, класс/кластер, плюсы, кнопки."""

    def __init__(self, p: pd.Series, pros: list[str] | None = None, badge: str = "",
                 on_similar=None, on_like=None, compact: bool = False, note: str = "", on_favorite=None,
                 is_favorite: bool = False, on_details=None):
        super().__init__()
        self.setObjectName("card")
        self.on_details, self.phone_id = on_details, int(p["id"])
        if on_details:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip("Нажмите на карточку — откроются все характеристики")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(14)
        img = QLabel()
        img.setPixmap(phone_pixmap(p, 90 if compact else 120, 130 if compact else 175))
        img.setAlignment(Qt.AlignCenter)
        lay.addWidget(img)

        col = QVBoxLayout()
        col.setSpacing(4)
        if badge:
            b = QLabel(badge)
            b.setObjectName("badge")
            col.addWidget(b, alignment=Qt.AlignLeft)
        name = QLabel(p.get("full_name", f"{p['brand']} {p['model']}"))
        name.setObjectName("cardTitle")
        name.setWordWrap(True)
        col.addWidget(name)
        price = QLabel(money(p["price_rub"]))
        price.setObjectName("cardPrice")
        col.addWidget(price)
        badges = list(p.get("badges", []) or [])
        if badges:
            row = QHBoxLayout()
            row.setSpacing(6)
            for text in badges:
                chip = QLabel(text)
                good = text.startswith("выгодно")
                bad = text.startswith("переплата")
                chip.setObjectName("chipGood" if good else "chipBad" if bad else "chipInfo")
                row.addWidget(chip)
            row.addStretch()
            col.addLayout(row)
        group = str(p.get("cluster_name", "")).split(" · ")[-1]  # шпиль уже виден по классу
        meta = hint(f"{p.get('class_name', '')} · {group}" + (f" · {note}" if note else ""))
        col.addWidget(meta)
        if not compact:
            specs = hint(f"{p['ram_gb']:.0f}/{p['rom_gb']:.0f} ГБ · экран {p['screen_in']:g}″ {p['refresh_hz']:.0f} Гц · "
                         f"≈{p['active_h']:.0f} ч активно / {p['standby_d']:.0f} дн. ожидания")
            col.addWidget(specs)
        for pro in (pros or [])[:3]:
            col.addWidget(QLabel(f"✓ {pro}"))
        col.addStretch()
        btns = QHBoxLayout()
        if on_details:
            b = QPushButton("Все характеристики")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda: on_details(int(p["id"])))
            btns.addWidget(b)
        if on_similar:
            b = QPushButton("Похожие")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda: on_similar(int(p["id"])))
            btns.addWidget(b)
        if on_like:
            b = QPushButton("Подходит")
            b.setObjectName("like")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda: on_like(int(p["id"])))
            btns.addWidget(b)
        if on_favorite:
            b = QPushButton("★ В избранном" if is_favorite else "☆ В избранное")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda: (on_favorite(str(p["model"])),
                                       b.setText("★ В избранном" if b.text().startswith("☆") else "☆ В избранное")))
            btns.addWidget(b)
        btns.addStretch()
        col.addLayout(btns)
        lay.addLayout(col, 1)


    def mouseReleaseEvent(self, event):
        # клик по самой карточке (не по кнопке) — открыть все характеристики
        if self.on_details and event.button() == Qt.LeftButton:
            self.on_details(self.phone_id)
        super().mouseReleaseEvent(event)


def yes_no(v) -> str:
    return "есть" if int(v) else "нет"


def spec_rows(p: pd.Series) -> list[tuple[str, str, str]]:
    """Все характеристики телефона по разделам: (раздел, параметр, значение)."""
    from src.ml.features import effective_refresh
    rub = lambda v: f"{v:,.0f} ₽".replace(",", " ")
    fair = p.get("fair_price")
    rows = [
        ("Основное", "Бренд", p["brand"]), ("Основное", "Модель", p["model"]), ("Основное", "Система", p["os"]),
        ("Основное", "Год выпуска", f"{p['year']:.0f}"), ("Основное", "Класс", p.get("class_name", "")),
        ("Основное", "Группа (кластер)", p.get("cluster_name", "")),
        ("Цена", "Цена", rub(p["price_rub"])), ("Цена", "Источник цены", str(p.get("price_source", ""))),
        ("Цена", "Справедливая цена (модель)", rub(fair) if fair else "считается…"),
        ("Цена", "К справедливой", f"{p['price_rub'] / fair - 1:+.0%}" if fair else "—"),
        ("Производительность", "AnTuTu", f"{p['antutu_k']:,.0f} тыс. баллов".replace(",", " ")),
        ("Производительность", "Оперативная память", f"{p['ram_gb']:.0f} ГБ"),
        ("Производительность", "Встроенная память", f"{p['rom_gb']:.0f} ГБ"),
        ("Экран", "Диагональ", f"{p['screen_in']:g}″"),
        ("Экран", "Частота обновления", f"{p['refresh_hz']:.0f} Гц"),
        ("Экран", "Эффективная частота (выше 120 Гц × 0,2)", f"{float(effective_refresh(p['refresh_hz'])):.1f} Гц"),
        ("Камера", "Основная камера", f"{p['camera_mp']:.0f} МП"),
        ("Батарея", "Ёмкость", f"{p['battery_mah']:,.0f} мА·ч".replace(",", " ")),
        ("Батарея", "Мощность зарядки", f"{p['charge_w']:.0f} Вт"),
        ("Батарея", "Активная работа (оценка)", f"≈{p['active_h']:.1f} ч"),
        ("Батарея", "В режиме ожидания (оценка)", f"≈{p['standby_d']:.1f} дн."),
        ("Корпус", "Вес", f"{p['weight_g']:.0f} г"),
        ("Корпус", "Защищённый «броневик»", "да" if p["weight_g"] >= 300 else "нет"),
        ("Связь и функции", "5G", yes_no(p["has_5g"])), ("Связь и функции", "NFC", yes_no(p["nfc"])),
        ("Связь и функции", "eSIM", yes_no(p["esim"])), ("Связь и функции", "ИК-порт (пульт)", yes_no(p["ir_port"])),
    ]
    return [(sec, name, str(val)) for sec, name, val in rows]


class SpecsDialog(QDialog):
    """Все характеристики телефона: карточка, полная таблица по разделам, история цены и совет."""

    def __init__(self, p: pd.Series, parent=None, history=None, advice_text: str | None = None,
                 on_similar=None, on_favorite=None, is_favorite: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Характеристики: {p.get('full_name', p['model'])}")
        self.resize(1000, 760)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        top = QHBoxLayout()
        top.addWidget(PhoneCard(p, on_favorite=on_favorite, is_favorite=is_favorite), 3)
        side = QVBoxLayout()
        canvas = charts.MplCanvas(4.5, 2.4)
        charts.plot_price_history(canvas, history if history is not None else pd.DataFrame(),
                                  "История цены", p.get("fair_price"))
        side.addWidget(canvas, 1)
        if advice_text:
            adv = QLabel(advice_text)
            adv.setObjectName("cardTitle")
            adv.setWordWrap(True)
            side.addWidget(adv)
        top.addLayout(side, 2)
        lay.addLayout(top)
        model = PandasModel(pd.DataFrame(spec_rows(p), columns=["Раздел", "Параметр", "Значение"]))
        table = make_table(model)
        table.setSortingEnabled(False)
        lay.addWidget(table, 1)
        btns = QHBoxLayout()
        btns.addStretch()
        if on_similar:
            b = QPushButton("Похожие телефоны")
            b.clicked.connect(lambda: (self.accept(), on_similar(int(p["id"]))))
            btns.addWidget(b)
        close = QPushButton("Закрыть")
        close.setObjectName("primary")
        close.clicked.connect(self.accept)
        btns.addWidget(close)
        lay.addLayout(btns)


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())


# ---------- «Похожие» ----------
class SimilarDialog(QDialog):
    """Окно телефона с подборкой похожих по выбранному параметру."""

    def __init__(self, pipe, phone_id: int, parent=None, by: str = "all", fair: dict | None = None,
                 history_fn=None, advice_fn=None, on_favorite=None, favorites=None):
        super().__init__(parent)
        self.pipe, self.fair = pipe, fair
        self.history_fn, self.advice_fn = history_fn, advice_fn
        self.on_favorite, self.favorites = on_favorite, favorites if favorites is not None else set()
        self.setWindowTitle("Похожие телефоны")
        self.resize(1040, 820)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        top = QHBoxLayout()
        self.head = QVBoxLayout()
        top.addLayout(self.head, 3)
        side = QVBoxLayout()
        self.history_canvas = charts.MplCanvas(4.5, 2.4)
        self.advice = QLabel()
        self.advice.setObjectName("cardTitle")
        self.advice.setWordWrap(True)
        side.addWidget(self.history_canvas, 1)
        side.addWidget(self.advice)
        top.addLayout(side, 2)
        lay.addLayout(top)
        row = QHBoxLayout()
        row.addWidget(QLabel("Похожие по:"))
        self.by = QComboBox()
        for key, (title, _) in SIMILAR_BY.items():
            if key == "undervalued" and not fair:
                continue
            self.by.addItem(title, key)
        self.by.setCurrentIndex(max(0, self.by.findData(by)))
        self.by.currentIndexChanged.connect(lambda _: self.show_phone(self.phone_id))
        row.addWidget(self.by)
        row.addStretch()
        lay.addLayout(row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setObjectName("page")
        self.grid = QGridLayout(holder)
        self.grid.setSpacing(12)
        scroll.setWidget(holder)
        lay.addWidget(scroll, 1)
        self.show_phone(phone_id)

    def show_phone(self, phone_id: int):
        self.phone_id = phone_id
        clear_layout(self.head)
        p0 = phone_row(self.pipe, phone_id, self.fair)
        self.head.addWidget(PhoneCard(p0, badge="Выбранный телефон", on_favorite=self.on_favorite,
                                      is_favorite=str(p0["model"]) in self.favorites))
        if self.history_fn:
            hist = self.history_fn(phone_id)
            charts.plot_price_history(self.history_canvas, hist, f"История цены: {p0['full_name']}",
                                      p0.get("fair_price"))
            text = self.advice_fn(phone_id, hist) if self.advice_fn else None
            self.advice.setText(text or "")
        clear_layout(self.grid)
        sim = similar_phones(self.pipe, phone_id, self.by.currentData(), n=6, fair=self.fair)
        for k, r in sim.iterrows():
            p = phone_row(self.pipe, int(r["id"]), self.fair)
            note = f"сходство {r['Сходство']}, {r['Разница в цене']}"
            if r.get("К справедливой"):
                note += f", к справедливой {r['К справедливой']}"
            card = PhoneCard(p, compact=True, on_similar=self.show_phone, note=note, on_favorite=self.on_favorite,
                             is_favorite=str(p["model"]) in self.favorites)
            self.grid.addWidget(card, k // 2, k % 2)


# ---------- «Дуэль» ----------
DUEL_ROWS = [("price_rub", "Цена, ₽", -1), ("antutu_k", "Производительность (AnTuTu, тыс.)", 1),
             ("camera_mp", "Камера, МП", 1), ("active_h", "Активная работа, ч", 1),
             ("standby_d", "Ожидание, дней", 1), ("battery_mah", "Батарея, мА·ч", 1),
             ("charge_w", "Зарядка, Вт", 1), ("ram_gb", "ОЗУ, ГБ", 1), ("rom_gb", "Память, ГБ", 1),
             ("refresh_hz", "Частота экрана, Гц", 1), ("screen_in", "Экран, дюйм", 0),
             ("weight_g", "Вес, г", -1), ("year", "Год", 1)]


def phone_picker(pipe) -> QComboBox:
    box = QComboBox()
    box.setEditable(True)
    box.setInsertPolicy(QComboBox.NoInsert)
    for _, p in pipe.phones_.sort_values(["brand", "model"]).iterrows():
        box.addItem(full_name(p["brand"], p["model"]), int(p["id"]))
    box.completer().setFilterMode(Qt.MatchContains)
    box.completer().setCompletionMode(QCompleter.PopupCompletion)
    return box


class DuelWidget(QWidget):
    """Сравнение двух телефонов: картинки, радар, таблица «кто сильнее»."""

    def __init__(self):
        super().__init__()
        self.pipe = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(hint("Выберите два телефона — покажу, кто сильнее по каждому параметру."))
        self.top = QHBoxLayout()
        lay.addLayout(self.top)
        self.cards = QHBoxLayout()
        lay.addLayout(self.cards)
        self.canvas = charts.MplCanvas(6, 4)
        self.table = PandasModel()
        lay.addWidget(split(Qt.Horizontal, self.canvas, make_table(self.table), sizes=[500, 500]), 1)
        self.score = QLabel()
        self.score.setObjectName("cardTitle")
        lay.addWidget(self.score)

    def set_pipe(self, pipe, a_id: int | None = None, b_id: int | None = None):
        self.pipe = pipe
        clear_layout(self.top)
        self.a, self.b = phone_picker(pipe), phone_picker(pipe)
        self.top.addWidget(self.a, 1)
        self.top.addWidget(QLabel("против"))
        self.top.addWidget(self.b, 1)
        for box, pid, default in ((self.a, a_id, 0), (self.b, b_id, 1)):
            idx = box.findData(pid) if pid is not None else -1
            box.setCurrentIndex(idx if idx >= 0 else default)
            box.currentIndexChanged.connect(lambda _: self.refresh())
        self.refresh()

    def refresh(self):
        if self.pipe is None or self.a.currentData() is None or self.b.currentData() is None:
            return
        pa, pb = phone_row(self.pipe, self.a.currentData()), phone_row(self.pipe, self.b.currentData())
        clear_layout(self.cards)
        self.cards.addWidget(PhoneCard(pa, compact=True))
        self.cards.addWidget(PhoneCard(pb, compact=True))
        charts.plot_reco_radar(self.canvas, self.pipe, [int(pa["id"]), int(pb["id"])], top=2,
                               title="Дуэль: характеристики (перцентиль в базе)")
        rows, wins = [], [0, 0]
        for col, title, sign in DUEL_ROWS:
            va, vb = pa[col], pb[col]
            win = "—"
            if sign and va != vb:
                a_better = (va > vb) == (sign > 0)
                wins[0 if a_better else 1] += 1
                win = pa["full_name"] if a_better else pb["full_name"]
            fmt = (lambda v: f"{v:,.0f}".replace(",", " ")) if col in ("price_rub", "battery_mah") else (lambda v: f"{v:g}")
            rows.append({"Параметр": title, "Первый": fmt(va), "Второй": fmt(vb), "Сильнее": win})
        self.table.set_df(pd.DataFrame(rows))
        leader = pa["full_name"] if wins[0] > wins[1] else pb["full_name"] if wins[1] > wins[0] else "ничья"
        self.score.setText(f"Счёт {wins[0]} : {wins[1]} — {leader}")


# ---------- «Угадай класс» ----------
class GuessGame(QWidget):
    """Мини-игра: по характеристикам угадать класс телефона и сравнить себя с нейросетью."""

    def __init__(self):
        super().__init__()
        self.pipe, self.nn_pred, self.current = None, None, None
        self.you = self.nn = self.rounds = 0
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(hint("Цена скрыта. Угадайте класс телефона по характеристикам. Соперник — нейросеть, "
                           "обученная на остальных телефонах (кросс-валидация, свой ответ для этого телефона "
                           "она не видела)."))
        self.card_box = QHBoxLayout()
        lay.addLayout(self.card_box)
        self.specs = QLabel()
        self.specs.setWordWrap(True)
        lay.addWidget(self.specs)
        btns = QHBoxLayout()
        for c, name in CLASS_NAMES.items():
            b = QPushButton(f"{c} — {name}")
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, cls=c: self.answer(cls))
            btns.addWidget(b)
        nxt = QPushButton("Следующий телефон")
        nxt.setObjectName("primary")
        nxt.clicked.connect(self.next_phone)
        btns.addStretch()
        btns.addWidget(nxt)
        lay.addLayout(btns)
        self.result = QLabel()
        self.result.setObjectName("cardTitle")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        lay.addStretch()

    def set_pipe(self, pipe):
        from sklearn.model_selection import cross_val_predict
        from sklearn.neural_network import MLPClassifier
        self.pipe = pipe
        mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=0)
        self.nn_pred = cross_val_predict(mlp, pipe.z_, pipe.labels_, cv=5)
        self.you = self.nn = self.rounds = 0
        self.next_phone()

    def next_phone(self):
        if self.pipe is None:
            return
        self.current = int(np.random.default_rng().integers(len(self.pipe.phones_)))
        p = self.pipe.phones_.iloc[self.current]
        clear_layout(self.card_box)
        img = QLabel()
        img.setPixmap(phone_pixmap(p, 120, 175))
        self.card_box.addWidget(img)
        self.card_box.addWidget(QLabel(f"<b style='font-size:14pt'>{full_name(p['brand'], p['model'])}</b>"), 1)
        self.specs.setText(
            f"{p['year']:.0f} год · {p['ram_gb']:.0f}/{p['rom_gb']:.0f} ГБ · AnTuTu {p['antutu_k']:.0f} тыс. · "
            f"камера {p['camera_mp']:.0f} МП · экран {p['screen_in']:g}″ {p['refresh_hz']:.0f} Гц · "
            f"батарея {p['battery_mah']:.0f} мА·ч · зарядка {p['charge_w']:.0f} Вт")
        self.result.setText(f"Счёт: вы {self.you} · нейросеть {self.nn} · раундов {self.rounds}")
        self.answered = False

    def answer(self, cls: int):
        if self.pipe is None or self.answered:
            return
        self.answered = True
        truth = int(self.pipe.labels_[self.current])
        nn = int(self.nn_pred[self.current])
        self.rounds += 1
        self.you += cls == truth
        self.nn += nn == truth
        p = self.pipe.phones_.iloc[self.current]
        self.result.setText(
            f"Правильно: {CLASS_NAMES[truth]} ({money(p['price_rub'])}). "
            f"Вы: {'✓' if cls == truth else '✗'} {CLASS_NAMES[cls]}, нейросеть: {'✓' if nn == truth else '✗'} "
            f"{CLASS_NAMES[nn]}.   Счёт: вы {self.you} · нейросеть {self.nn} · раундов {self.rounds}")


# ---------- «Песочница модели» ----------
class SandboxWidget(QWidget):
    """Эксперименты с гиперпараметрами: число кластеров, слои нейросети, число анкет."""
    train_requested = Signal(dict)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(hint("Поменяйте настройки и обучите модель заново — карта кластеров, метрики и рекомендации "
                           "сразу пересчитаются. «Сохранить» делает эту модель основной."))
        box = QGroupBox("Настройки")
        g = QGridLayout(box)
        self.auto_k = QCheckBox("подобрать автоматически по силуэту (в каждом шпиле)")
        self.auto_k.setChecked(True)
        self.k = QSpinBox()
        self.k.setRange(2, 8)
        self.k.setValue(5)
        self.auto_k.toggled.connect(lambda on: self.k.setEnabled(not on))
        self.k.setEnabled(False)
        self.hidden = QComboBox()
        for h in [(32,), (64, 32), (128, 64), (128, 64, 32)]:
            self.hidden.addItem(" × ".join(map(str, h)), h)
        self.hidden.setCurrentIndex(1)
        self.alpha = QComboBox()
        for a in [1e-4, 1e-3, 1e-2, 1e-1]:
            self.alpha.addItem(f"{a:g}", a)
        self.alpha.setCurrentIndex(1)
        self.surveys = QSlider(Qt.Horizontal)
        self.surveys.setRange(1, 10)
        self.surveys.setValue(6)
        self.surveys_lbl = QLabel("6000")
        self.surveys_lbl.setObjectName("value")
        self.surveys.valueChanged.connect(lambda v: self.surveys_lbl.setText(f"{v * 1000}"))
        g.addWidget(QLabel("Групп k в шпиле"), 0, 0)
        g.addWidget(self.k, 0, 1)
        g.addWidget(self.auto_k, 0, 2)
        g.addWidget(QLabel("Скрытые слои MLP"), 1, 0)
        g.addWidget(self.hidden, 1, 1)
        g.addWidget(QLabel("Регуляризация alpha"), 2, 0)
        g.addWidget(self.alpha, 2, 1)
        g.addWidget(QLabel("Обучающих анкет"), 3, 0)
        g.addWidget(self.surveys, 3, 1)
        g.addWidget(self.surveys_lbl, 3, 2)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.btn = QPushButton("Обучить с этими настройками")
        self.btn.setObjectName("primary")
        self.btn.clicked.connect(self._train)
        self.btn_save = QPushButton("Сохранить как основную модель")
        row.addWidget(self.btn)
        row.addWidget(self.btn_save)
        row.addStretch()
        lay.addLayout(row)
        self.history = PandasModel()
        self._rows = []
        lay.addWidget(QLabel("История экспериментов"))
        lay.addWidget(make_table(self.history), 1)

    def _train(self):
        self.btn.setEnabled(False)
        self.btn.setText("Обучаю…")
        self.train_requested.emit({"k": None if self.auto_k.isChecked() else self.k.value(),
                                   "n_surveys": self.surveys.value() * 1000,
                                   "mlp_params": {"hidden": self.hidden.currentData(),
                                                  "alpha": self.alpha.currentData()}})

    def add_result(self, params: dict, pipe):
        self.btn.setEnabled(True)
        self.btn.setText("Обучить с этими настройками")
        t = pipe.metrics_["test"]
        self._rows.append({"k": pipe.k_, "слои": " × ".join(map(str, params["mlp_params"]["hidden"])),
                           "alpha": params["mlp_params"]["alpha"], "анкет": params["n_surveys"],
                           "accuracy": round(t["accuracy"], 3), "F1": round(t["f1_macro"], 3),
                           "ROC AUC": round(t["roc_auc_ovr"], 3)})
        self.history.set_df(pd.DataFrame(self._rows[::-1]))

    def reset_button(self):
        self.btn.setEnabled(True)
        self.btn.setText("Обучить с этими настройками")


def pool() -> QThreadPool:
    return QThreadPool.globalInstance()
