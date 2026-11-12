"""Главное окно: Опрос → Результат, Каталог, Избранное; в про-режиме — Аналитика ML, Дуэль/игра/песочница, Данные и API."""
import json

import numpy as np
import pandas as pd
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from src import db
from src.ml import clustering, experiments
from src.ml.classifier import REGISTRY_PATH, PhonePipeline
from src.ml.clustering import TIER_NAMES
from src.ml.labeling import (CLASS_NAMES, HZ_PREF, IMPORTANCE_DEFAULTS, IMPORTANCE_FIELDS, IMPORTANCE_HINTS,
                             IMPORTANCE_TITLES, OPTION_PREF, PRESET_HINTS, PRESETS, SCREEN_PREF, SIMPLE_IMPORTANCE)
from src.ml.price_forecast import PriceForecast, advice
from src.ml.price_model import PriceModel
from src.ml.recommender import explain, full_name, phone_row, recommend
from src.providers.base import ENV_PATH, ProviderError, all_providers
from src.ui import charts
from src.ui.style import build_palette, build_qss
from src.ui.widgets import (DUEL_ROWS, DuelWidget, GuessGame, PandasModel, PhoneCard, SandboxWidget, SimilarDialog,
                            Task, clear_layout, hint, make_table, page, pool, selected_id, split)

CATALOG_COLUMNS = {
    "id": "id", "brand": "Бренд", "model": "Модель", "price_rub": "Цена, ₽", "os": "ОС", "year": "Год",
    "ram_gb": "ОЗУ, ГБ", "rom_gb": "Память, ГБ", "battery_mah": "Батарея, мА·ч", "active_h": "Активно, ч",
    "standby_d": "Ожидание, дн", "screen_in": "Экран, ″", "refresh_hz": "Гц", "weight_g": "Вес, г",
    "camera_mp": "Камера, МП", "antutu_k": "AnTuTu, тыс.", "charge_w": "Зарядка, Вт", "has_5g": "5G",
    "nfc": "NFC", "esim": "eSIM", "ir_port": "ИК", "price_source": "Источник цены",
}
API_KEYS = [("YANDEX_MARKET_API_KEY", "Яндекс Маркет: Api-Key"), ("YANDEX_MARKET_BUSINESS_ID", "Яндекс Маркет: ID кабинета"),
            ("OZON_CLIENT_ID", "Ozon: Client-Id"), ("OZON_API_KEY", "Ozon: Api-Key"),
            ("WB_API_TOKEN", "Wildberries: токен «Цены и скидки»"),
            ("AVITO_CLIENT_ID", "Avito: client_id"), ("AVITO_CLIENT_SECRET", "Avito: client_secret")]
TAB_SURVEY, TAB_RESULT, TAB_CATALOG, TAB_FAV, TAB_ANALYTICS, TAB_FUN, TAB_DATA = range(7)
PRO_TABS = (TAB_ANALYTICS, TAB_FUN, TAB_DATA)
SORTS = {"по оценке": ("Оценка", False), "сначала дешевле": ("Цена, ₽", True),
         "дольше работают": ("Активно, ч", False), "лучше камера": ("Камера, МП", False),
         "выгоднее (к справедливой цене)": ("К справедливой, %", True)}
CARD_STEP = 5


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Подбор смартфона")
        self.resize(1400, 900)
        self.settings = QSettings("PhonePicker", "PhonePicker")
        self.pipe: PhonePipeline | None = None
        self.price_model: PriceModel | None = None
        self.forecast: PriceForecast | None = None
        self.fair: dict | None = None
        self.exp: dict | None = experiments.load_results()
        self.survey_id: int | None = None
        self.last_answers: dict | None = None
        self.last_res: dict | None = None
        self.cards_shown = 0
        self.favorites: set[str] = set(json.loads(self.settings.value("favorites", "[]")))

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_survey_tab(), "Опрос")
        self.tabs.addTab(self._build_result_tab(), "Результат")
        self.tabs.addTab(self._build_catalog_tab(), "Каталог")
        self.tabs.addTab(self._build_favorites_tab(), "Избранное")
        self.tabs.addTab(self._build_analytics_tab(), "Аналитика ML")
        self.tabs.addTab(self._build_fun_tab(), "Дуэль, игра, песочница")
        self.tabs.addTab(self._build_data_tab(), "Данные и API")
        self.tabs.currentChanged.connect(lambda i: self.refresh_favorites() if i == TAB_FAV else None)

        # переключатель режима: сегмент «Простой | Про» + кнопка темы
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 2, 12, 2)
        cl.setSpacing(8)
        segment = QFrame()
        segment.setObjectName("segment")
        sg = QHBoxLayout(segment)
        sg.setContentsMargins(3, 3, 3, 3)
        sg.setSpacing(2)
        self.mode_group = QButtonGroup(self)
        self.btn_simple = QPushButton("Простой")
        self.btn_pro = QPushButton("Про")
        for b, tip in ((self.btn_simple, "Для новичка: пресеты, бюджет, 4 ползунка, карточки"),
                       (self.btn_pro, "Для опытных: все параметры, аналитика модели, дуэль, игра и песочница")):
            b.setObjectName("segBtn")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            self.mode_group.addButton(b)
            sg.addWidget(b)
        self.btn_pro.toggled.connect(self.set_mode)
        self.btn_theme = QPushButton()
        self.btn_theme.setObjectName("themeBtn")
        self.btn_theme.setCursor(Qt.PointingHandCursor)
        self.btn_theme.clicked.connect(self.toggle_theme)
        cl.addWidget(segment)
        cl.addWidget(self.btn_theme)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        self.theme = self.settings.value("theme", "light")
        self.apply_theme(self.theme, rerender=False)

        db.init_db()
        self.reload_phones()
        # системы, которых нет в базе (например, iOS), в выпадающих списках недоступны
        present = set(self.phones_all["os"])
        for combo in (self.os_pref, self.f_os):
            for i in range(1, combo.count()):
                combo.model().item(i).setEnabled(combo.itemText(i) in present)
        self.refresh_catalog()
        self.refresh_db_stats()
        self.reload_profiles()
        pro = self.settings.value("pro_mode", False, type=bool)
        (self.btn_pro if pro else self.btn_simple).setChecked(True)
        self.set_mode(pro)
        self.pipe = PhonePipeline.load()
        if self.pipe is None or len(self.pipe.phones_) != len(self.phones):
            self.retrain()
        else:
            self.on_model_ready(self.pipe, save=False)

    # ===== Данные в памяти (кэш вместо чтения БД на каждый фильтр) =====
    def reload_phones(self):
        self.phones_all = db.get_phones(include_archived=True)
        self.phones = self.phones_all[self.phones_all["archived"].fillna(0) == 0]

    # ===== Тема =====
    def apply_theme(self, theme: str, rerender: bool = True):
        self.theme = theme
        self.settings.setValue("theme", theme)
        charts.set_theme(theme)
        app = QApplication.instance()
        app.setPalette(build_palette(theme))
        app.setStyleSheet(build_qss(theme))
        self.btn_theme.setText("☀  Светлая" if theme == "dark" else "☾  Тёмная")
        self.btn_theme.setToolTip("Переключить тему оформления")
        if rerender and self.pipe is not None:
            self.refresh_analytics()
            self.refresh_aux()
            self.duel.refresh()
            if self.last_res:
                self.render_results()

    def toggle_theme(self):
        self.apply_theme("light" if self.theme == "dark" else "dark")

    # ===== Режимы =====
    def set_mode(self, pro: bool):
        self.settings.setValue("pro_mode", pro)
        for i in PRO_TABS:
            self.tabs.setTabVisible(i, pro)
        for key, row in self.slider_rows.items():
            for w in row:
                w.setVisible(pro or key in SIMPLE_IMPORTANCE)
        for field in self.pro_fields:
            self.req_form.setRowVisible(field, pro)
        self.result_stack.setCurrentIndex(1 if pro else 0)
        self.catalog_archive_box.setVisible(pro)
        self.survey_hint.setText(
            "Ответы превращаются в вектор признаков, нейросеть определяет класс телефона, а внутри класса варианты "
            "ранжируются по вашим приоритетам." if pro else
            "Выберите, для чего телефон, укажите бюджет — остальное подберём. Больше настроек — в «Про-режиме».")

    # ===== Опрос =====
    def _build_survey_tab(self) -> QWidget:
        w, outer = page()
        outer.addWidget(QLabel("<span style='font-size:16pt; font-weight:600'>Расскажите, какой телефон вам нужен</span>"))
        self.survey_hint = hint("")
        outer.addWidget(self.survey_hint)

        presets = QHBoxLayout()
        presets.addWidget(QLabel("Для чего телефон:"))
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        for name in PRESETS:
            b = QPushButton(name)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(PRESET_HINTS.get(name, ""))
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            self.preset_group.addButton(b)
            presets.addWidget(b)
        presets.addStretch()
        outer.addLayout(presets)

        profiles = QHBoxLayout()
        profiles.addWidget(hint("Мои профили:"))
        self.profile_box = QComboBox()
        self.profile_box.setMinimumWidth(200)
        b_load = QPushButton("Загрузить")
        b_load.clicked.connect(self.load_profile)
        b_save = QPushButton("Сохранить текущие ответы…")
        b_save.clicked.connect(self.save_profile)
        b_del = QPushButton("Удалить")
        b_del.clicked.connect(self.delete_profile)
        b_reset = QPushButton("Сбросить настройки")
        b_reset.clicked.connect(self.reset_survey)
        for wdg in (self.profile_box, b_load, b_save, b_del):
            profiles.addWidget(wdg)
        profiles.addStretch()
        profiles.addWidget(b_reset)
        outer.addLayout(profiles)

        row = QHBoxLayout()
        row.setSpacing(14)
        left = QGroupBox("Бюджет и требования")
        form = QFormLayout(left)
        form.setVerticalSpacing(10)
        self.req_form = form
        self.budget = QSpinBox()
        self.budget.setRange(5000, 300000)
        self.budget.setSingleStep(1000)
        self.budget.setValue(35000)
        self.budget.setSuffix(" ₽")
        self.budget.setGroupSeparatorShown(True)
        form.addRow("Бюджет", self.budget)
        self.os_pref = QComboBox()
        self.os_pref.addItems(["Неважно", "Android", "iOS"])
        form.addRow("Система", self.os_pref)
        self.need_nfc = QCheckBox("NFC — оплата телефоном")
        self.need_nfc.setChecked(True)
        form.addRow(self.need_nfc)
        self.min_rom = QComboBox()
        self.min_rom.addItems(["64", "128", "256", "512"])
        self.min_rom.setCurrentText("128")
        self.screen_pref = QComboBox()
        self.screen_pref.addItems([title for title, _ in SCREEN_PREF.values()])
        self.min_hz = QComboBox()
        self.min_hz.addItems(["Неважно"] + [f"от {hz} Гц" for hz in HZ_PREF[1:]])
        self.esim_pref = QComboBox()
        self.esim_pref.addItems(list(OPTION_PREF.values()))
        self.ir_pref = QComboBox()
        self.ir_pref.addItems(list(OPTION_PREF.values()))
        self.gaming = QCheckBox("Игры")
        self.photo = QCheckBox("Фото и видео")
        self.work = QCheckBox("Работа и мессенджеры")
        scen = QWidget()
        sl = QHBoxLayout(scen)
        sl.setContentsMargins(0, 0, 0, 0)
        for cb in (self.gaming, self.photo, self.work):
            sl.addWidget(cb)
        sl.addStretch()
        self.pro_fields = [self.min_rom, self.screen_pref, self.min_hz, self.esim_pref, self.ir_pref, scen]
        for title, widget in (("Память от, ГБ", self.min_rom), ("Размер экрана", self.screen_pref),
                              ("Частота экрана", self.min_hz), ("eSIM", self.esim_pref),
                              ("ИК-порт (пульт)", self.ir_pref), ("Сценарии", scen)):
            form.addRow(title, widget)
        self.preset_info = QFrame()
        self.preset_info.setObjectName("presetInfo")
        pil = QVBoxLayout(self.preset_info)
        self.preset_info_text = QLabel("Выберите пресет вверху — здесь появится, на что он делает упор. "
                                       "Наведите на ползунок, чтобы узнать, что он значит.")
        self.preset_info_text.setWordWrap(True)
        pil.addWidget(self.preset_info_text)
        form.addRow(self.preset_info)

        right = QGroupBox("Что для вас важнее")
        grid = QGridLayout(right)
        grid.setVerticalSpacing(12)
        grid.addWidget(hint("0 — совсем не важно, 10 — очень важно. Наведите курсор — подсказка."), 0, 0, 1, 3)
        self.sliders, self.slider_rows = {}, {}
        for row_i, key in enumerate(IMPORTANCE_FIELDS, start=1):
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 10)
            s.setValue(IMPORTANCE_DEFAULTS[key])
            val = QLabel(str(s.value()))
            val.setObjectName("value")
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            s.valueChanged.connect(lambda v, lbl=val: lbl.setText(str(v)))
            title = QLabel(IMPORTANCE_TITLES[key])
            for wdg in (title, s):
                wdg.setToolTip(IMPORTANCE_HINTS.get(key, ""))
            grid.addWidget(title, row_i, 0)
            grid.addWidget(s, row_i, 1)
            grid.addWidget(val, row_i, 2)
            self.sliders[key] = s
            self.slider_rows[key] = (title, s, val)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(len(IMPORTANCE_FIELDS) + 1, 1)

        row.addWidget(left, 1)
        row.addWidget(right, 1)
        outer.addLayout(row, 1)

        bottom = QHBoxLayout()
        self.model_status = hint("Модель обучается…")
        self.btn_pick = QPushButton("Подобрать смартфон")
        self.btn_pick.setObjectName("primary")
        self.btn_pick.setCursor(Qt.PointingHandCursor)
        self.btn_pick.setMinimumWidth(260)
        self.btn_pick.clicked.connect(self.on_pick)
        bottom.addWidget(self.model_status, 1)
        bottom.addWidget(self.btn_pick)
        outer.addLayout(bottom)
        return w

    def reset_survey(self):
        for key, s in self.sliders.items():
            s.setValue(IMPORTANCE_DEFAULTS[key])
        for cb in (self.gaming, self.photo, self.work):
            cb.setChecked(False)
        self.need_nfc.setChecked(True)
        for combo in (self.os_pref, self.screen_pref, self.min_hz, self.esim_pref, self.ir_pref):
            combo.setCurrentIndex(0)
        self.min_rom.setCurrentText("128")
        self.budget.setValue(35000)
        checked = self.preset_group.checkedButton()
        if checked:
            self.preset_group.setExclusive(False)
            checked.setChecked(False)
            self.preset_group.setExclusive(True)
        self.preset_info_text.setText("Настройки сброшены к значениям по умолчанию.")

    def apply_preset(self, name: str):
        for key, s in self.sliders.items():
            s.setValue(IMPORTANCE_DEFAULTS[key])
        values = PRESETS[name]
        for key, v in values.items():
            if key in self.sliders:
                self.sliders[key].setValue(v)
        self.gaming.setChecked(bool(values.get("gaming")))
        self.photo.setChecked(bool(values.get("photo")))
        self.work.setChecked(bool(values.get("work")))
        self.screen_pref.setCurrentIndex(values.get("screen_pref", 0))
        self.min_hz.setCurrentIndex(HZ_PREF.index(values.get("min_hz", 0)))
        self.esim_pref.setCurrentIndex(values.get("esim_pref", 0))
        self.preset_info_text.setText(f"<b>{name}:</b> {PRESET_HINTS.get(name, '')}")

    def answers(self) -> dict:
        a = {k: s.value() for k, s in self.sliders.items()}
        a.update(budget_rub=float(self.budget.value()), os_pref=self.os_pref.currentIndex(),
                 min_rom=int(self.min_rom.currentText()), need_nfc=int(self.need_nfc.isChecked()),
                 esim_pref=self.esim_pref.currentIndex(), ir_pref=self.ir_pref.currentIndex(),
                 screen_pref=self.screen_pref.currentIndex(), min_hz=HZ_PREF[self.min_hz.currentIndex()],
                 gaming=int(self.gaming.isChecked()), photo=int(self.photo.isChecked()),
                 work=int(self.work.isChecked()))
        return a

    def set_answers(self, a: dict):
        for k, s in self.sliders.items():
            s.setValue(int(a.get(k, IMPORTANCE_DEFAULTS[k])))
        self.budget.setValue(int(a.get("budget_rub", 35000)))
        self.os_pref.setCurrentIndex(int(a.get("os_pref", 0)))
        self.min_rom.setCurrentText(str(int(a.get("min_rom", 128))))
        self.need_nfc.setChecked(bool(a.get("need_nfc", 1)))
        self.esim_pref.setCurrentIndex(int(a.get("esim_pref", 0)))
        self.ir_pref.setCurrentIndex(int(a.get("ir_pref", 0)))
        self.screen_pref.setCurrentIndex(int(a.get("screen_pref", 0)))
        self.min_hz.setCurrentIndex(HZ_PREF.index(int(a.get("min_hz", 0))))
        self.gaming.setChecked(bool(a.get("gaming")))
        self.photo.setChecked(bool(a.get("photo")))
        self.work.setChecked(bool(a.get("work")))

    # --- профили
    def _profiles(self) -> dict:
        return json.loads(self.settings.value("profiles", "{}"))

    def reload_profiles(self):
        self.profile_box.clear()
        self.profile_box.addItems(sorted(self._profiles()))

    def save_profile(self):
        name, ok = QInputDialog.getText(self, "Профиль", "Название профиля:")
        if ok and name.strip():
            prof = self._profiles()
            prof[name.strip()] = self.answers()
            self.settings.setValue("profiles", json.dumps(prof, ensure_ascii=False))
            self.reload_profiles()
            self.profile_box.setCurrentText(name.strip())

    def load_profile(self):
        a = self._profiles().get(self.profile_box.currentText())
        if a:
            self.set_answers(a)
            self.preset_info_text.setText(f"Загружен профиль «{self.profile_box.currentText()}».")

    def delete_profile(self):
        prof = self._profiles()
        prof.pop(self.profile_box.currentText(), None)
        self.settings.setValue("profiles", json.dumps(prof, ensure_ascii=False))
        self.reload_profiles()

    # ===== Подбор =====
    def on_pick(self):
        if self.pipe is None:
            QMessageBox.information(self, "Подождите", "Модель ещё обучается.")
            return
        a = self.answers()
        res = recommend(self.pipe, a, top_n=30, fair=self.fair, diverse=self.diverse_box.isChecked())
        self.last_answers, self.last_res = a, res
        self.survey_id = db.save_survey(a, res["pred_class"])
        best = res["table"].iloc[0]
        self.result_header.setText(
            f"<span style='font-size:15pt; font-weight:600'>Вам подойдёт: {best['Телефон']}</span><br>"
            f"<span style='color:#7A8194'>Нейросеть отнесла запрос к классу <b>{res['class_name']}</b> "
            f"({res['proba'][res['pred_class']]:.0%}), ближайшая группа — <b>{res['cluster_name']}</b>. "
            f"Подходящих под все требования моделей: {res['feasible']}. "
            f"Отметьте, что подошло, — это поможет дообучить модель.</span>")
        hint_ = res["hint"]
        self.hint_bar.setVisible(bool(hint_))
        if hint_:
            self.hint_text.setText(hint_["text"])
            self.hint_btn.setVisible(hint_["key"] != "budget" or "Самый дешёвый" in hint_["text"])
            self.hint_btn.setText("Поднять бюджет" if hint_["key"] == "budget" else "Ослабить это требование")
        self.render_results()
        self.tabs.setCurrentIndex(TAB_RESULT)

    def relax_requirement(self):
        key = (self.last_res or {}).get("hint", {}) or {}
        key = key.get("key")
        if key == "esim":
            self.esim_pref.setCurrentIndex(1)
        elif key == "ir":
            self.ir_pref.setCurrentIndex(1)
        elif key == "screen":
            self.screen_pref.setCurrentIndex(0)
        elif key == "hz":
            self.min_hz.setCurrentIndex(0)
        elif key == "rom":
            self.min_rom.setCurrentIndex(max(0, self.min_rom.currentIndex() - 1))
        elif key == "os":
            self.os_pref.setCurrentIndex(0)
        elif key == "nfc":
            self.need_nfc.setChecked(False)
        elif key == "budget":
            self.budget.setValue(int(self.budget.value() * 1.3))
        self.on_pick()

    def sorted_table(self) -> pd.DataFrame:
        t = self.last_res["table"].copy()
        if "Справедливая, ₽" in t:
            t["К справедливой, %"] = ((t["Цена, ₽"] / t["Справедливая, ₽"] - 1) * 100).round(1)
        col, asc = SORTS[self.sort_box.currentText()]
        if col in t and t[col].notna().any():
            t = t.sort_values(col, ascending=asc, kind="stable")
        return t

    def render_results(self):
        if not self.last_res:
            return
        res = self.last_res
        n = int(self.count_box.currentText())
        table = self.sorted_table()
        self.result_model.set_df(table.head(n).drop(columns=["_pros", "_badges"]))
        self.result_view.setColumnHidden(0, True)
        ids = table["id"].head(n).tolist()
        charts.plot_proba(self.proba_canvas, res["proba"])
        charts.plot_reco_scores(self.scores_canvas, table.head(10))
        charts.plot_reco_radar(self.radar_canvas, self.pipe, ids)
        charts.plot_clusters(self.result_cluster_canvas, self.pipe, ids)
        self.result_view.selectRow(0)
        self.show_explain()
        # простой режим: блоки + «показать ещё»
        clear_layout(self.cards_layout)
        shown = set()
        for block in res["blocks"]:
            self.cards_layout.addWidget(self._block_title(block["title"], block.get("note", "")))
            for r in block["rows"]:
                self.cards_layout.addWidget(self._card(r))
                shown.add(r["id"])
        self.more_rows = [r for _, r in table.iterrows() if r["id"] not in shown]
        self.cards_shown = 0
        self.more_title = self._block_title("Ещё варианты", "по убыванию оценки")
        self.cards_layout.addWidget(self.more_title)
        self.more_holder = QVBoxLayout()
        self.cards_layout.addLayout(self.more_holder)
        self.show_more_cards()
        self.cards_layout.addStretch()

    def _block_title(self, title: str, note: str) -> QLabel:
        lbl = QLabel(f"{title} <span style='font-size:9pt; color:#7A8194; font-weight:400'>{note}</span>")
        lbl.setObjectName("blockTitle")
        return lbl

    def _card(self, r, badge: str = "") -> PhoneCard:
        p = phone_row(self.pipe, int(r["id"]), self.fair)
        return PhoneCard(p, pros=r["_pros"], badge=badge, on_similar=self.open_similar, on_like=self.like_phone,
                         on_favorite=self.toggle_favorite, is_favorite=str(p["model"]) in self.favorites)

    def show_more_cards(self):
        nxt = self.more_rows[self.cards_shown:self.cards_shown + CARD_STEP]
        for r in nxt:
            self.more_holder.addWidget(self._card(r))
        self.cards_shown += len(nxt)
        self.more_btn.setEnabled(self.cards_shown < len(self.more_rows))

    # ===== Результат =====
    def _build_result_tab(self) -> QWidget:
        w, lay = page()
        banner = QFrame()
        banner.setObjectName("banner")
        bl = QVBoxLayout(banner)
        bl.setContentsMargins(16, 12, 16, 12)
        self.result_header = QLabel("Пройдите опрос на первой вкладке — здесь появятся рекомендации.")
        self.result_header.setWordWrap(True)
        bl.addWidget(self.result_header)
        lay.addWidget(banner)

        self.hint_bar = QFrame()
        self.hint_bar.setObjectName("hintBar")
        hb = QHBoxLayout(self.hint_bar)
        self.hint_text = QLabel()
        self.hint_text.setWordWrap(True)
        self.hint_btn = QPushButton("Ослабить это требование")
        self.hint_btn.clicked.connect(self.relax_requirement)
        hb.addWidget(self.hint_text, 1)
        hb.addWidget(self.hint_btn)
        self.hint_bar.hide()
        lay.addWidget(self.hint_bar)

        controls = QHBoxLayout()
        self.diverse_box = QCheckBox("Разнообразие выдачи (≤ 2 модели бренда в топ-5, ≥ 3 группы в топ-10)")
        self.diverse_box.setChecked(True)
        self.diverse_box.toggled.connect(lambda _: self.last_res and self.on_pick())
        self.count_box = QComboBox()
        self.count_box.addItems(["10", "20", "30"])
        self.count_box.currentIndexChanged.connect(lambda _: self.render_results())
        self.sort_box = QComboBox()
        self.sort_box.addItems(list(SORTS))
        self.sort_box.currentIndexChanged.connect(lambda _: self.render_results())
        b_export = QPushButton("Экспорт подборки…")
        b_export.clicked.connect(self.export_selection)
        controls.addWidget(self.diverse_box)
        controls.addStretch()
        controls.addWidget(hint("Сортировка:"))
        controls.addWidget(self.sort_box)
        controls.addWidget(hint("Показать в таблице:"))
        controls.addWidget(self.count_box)
        controls.addWidget(b_export)
        lay.addLayout(controls)

        self.result_stack = QStackedWidget()
        lay.addWidget(self.result_stack, 1)

        # простой режим: блоки карточек
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setObjectName("page")
        outer = QVBoxLayout(holder)
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setSpacing(10)
        outer.addLayout(self.cards_layout)
        self.more_btn = QPushButton("Показать ещё")
        self.more_btn.clicked.connect(self.show_more_cards)
        outer.addWidget(self.more_btn, alignment=Qt.AlignHCenter)
        scroll.setWidget(holder)
        self.result_stack.addWidget(scroll)

        # про-режим: таблица и графики
        self.result_model = PandasModel()
        self.result_view = make_table(self.result_model)
        self.result_view.selectionModel().selectionChanged.connect(lambda *_: self.show_explain())
        btns = QHBoxLayout()
        for text, liked, name in (("Подходит", True, "like"), ("Не подходит", False, "dislike")):
            b = QPushButton(text)
            b.setObjectName(name)
            b.clicked.connect(lambda _=False, v=liked: self.on_feedback(v))
            btns.addWidget(b)
        b_sim = QPushButton("Похожие и история цены")
        b_sim.clicked.connect(lambda: self._with_selected(self.result_view, self.result_model, self.open_similar))
        b_duel = QPushButton("В дуэль с лидером")
        b_duel.clicked.connect(self.duel_with_best)
        b_fav = QPushButton("☆ В избранное")
        b_fav.clicked.connect(lambda: self._with_selected(self.result_view, self.result_model, self.favorite_by_id))
        for b in (b_sim, b_duel, b_fav):
            btns.addWidget(b)
        btns.addWidget(hint("выделите строку"))
        btns.addStretch()
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.result_view)
        tl.addLayout(btns)
        self.proba_canvas = charts.MplCanvas(4, 3)
        self.scores_canvas = charts.MplCanvas(6, 3)
        self.radar_canvas = charts.MplCanvas(6, 4)
        self.result_cluster_canvas = charts.MplCanvas(8, 4)
        self.explain_canvas = charts.MplCanvas(8, 4)
        chart_tabs = QTabWidget()
        chart_tabs.addTab(split(Qt.Horizontal, self.proba_canvas, self.scores_canvas, sizes=[400, 700]),
                          "Класс и оценки")
        chart_tabs.addTab(self.explain_canvas, "Почему он?")
        chart_tabs.addTab(self.radar_canvas, "Сравнение топ-3")
        chart_tabs.addTab(self.result_cluster_canvas, "На карте групп")
        self.result_stack.addWidget(split(Qt.Vertical, top, chart_tabs, sizes=[380, 360]))
        return w

    def _with_selected(self, view, model, fn):
        pid = selected_id(view, model)
        if pid is None:
            QMessageBox.information(self, "Телефон", "Сначала выделите строку с телефоном.")
            return
        fn(pid)

    def show_explain(self):
        pid = selected_id(self.result_view, self.result_model)
        if pid is None or self.last_answers is None:
            return
        p = phone_row(self.pipe, pid)
        charts.plot_contrib(self.explain_canvas, explain(self.pipe, self.last_answers, pid),
                            f"Почему {p['full_name']}: вклад параметров в оценку")

    def phone_advice(self, phone_id: int, hist: pd.DataFrame) -> str | None:
        if self.forecast is None or self.pipe is None:
            return None
        i = np.flatnonzero(self.pipe.phones_["id"].to_numpy() == phone_id)
        tier = TIER_NAMES.get(int(self.pipe.labels_[i[0]])) if len(i) else None
        text = advice(hist, self.forecast.change(tier) if tier else None)
        fair = (self.fair or {}).get(phone_id)
        if fair and len(i):
            diff = self.pipe.phones_.iloc[i[0]]["price_rub"] / fair - 1
            text = (text + " " if text else "") + f"Справедливая цена по модели — {fair:,.0f} ₽ ({diff:+.0%}).".replace(",", " ")
        return text

    def open_similar(self, phone_id: int):
        if self.pipe is not None:
            SimilarDialog(self.pipe, phone_id, self, fair=self.fair, history_fn=db.get_price_history,
                          advice_fn=self.phone_advice, on_favorite=self.toggle_favorite,
                          favorites=self.favorites).exec()

    def like_phone(self, phone_id: int):
        if self.survey_id is not None:
            db.save_feedback(self.survey_id, phone_id, True)
            self.refresh_db_stats()
            self.statusBar().showMessage("Спасибо! Отзыв сохранён и будет учтён при переобучении.", 5000)

    def on_feedback(self, liked: bool):
        pid = selected_id(self.result_view, self.result_model)
        if pid is None or self.survey_id is None:
            QMessageBox.information(self, "Отзыв", "Выберите строку с телефоном.")
            return
        db.save_feedback(self.survey_id, pid, liked)
        self.refresh_db_stats()
        self.statusBar().showMessage("Спасибо! Отзыв сохранён и будет учтён при переобучении.", 5000)

    def duel_with_best(self):
        pid = selected_id(self.result_view, self.result_model)
        if pid is None or self.result_model.df.empty:
            return
        best = int(self.result_model.df.iloc[0]["id"])
        other = pid if pid != best else int(self.result_model.df.iloc[1]["id"])
        self.duel.set_pipe(self.pipe, best, other)
        self.tabs.setCurrentIndex(TAB_FUN)
        self.fun_tabs.setCurrentIndex(0)

    def export_selection(self):
        if not self.last_res:
            QMessageBox.information(self, "Экспорт", "Сначала подберите телефоны.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить подборку", "подборка.xlsx", "Excel (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        table = self.sorted_table().drop(columns=["_pros", "_badges", "id"])
        if path.lower().endswith(".csv"):
            table.to_csv(path, index=False, encoding="utf-8-sig")
        else:
            blocks = pd.DataFrame([{"блок": b["title"], "телефон": r["Телефон"], "цена, ₽": r["Цена, ₽"],
                                    "почему": r["Почему"]} for b in self.last_res["blocks"] for r in b["rows"]])
            answers = pd.DataFrame([{"параметр": k, "значение": v} for k, v in self.last_answers.items()])
            with pd.ExcelWriter(path, engine="openpyxl") as xl:
                blocks.to_excel(xl, sheet_name="Блоки", index=False)
                table.to_excel(xl, sheet_name="Топ", index=False)
                answers.to_excel(xl, sheet_name="Ответы", index=False)
        self.statusBar().showMessage(f"Подборка сохранена: {path}", 6000)

    # ===== Избранное =====
    def toggle_favorite(self, model: str):
        if model in self.favorites:
            self.favorites.discard(model)
        else:
            self.favorites.add(model)
        self.settings.setValue("favorites", json.dumps(sorted(self.favorites), ensure_ascii=False))
        self.statusBar().showMessage(f"В избранном: {len(self.favorites)}", 3000)

    def favorite_by_id(self, phone_id: int):
        self.toggle_favorite(str(self.pipe.phones_.loc[self.pipe.phones_["id"] == phone_id, "model"].iloc[0]))

    def _build_favorites_tab(self) -> QWidget:
        w, lay = page()
        top = QHBoxLayout()
        top.addWidget(QLabel("<span style='font-size:14pt; font-weight:600'>Избранное и сравнение</span>"))
        top.addStretch()
        b_exp = QPushButton("Экспорт сравнения…")
        b_exp.clicked.connect(self.export_favorites)
        b_clear = QPushButton("Очистить")
        b_clear.clicked.connect(self.clear_favorites)
        top.addWidget(b_exp)
        top.addWidget(b_clear)
        lay.addLayout(top)
        lay.addWidget(hint("Добавляйте телефоны кнопкой «☆ В избранное» в карточках, результате и каталоге. "
                           "Ниже — таблица сравнения всех выбранных: зелёным в колонке «лучший» — у кого параметр сильнее."))
        self.fav_compare = PandasModel()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setObjectName("page")
        self.fav_cards = QGridLayout(holder)
        scroll.setWidget(holder)
        lay.addWidget(split(Qt.Vertical, scroll, make_table(self.fav_compare), sizes=[380, 320]), 1)
        return w

    def favorite_rows(self) -> pd.DataFrame:
        if self.pipe is None:
            return pd.DataFrame()
        return self.pipe.phones_[self.pipe.phones_["model"].isin(self.favorites)]

    def refresh_favorites(self):
        clear_layout(self.fav_cards)
        fav = self.favorite_rows()
        if fav.empty:
            self.fav_cards.addWidget(hint("Пока пусто."), 0, 0)
            self.fav_compare.set_df(pd.DataFrame())
            return
        for k, (_, p) in enumerate(fav.iterrows()):
            card = PhoneCard(phone_row(self.pipe, int(p["id"]), self.fair), compact=True, on_similar=self.open_similar,
                             on_favorite=self.toggle_favorite, is_favorite=True)
            self.fav_cards.addWidget(card, k // 3, k % 3)
        rows = []
        names = [full_name(b, m) for b, m in zip(fav["brand"], fav["model"])]
        for col, title, sign in DUEL_ROWS:
            vals = fav[col].to_numpy(float)
            row = {"параметр": title}
            row.update({n: (f"{v:,.0f}".replace(",", " ") if v >= 1000 else f"{v:g}") for n, v in zip(names, vals)})
            if sign:
                row["лучший"] = names[int(np.argmax(vals * sign))]
            rows.append(row)
        if self.fair:
            row = {"параметр": "К справедливой цене"}
            row.update({n: f"{p / self.fair.get(int(i), p) - 1:+.0%}" for n, p, i in zip(names, fav["price_rub"], fav["id"])})
            rows.append(row)
        self.fav_compare.set_df(pd.DataFrame(rows))

    def export_favorites(self):
        if self.fav_compare.df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сравнение", "сравнение.xlsx", "Excel (*.xlsx)")
        if path:
            self.fav_compare.df.to_excel(path, index=False, engine="openpyxl")
            self.statusBar().showMessage(f"Сравнение сохранено: {path}", 6000)

    def clear_favorites(self):
        self.favorites.clear()
        self.settings.setValue("favorites", "[]")
        self.refresh_favorites()

    # ===== Каталог =====
    def _build_catalog_tab(self) -> QWidget:
        w, lay = page()
        box = QGroupBox("Фильтры")
        f = QGridLayout(box)
        f.setHorizontalSpacing(12)
        self.f_search = QLineEdit()
        self.f_search.setPlaceholderText("Поиск по названию…")
        self.f_price_min, self.f_price_max = QSpinBox(), QSpinBox()
        for s, v in ((self.f_price_min, 0), (self.f_price_max, 300000)):
            s.setRange(0, 300000)
            s.setSingleStep(5000)
            s.setValue(v)
            s.setSuffix(" ₽")
            s.setGroupSeparatorShown(True)
        self.f_ram = QComboBox()
        self.f_ram.addItems(["любая", "4", "6", "8", "12", "16"])
        self.f_rom = QComboBox()
        self.f_rom.addItems(["любая", "64", "128", "256", "512"])
        self.f_active = QSpinBox()
        self.f_active.setRange(0, 40)
        self.f_active.setSuffix(" ч")
        self.f_screen = QDoubleSpinBox()
        self.f_screen.setRange(0, 9)
        self.f_screen.setSingleStep(0.1)
        self.f_screen.setValue(9)
        self.f_screen.setPrefix("до ")
        self.f_os = QComboBox()
        self.f_os.addItems(["любая", "Android", "iOS"])
        self.f_brand = QComboBox()
        self.f_nfc, self.f_5g, self.f_esim, self.f_ir = QCheckBox("NFC"), QCheckBox("5G"), QCheckBox("eSIM"), QCheckBox("ИК")
        self.f_rugged = QCheckBox("только защищённые")
        self.f_deal = QCheckBox("только «выгодно» (≥ 15% ниже справедливой)")
        self.f_hz = QComboBox()
        self.f_hz.addItems(["любая"] + [f"от {hz} Гц" for hz in HZ_PREF[1:]])
        items = [("Название", self.f_search), ("Цена от", self.f_price_min), ("Цена до", self.f_price_max),
                 ("ОЗУ от, ГБ", self.f_ram), ("Память от, ГБ", self.f_rom), ("Активная работа от", self.f_active),
                 ("Экран, дюйм", self.f_screen), ("Частота экрана", self.f_hz), ("ОС", self.f_os), ("Бренд", self.f_brand)]
        for i, (title, widget) in enumerate(items):
            f.addWidget(hint(title), i // 5 * 2, i % 5)
            f.addWidget(widget, i // 5 * 2 + 1, i % 5)
        checks = QHBoxLayout()
        checks.setSpacing(14)
        for cb in (self.f_nfc, self.f_5g, self.f_esim, self.f_ir, self.f_rugged, self.f_deal):
            checks.addWidget(cb)
        checks.addStretch()
        f.addWidget(hint("Наличие"), 4, 0)
        f.addLayout(checks, 5, 0, 1, 5)
        for wdg in (self.f_price_min, self.f_price_max, self.f_active, self.f_screen):
            wdg.valueChanged.connect(self.refresh_catalog)
        for wdg in (self.f_ram, self.f_rom, self.f_os, self.f_brand, self.f_hz):
            wdg.currentIndexChanged.connect(self.refresh_catalog)
        for wdg in (self.f_nfc, self.f_5g, self.f_esim, self.f_ir, self.f_rugged, self.f_deal):
            wdg.toggled.connect(self.refresh_catalog)
        self.f_search.textChanged.connect(self.refresh_catalog)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.catalog_count = hint("")
        row.addWidget(self.catalog_count, 1)
        self.catalog_archive_box = QWidget()
        ab = QHBoxLayout(self.catalog_archive_box)
        ab.setContentsMargins(0, 0, 0, 0)
        self.f_archived = QCheckBox("показывать архив")
        self.f_archived.toggled.connect(self.refresh_catalog)
        b_arch = QPushButton("В архив / вернуть")
        b_arch.setToolTip("Снятые с продажи модели не участвуют в подборе, но остаются в базе")
        b_arch.clicked.connect(lambda: self._with_selected(self.catalog_view, self.catalog_model, self.toggle_archive))
        ab.addWidget(self.f_archived)
        ab.addWidget(b_arch)
        row.addWidget(self.catalog_archive_box)
        b_fav = QPushButton("☆ В избранное")
        b_fav.clicked.connect(lambda: self._with_selected(self.catalog_view, self.catalog_model, self.favorite_by_id))
        b_sim = QPushButton("Похожие и история цены")
        b_sim.clicked.connect(lambda: self._with_selected(self.catalog_view, self.catalog_model, self.open_similar))
        row.addWidget(b_fav)
        row.addWidget(b_sim)
        lay.addLayout(row)
        self.catalog_model = PandasModel()
        self.catalog_view = make_table(self.catalog_model)
        self.catalog_view.doubleClicked.connect(
            lambda _: self._with_selected(self.catalog_view, self.catalog_model, self.open_similar))
        lay.addWidget(self.catalog_view, 1)
        lay.addWidget(hint("Двойной клик по строке — карточка телефона, история цены и похожие."))
        return w

    def toggle_archive(self, phone_id: int):
        row = self.phones_all[self.phones_all["id"] == phone_id]
        if row.empty:
            return
        db.set_archived(phone_id, not bool(row["archived"].fillna(0).iloc[0]))
        self.reload_phones()
        self.refresh_catalog()
        self.statusBar().showMessage("Архив изменён. Нажмите «Переобучить», чтобы подбор это учёл.", 6000)

    def refresh_catalog(self, *_):
        df = self.phones_all if self.f_archived.isChecked() else self.phones
        if self.f_brand.count() == 0:
            self.f_brand.blockSignals(True)
            self.f_brand.addItems(["любой"] + sorted(self.phones_all["brand"].unique()))
            self.f_brand.blockSignals(False)
        m = (df["price_rub"] >= self.f_price_min.value()) & (df["price_rub"] <= self.f_price_max.value())
        m &= df["active_h"] >= self.f_active.value()
        m &= df["screen_in"] <= self.f_screen.value()
        if self.f_ram.currentIndex():
            m &= df["ram_gb"] >= int(self.f_ram.currentText())
        if self.f_rom.currentIndex():
            m &= df["rom_gb"] >= int(self.f_rom.currentText())
        if self.f_os.currentIndex():
            m &= df["os"] == self.f_os.currentText()
        if self.f_brand.currentIndex() > 0:
            m &= df["brand"] == self.f_brand.currentText()
        if self.f_hz.currentIndex():
            m &= df["refresh_hz"] >= HZ_PREF[self.f_hz.currentIndex()]
        for cb, col in ((self.f_nfc, "nfc"), (self.f_5g, "has_5g"), (self.f_esim, "esim"), (self.f_ir, "ir_port")):
            if cb.isChecked():
                m &= df[col] == 1
        if self.f_rugged.isChecked():
            m &= df["weight_g"] >= 300
        text = self.f_search.text().strip().lower()
        if text:
            m &= (df["brand"] + " " + df["model"]).str.lower().str.contains(text, regex=False)
        sel = df[m]
        view = sel[list(CATALOG_COLUMNS)].rename(columns=CATALOG_COLUMNS)
        if self.fair:
            fair = sel["id"].map(self.fair)
            view.insert(4, "Справедливая, ₽", fair.round(-2))
            view.insert(5, "К справедливой, %", ((sel["price_rub"] / fair - 1) * 100).round(1))
            if self.f_deal.isChecked():
                view = view[view["К справедливой, %"] <= -15]
        if self.pipe is not None:
            pos = pd.Series(range(len(self.pipe.phones_)), index=self.pipe.phones_["id"]).reindex(view["id"])
            view["Класс"] = [CLASS_NAMES[self.pipe.labels_[int(p)]] if pd.notna(p) else "" for p in pos]
            view["Группа"] = [self.pipe.cluster_names_[int(self.pipe.clusters_[int(p)])] if pd.notna(p) else ""
                              for p in pos]
        if self.f_archived.isChecked():
            view["В архиве"] = sel.loc[view.index, "archived"].fillna(0).astype(int).map({0: "", 1: "да"}).values
        self.catalog_model.set_df(view)
        self.catalog_view.setColumnHidden(0, True)
        self.catalog_count.setText(f"Показано {len(view)} из {len(df)} моделей")

    # ===== Аналитика =====
    def _build_analytics_tab(self) -> QWidget:
        w, lay = page()
        top = QHBoxLayout()
        self.btn_retrain = QPushButton("Переобучить с учётом отзывов")
        self.btn_retrain.clicked.connect(self.retrain)
        self.btn_exp = QPushButton("Сравнить модели (2–4 мин)")
        self.btn_exp.clicked.connect(self.run_experiments)
        b_xlsx = QPushButton("Экспорт всего в Excel")
        b_xlsx.clicked.connect(self.export_report)
        self.metrics_label = hint("")
        for b in (self.btn_retrain, self.btn_exp, b_xlsx):
            top.addWidget(b)
        top.addSpacing(10)
        top.addWidget(self.metrics_label, 1)
        lay.addLayout(top)

        C = charts.MplCanvas
        self.c_clusters, self.c_radar, self.c_pricebox = C(9, 5), C(6, 4.5), C(6, 4)
        self.c_priceperf, self.c_classes, self.c_crossmx = C(10, 5), C(6, 3.5), C(6, 3.5)
        self.c_elbow, self.c_sil, self.c_dendro = C(9, 3.5), C(9, 4), C(10, 5)
        self.c_corr, self.c_weights, self.c_train, self.c_ablation = C(7, 6), C(5, 5), C(10, 4), C(9, 4)
        self.c_roc, self.c_regions, self.c_enserr = C(6, 4.5), C(7, 5), C(6, 4)
        self.c_regr, self.c_tree, self.c_forecast = C(10, 6), C(12, 5), C(10, 4.5)
        self.profile_model, self.tier_model = PandasModel(), PandasModel()
        self.algo_model, self.weights_exp_model = PandasModel(), PandasModel()
        self.ablation_model, self.weights_model = PandasModel(), PandasModel()
        self.regr_model, self.suspicious_model = PandasModel(), PandasModel()
        self.forecast_model, self.wine_model = PandasModel(), PandasModel()

        sub = QTabWidget()
        sub.addTab(split(Qt.Vertical, self.c_clusters, make_table(self.profile_model), sizes=[520, 200]),
                   "Карта групп")
        sub.addTab(split(Qt.Horizontal, self.c_radar, self.c_pricebox), "Профили групп")
        sub.addTab(self.c_priceperf, "Цена и мощность")
        sub.addTab(split(Qt.Horizontal, self.c_classes, self.c_crossmx), "Классы")
        kw = QWidget()
        kl = QVBoxLayout(kw)
        kl.setContentsMargins(0, 0, 0, 0)
        kl.addWidget(hint("Группы ищутся отдельно в каждом ценовом шпиле; k — по силуэту (3–5, группа ≥ 8 моделей); "
                          "ARI — насколько разбиение одинаково при разных random_state (1 — всегда одинаково)."))
        kl.addWidget(split(Qt.Vertical, make_table(self.tier_model), self.c_elbow, self.c_sil, sizes=[150, 250, 300]))
        sub.addTab(kw, "Выбор k и стабильность")
        sub.addTab(self.c_dendro, "Дендрограмма")
        sub.addTab(split(Qt.Horizontal, self.c_corr, self.c_weights,
                         split(Qt.Vertical, make_table(self.weights_model), make_table(self.weights_exp_model),
                               make_table(self.algo_model)), sizes=[600, 400, 400]),
                   "Признаки и веса")
        sub.addTab(self.c_train, "Нейросеть")
        sub.addTab(split(Qt.Vertical, self.c_ablation, make_table(self.ablation_model), sizes=[420, 260]),
                   "Сравнение моделей")
        sub.addTab(split(Qt.Horizontal, self.c_roc, split(Qt.Vertical, self.c_regions, self.c_enserr, sizes=[420, 280]),
                         sizes=[600, 700]), "Ансамбли и ROC")
        regr = QTabWidget()
        regr.addTab(split(Qt.Vertical, self.c_regr, make_table(self.regr_model), sizes=[560, 200]), "Сравнение методов")
        regr.addTab(self.c_tree, "Дерево решений")
        sus = QWidget()
        sl = QVBoxLayout(sus)
        sl.addWidget(hint("Модели, чья цена сильнее всего отличается от справедливой. Это или реально выгодные / "
                          "переоценённые телефоны, или ошибки в данных (не та версия, не та цена) — список на проверку."))
        sl.addWidget(make_table(self.suspicious_model), 1)
        regr.addTab(sus, "Подозрительные цены")
        sub.addTab(regr, "Регрессия цены")
        sub.addTab(split(Qt.Vertical, self.c_forecast, make_table(self.forecast_model), sizes=[420, 280]),
                   "Прогноз цен")
        wine = QWidget()
        wl = QVBoxLayout(wine)
        wl.addWidget(hint("Проверка методов на эталонном наборе UCI Wine (178 вин, 3 сорта): если наши модели дают "
                          "точность, близкую к опубликованной, значит они реализованы правильно."))
        wl.addWidget(make_table(self.wine_model), 1)
        sub.addTab(wine, "Эталон Wine")
        lay.addWidget(sub, 1)
        return w

    def refresh_analytics(self):
        p = self.pipe
        charts.plot_clusters(self.c_clusters, p)
        self.profile_model.set_df(clustering.cluster_profile_table(p.phones_, p.clusters_, p.cluster_names_)
                                  .reset_index())
        self.tier_model.set_df(pd.DataFrame(p.tier_stats_))
        charts.plot_cluster_radar(self.c_radar, p)
        charts.plot_price_box(self.c_pricebox, p)
        charts.plot_price_perf(self.c_priceperf, p)
        charts.plot_classes(self.c_classes, p)
        charts.plot_class_cluster_matrix(self.c_crossmx, p)
        budget = p.x_[(p.labels_ == 0) & ~p.rugged_]
        k_budget = next((s["k"] for s in p.tier_stats_ if s["шпиль"] == "Бюджет"), 4)
        charts.plot_elbow_silhouette(self.c_elbow, clustering.elbow(budget, 8), clustering.silhouette_by_k(budget, range(2, 9)),
                                     k_budget)
        charts.plot_silhouette(self.c_sil, clustering.silhouette_values(p.x_, p.clusters_), p.clusters_,
                               p.cluster_names_)
        charts.plot_dendrogram(self.c_dendro, clustering.hierarchical_linkage(p.x_), p.phones_["model"].tolist())
        charts.plot_corr(self.c_corr, p)
        wt = p.space_.weights_table()
        self.weights_model.set_df(wt)
        charts.plot_weights(self.c_weights, wt)
        charts.plot_training(self.c_train, p)
        charts.plot_ensemble_error(self.c_enserr)
        t = p.metrics_["test"]
        self.metrics_label.setText(
            f"Нейросеть на тесте: accuracy {t['accuracy']:.3f} · F1 {t['f1_macro']:.3f} · "
            f"ROC AUC {t['roc_auc_ovr']:.3f} · групп: {p.k_} · отзывов учтено: {p.n_feedback_}")
        self.refresh_experiments()

    def refresh_experiments(self):
        if not self.exp:
            return
        charts.plot_ablation(self.c_ablation, self.exp["ablation"])
        self.ablation_model.set_df(self.exp["ablation"].round(4))
        self.algo_model.set_df(self.exp["algorithms"].round(4))
        self.weights_exp_model.set_df(self.exp["weights"].round(4))
        if "roc" in self.exp:
            charts.plot_roc(self.c_roc, self.exp["roc"])
        if "regions" in self.exp:
            charts.plot_regions(self.c_regions, self.exp["regions"])
        if "wine" in self.exp:
            self.wine_model.set_df(self.exp["wine"].round(3))

    def refresh_aux(self):
        if self.price_model is not None:
            pm = self.price_model
            charts.plot_regression(self.c_regr, pm)
            charts.plot_tree(self.c_tree, pm)
            self.regr_model.set_df(pm.results_.round(3))
            o = pm.oof_.assign(diff=lambda d: (d["price"] / d["fair"] - 1) * 100)
            names = self.pipe.phones_.set_index("id")
            o = o.assign(телефон=[full_name(names.loc[i, "brand"], names.loc[i, "model"]) for i in o["id"]],
                         источник=[names.loc[i, "price_source"] for i in o["id"]])
            sus = pd.concat([o.nsmallest(12, "diff"), o.nlargest(12, "diff")])
            self.suspicious_model.set_df(sus[["телефон", "price", "fair", "diff", "источник"]].rename(
                columns={"price": "цена, ₽", "fair": "справедливая, ₽", "diff": "разница, %"}).round(0))
        if self.forecast is not None and len(self.forecast.results_):
            charts.plot_forecast(self.c_forecast, self.forecast)
            pv = self.forecast.results_.pivot_table(index="метод", columns=["ряд", "горизонт"], values="MAE, п.п.")
            pv.columns = [f"{a} · {b}" for a, b in pv.columns]
            self.forecast_model.set_df(pv.round(2).reset_index())

    # ===== Дуэль, игра, песочница (про-режим) =====
    def _build_fun_tab(self) -> QWidget:
        w, lay = page()
        self.fun_tabs = QTabWidget()
        self.duel = DuelWidget()
        self.game = GuessGame()
        self.sandbox = SandboxWidget()
        self.sandbox.train_requested.connect(self.train_sandbox)
        self.sandbox.btn_save.clicked.connect(self.save_sandbox)
        self.fun_tabs.addTab(self.duel, "Дуэль")
        self.fun_tabs.addTab(self.game, "Угадай класс")
        self.fun_tabs.addTab(self.sandbox, "Песочница модели")
        lay.addWidget(self.fun_tabs, 1)
        return w

    def train_sandbox(self, params: dict):
        # слоты — методы окна (а не lambda), чтобы Qt выполнил их в главном потоке
        task = Task(lambda: (params, PhonePipeline(**params).fit(self.phones, db.get_liked_feedback(),
                                                                  db.get_liked_feedback(False))))
        task.signals.done.connect(self._sandbox_done)
        task.signals.error.connect(self._sandbox_error)
        pool().start(task)

    def _sandbox_error(self, msg: str):
        self.sandbox.reset_button()
        self.on_error(msg)

    def _sandbox_done(self, result):
        params, pipe = result
        self.sandbox.add_result(params, pipe)
        self.on_model_ready(pipe, save=False)
        self.statusBar().showMessage("Модель из песочницы активна (не сохранена). Карта и метрики обновлены.", 6000)

    def save_sandbox(self):
        if self.pipe is not None:
            self.pipe.save()
            self.statusBar().showMessage("Текущая модель сохранена как основная.", 5000)

    # ===== Обучение =====
    def retrain(self):
        self.btn_pick.setEnabled(False)
        self.btn_retrain.setEnabled(False)
        self.model_status.setText("Обучаем модель: метки → группы по шпилям → нейросеть…")
        task = Task(lambda: PhonePipeline().fit(self.phones, db.get_liked_feedback(), db.get_liked_feedback(False)))
        task.signals.done.connect(self.on_model_ready)
        task.signals.error.connect(self.on_error)
        pool().start(task)

    def on_model_ready(self, pipe: PhonePipeline, save: bool = True):
        self.pipe = pipe
        if save:
            pipe.save()
        self.btn_pick.setEnabled(True)
        self.btn_retrain.setEnabled(True)
        t = pipe.metrics_["test"]
        self.model_status.setText(f"Модель готова · точность {t['accuracy']:.0%} · "
                                  f"{len(pipe.phones_)} телефонов в {pipe.k_} группах")
        self.refresh_analytics()
        self.refresh_catalog()
        self.duel.set_pipe(pipe)
        self.game.set_pipe(pipe)
        self.start_aux()

    def start_aux(self):
        """Справедливая цена и прогноз цен — фоном, после основной модели."""
        pipe = self.pipe

        def work():
            pm = PriceModel.load(phones=pipe.phones_)
            if pm is None:
                pm = PriceModel().fit(pipe.phones_)
                pm.save()
            tiers = dict(zip(pipe.phones_["id"].astype(int), pipe.labels_.astype(int)))
            hist = db.get_price_history()
            fc = PriceForecast().fit(hist, tiers) if len(hist) > 50 else None
            return pm, fc
        self.statusBar().showMessage("Считаю справедливые цены и прогноз цен…")
        task = Task(work)
        task.signals.done.connect(self.on_aux_ready)
        task.signals.error.connect(self.on_error)
        pool().start(task)

    def on_aux_ready(self, result):
        self.price_model, self.forecast = result
        self.fair = self.price_model.fair_ if self.price_model else None
        self.refresh_aux()
        self.refresh_catalog()
        if self.last_res:
            self.on_pick()
        self.statusBar().showMessage("Справедливые цены и прогноз готовы.", 4000)

    def run_experiments(self):
        if self.pipe is None:
            return
        self.btn_exp.setEnabled(False)
        task = Task(experiments.run_all, self.pipe, with_progress=True)
        task.signals.progress.connect(lambda s: self.statusBar().showMessage(f"Эксперимент: {s}"))
        task.signals.done.connect(self.on_experiments_done)
        task.signals.error.connect(self.on_error)
        pool().start(task)

    def on_experiments_done(self, res):
        self.exp = res
        self.btn_exp.setEnabled(True)
        self.statusBar().showMessage("Сравнение моделей завершено", 5000)
        self.refresh_experiments()

    def export_report(self):
        path = experiments.export_excel(self.exp, self.price_model, self.forecast)
        QMessageBox.information(self, "Экспорт", f"Все таблицы экспериментов сохранены:\n{path}")

    def on_error(self, msg: str):
        self.btn_pick.setEnabled(self.pipe is not None)
        self.btn_retrain.setEnabled(True)
        self.btn_exp.setEnabled(True)
        QMessageBox.critical(self, "Ошибка", msg[:3000])

    # ===== Данные и API =====
    def _build_data_tab(self) -> QWidget:
        w, outer = page()
        lay = QHBoxLayout()
        lay.setSpacing(14)
        outer.addLayout(lay, 1)
        keys = QGroupBox("Ключи API магазинов")
        form = QFormLayout(keys)
        form.setVerticalSpacing(10)
        self.key_edits = {}
        current = self._read_env()
        for name, title in API_KEYS:
            e = QLineEdit(current.get(name, ""))
            if any(s in name for s in ("KEY", "TOKEN", "SECRET")):
                e.setEchoMode(QLineEdit.Password)
            form.addRow(title, e)
            self.key_edits[name] = e
        form.addRow(hint("API Яндекс Маркета, Ozon, Wildberries и Avito — для продавцов: ключи выдаются в кабинете "
                         "продавца, и обновятся цены моделей, которые есть среди ваших товаров/объявлений. "
                         "Без ключей используется локальная база."))
        b_save = QPushButton("Сохранить ключи")
        b_save.clicked.connect(self.save_keys)
        b_prices = QPushButton("Обновить цены")
        b_prices.clicked.connect(self.update_prices)
        btns = QHBoxLayout()
        btns.addWidget(b_save)
        btns.addWidget(b_prices)
        form.addRow(btns)

        data = QGroupBox("База данных и модели")
        dl = QVBoxLayout(data)
        self.db_stats = QLabel()
        b_import = QPushButton("Импортировать телефоны из CSV…")
        b_import.clicked.connect(self.import_csv)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появятся сообщения об импорте и обновлении цен")
        self.registry_model = PandasModel()
        dl.addWidget(self.db_stats)
        dl.addWidget(b_import)
        dl.addWidget(hint("Свежие цены из Telegram: python tools/scrape_telegram.py, затем python tools/merge_sources.py"))
        dl.addWidget(self.log, 1)
        dl.addWidget(QLabel("Реестр обученных моделей"))
        dl.addWidget(make_table(self.registry_model), 1)
        lay.addWidget(keys, 1)
        lay.addWidget(data, 1)
        return w

    @staticmethod
    def _read_env() -> dict:
        if not ENV_PATH.exists():
            return {}
        pairs = (line.split("=", 1) for line in ENV_PATH.read_text(encoding="utf-8").splitlines()
                 if "=" in line and not line.startswith("#"))
        return {k.strip(): v.strip() for k, v in pairs}

    def save_keys(self):
        ENV_PATH.write_text("\n".join(f"{k}={e.text().strip()}" for k, e in self.key_edits.items()) + "\n",
                            encoding="utf-8")
        self.log.append("Ключи сохранены в .env")

    def update_prices(self):
        providers = [p for p in all_providers() if p.is_configured()]
        if not providers:
            self.log.append("Ключи не заполнены — цены остаются из локальной базы.")
            return
        for prov in providers:
            try:
                found = prov.match_prices(self.phones)
            except ProviderError as e:
                self.log.append(f"Не удалось: {e}")
                continue
            for phone_id, price in found.items():
                db.update_price(phone_id, price, prov.name)
            self.log.append(f"{prov.name}: обновлено цен — {len(found)}")
        self.reload_phones()
        self.refresh_catalog()
        self.refresh_db_stats()

    def import_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "CSV с телефонами", "", "CSV (*.csv)")
        if not path:
            return
        try:
            n, report = db.import_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Импорт", str(e))
            return
        problems = {k: v for k, v in report.items() if k not in ("dtypes", "missing") and v}
        missing = {k: v for k, v in report["missing"].items() if v}
        self.log.append(f"Импортировано строк: {n}. Пропуски: {missing or 'нет'}. Ошибки: {problems or 'нет'}")
        self.log.append("Нажмите «Переобучить» на вкладке «Аналитика ML», чтобы учесть новые телефоны.")
        self.reload_phones()
        self.refresh_catalog()
        self.refresh_db_stats()

    def refresh_db_stats(self):
        s = db.stats()
        archived = int(self.phones_all["archived"].fillna(0).sum()) if hasattr(self, "phones_all") else 0
        self.db_stats.setText(f"Телефонов: <b>{s['phones']}</b> (в архиве {archived}) · анкет: {s['survey_answers']} · "
                              f"отзывов: {s['feedback']} · записей истории цен: {s['price_history']}")
        try:
            reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))[::-1]
            self.registry_model.set_df(pd.DataFrame(reg))
        except Exception:
            pass
