"""Главное окно: Опрос → Результат, Каталог; в про-режиме — Аналитика ML, Игры и эксперименты, Данные и API."""
import pandas as pd
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QStackedWidget, QTabWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from src import db
from src.ml import clustering, experiments
from src.ml.classifier import PhonePipeline
from src.ml.labeling import (CLASS_NAMES, HZ_PREF, IMPORTANCE_DEFAULTS, IMPORTANCE_FIELDS, IMPORTANCE_TITLES,
                             OPTION_PREF, PRESETS, SCREEN_PREF, SIMPLE_IMPORTANCE)
from src.ml.recommender import explain, phone_row, recommend
from src.providers.base import ENV_PATH, ProviderError, all_providers
from src.ui import charts
from src.ui.widgets import (DuelWidget, GuessGame, PandasModel, PhoneCard, SandboxWidget, SimilarDialog, Task,
                            clear_layout, hint, make_table, page, pool, selected_id, split)

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
TAB_SURVEY, TAB_RESULT, TAB_CATALOG, TAB_ANALYTICS, TAB_FUN, TAB_DATA = range(6)
PRO_TABS = (TAB_ANALYTICS, TAB_FUN, TAB_DATA)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Подбор смартфона")
        self.resize(1380, 880)
        self.settings = QSettings("PhonePicker", "PhonePicker")
        self.pipe: PhonePipeline | None = None
        self.exp: dict | None = experiments.load_results()
        self.survey_id: int | None = None
        self.last_answers: dict | None = None

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_survey_tab(), "Опрос")
        self.tabs.addTab(self._build_result_tab(), "Результат")
        self.tabs.addTab(self._build_catalog_tab(), "Каталог")
        self.tabs.addTab(self._build_analytics_tab(), "Аналитика ML")
        self.tabs.addTab(self._build_fun_tab(), "Дуэль, игра, песочница")
        self.tabs.addTab(self._build_data_tab(), "Данные и API")

        self.pro = QCheckBox("Про-режим")
        self.pro.setObjectName("switch")
        self.pro.setCursor(Qt.PointingHandCursor)
        self.pro.setToolTip("Для опытных: все параметры, аналитика модели, дуэль, игра и песочница")
        self.pro.toggled.connect(self.set_mode)
        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 12, 0)
        cl.addWidget(self.pro)
        self.tabs.setCornerWidget(corner, Qt.TopRightCorner)

        db.init_db()
        # системы, которых нет в базе (например, iOS), в выпадающих списках недоступны
        present = set(db.get_phones()["os"])
        for combo in (self.os_pref, self.f_os):
            for i in range(1, combo.count()):
                combo.model().item(i).setEnabled(combo.itemText(i) in present)
        self.refresh_catalog()
        self.refresh_db_stats()
        self.pro.setChecked(self.settings.value("pro_mode", False, type=bool))
        self.set_mode(self.pro.isChecked())
        self.pipe = PhonePipeline.load()
        if self.pipe is None or len(self.pipe.phones_) != db.stats()["phones"]:
            self.retrain()
        else:
            self.on_model_ready(self.pipe, save=False)

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
            b.clicked.connect(lambda _=False, n=name: self.apply_preset(n))
            self.preset_group.addButton(b)
            presets.addWidget(b)
        presets.addStretch()
        outer.addLayout(presets)

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

        right = QGroupBox("Что для вас важнее")
        grid = QGridLayout(right)
        grid.setVerticalSpacing(12)
        grid.addWidget(hint("0 — совсем не важно, 10 — очень важно"), 0, 0, 1, 3)
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

    def answers(self) -> dict:
        a = {k: s.value() for k, s in self.sliders.items()}
        a.update(budget_rub=float(self.budget.value()), os_pref=self.os_pref.currentIndex(),
                 min_rom=int(self.min_rom.currentText()), need_nfc=int(self.need_nfc.isChecked()),
                 esim_pref=self.esim_pref.currentIndex(), ir_pref=self.ir_pref.currentIndex(),
                 screen_pref=self.screen_pref.currentIndex(), min_hz=HZ_PREF[self.min_hz.currentIndex()],
                 gaming=int(self.gaming.isChecked()), photo=int(self.photo.isChecked()),
                 work=int(self.work.isChecked()))
        return a

    def on_pick(self):
        if self.pipe is None:
            QMessageBox.information(self, "Подождите", "Модель ещё обучается.")
            return
        a = self.answers()
        res = recommend(self.pipe, a)
        self.last_answers = a
        self.survey_id = db.save_survey(a, res["pred_class"])
        table = res["table"]
        self.result_model.set_df(table.drop(columns=["_pros"]))
        self.result_view.setColumnHidden(0, True)
        best = table.iloc[0]
        self.result_header.setText(
            f"<span style='font-size:15pt; font-weight:600'>Вам подойдёт: {best['Телефон']}</span><br>"
            f"<span style='color:#7A8194'>Нейросеть отнесла запрос к классу <b>{res['class_name']}</b> "
            f"({res['proba'][res['pred_class']]:.0%}), ближайшая группа — <b>{res['cluster_name']}</b>. "
            f"Отметьте, что подошло, — это поможет дообучить модель.</span>")
        clear_layout(self.cards_layout)
        for k, (_, r) in enumerate(table.head(5).iterrows()):
            p = phone_row(self.pipe, int(r["id"]))
            self.cards_layout.addWidget(PhoneCard(p, pros=r["_pros"], badge="Лучший вариант" if k == 0 else f"№{k + 1}",
                                                  on_similar=self.open_similar, on_like=self.like_phone))
        self.cards_layout.addStretch()
        ids = table["id"].tolist()
        charts.plot_proba(self.proba_canvas, res["proba"])
        charts.plot_reco_scores(self.scores_canvas, table)
        charts.plot_reco_radar(self.radar_canvas, self.pipe, ids)
        charts.plot_clusters(self.result_cluster_canvas, self.pipe, ids)
        self.result_view.selectRow(0)
        self.show_explain()
        self.tabs.setCurrentIndex(TAB_RESULT)

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
        self.result_stack = QStackedWidget()
        lay.addWidget(self.result_stack, 1)

        # простой режим: карточки
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        holder.setObjectName("page")
        self.cards_layout = QVBoxLayout(holder)
        self.cards_layout.setSpacing(12)
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
        b_sim = QPushButton("Похожие")
        b_sim.clicked.connect(lambda: self._with_selected(self.result_view, self.result_model, self.open_similar))
        b_duel = QPushButton("В дуэль с лидером")
        b_duel.clicked.connect(self.duel_with_best)
        btns.addWidget(b_sim)
        btns.addWidget(b_duel)
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
        chart_tabs.addTab(self.result_cluster_canvas, "На карте кластеров")
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

    def open_similar(self, phone_id: int):
        if self.pipe is not None:
            SimilarDialog(self.pipe, phone_id, self).exec()

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
        for cb in (self.f_nfc, self.f_5g, self.f_esim, self.f_ir):
            checks.addWidget(cb)
        checks.addStretch()
        f.addWidget(hint("Наличие"), 4, 0)
        f.addLayout(checks, 5, 0, 1, 2)
        for wdg in (self.f_price_min, self.f_price_max, self.f_active, self.f_screen):
            wdg.valueChanged.connect(self.refresh_catalog)
        for wdg in (self.f_ram, self.f_rom, self.f_os, self.f_brand, self.f_hz):
            wdg.currentIndexChanged.connect(self.refresh_catalog)
        for wdg in (self.f_nfc, self.f_5g, self.f_esim, self.f_ir):
            wdg.toggled.connect(self.refresh_catalog)
        self.f_search.textChanged.connect(self.refresh_catalog)
        lay.addWidget(box)
        row = QHBoxLayout()
        self.catalog_count = hint("")
        row.addWidget(self.catalog_count, 1)
        b_sim = QPushButton("Открыть и показать похожие")
        b_sim.clicked.connect(lambda: self._with_selected(self.catalog_view, self.catalog_model, self.open_similar))
        row.addWidget(b_sim)
        lay.addLayout(row)
        self.catalog_model = PandasModel()
        self.catalog_view = make_table(self.catalog_model)
        self.catalog_view.doubleClicked.connect(
            lambda _: self._with_selected(self.catalog_view, self.catalog_model, self.open_similar))
        lay.addWidget(self.catalog_view, 1)
        lay.addWidget(hint("Двойной клик по строке — карточка телефона с подборкой похожих."))
        return w

    def refresh_catalog(self, *_):
        df = db.get_phones()
        if self.f_brand.count() == 0:
            self.f_brand.blockSignals(True)
            self.f_brand.addItems(["любой"] + sorted(df["brand"].unique()))
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
        text = self.f_search.text().strip().lower()
        if text:
            m &= (df["brand"] + " " + df["model"]).str.lower().str.contains(text, regex=False)
        sel = df[m]
        view = sel[list(CATALOG_COLUMNS)].rename(columns=CATALOG_COLUMNS)
        if self.pipe is not None:
            pos = pd.Series(range(len(self.pipe.phones_)), index=self.pipe.phones_["id"]).reindex(sel["id"])
            view["Класс"] = [CLASS_NAMES[self.pipe.labels_[int(p)]] if pd.notna(p) else "" for p in pos]
            view["Кластер"] = [self.pipe.cluster_names_[int(self.pipe.clusters_[int(p)])] if pd.notna(p) else ""
                               for p in pos]
        self.catalog_model.set_df(view)
        self.catalog_view.setColumnHidden(0, True)
        self.catalog_count.setText(f"Показано {len(view)} из {len(df)} моделей")

    # ===== Аналитика =====
    def _build_analytics_tab(self) -> QWidget:
        w, lay = page()
        top = QHBoxLayout()
        self.btn_retrain = QPushButton("Переобучить с учётом отзывов")
        self.btn_retrain.clicked.connect(self.retrain)
        self.btn_exp = QPushButton("Сравнить модели (1–2 мин)")
        self.btn_exp.clicked.connect(self.run_experiments)
        self.metrics_label = hint("")
        top.addWidget(self.btn_retrain)
        top.addWidget(self.btn_exp)
        top.addSpacing(10)
        top.addWidget(self.metrics_label, 1)
        lay.addLayout(top)

        C = charts.MplCanvas
        self.c_clusters, self.c_radar, self.c_pricebox = C(9, 5), C(6, 4.5), C(6, 4)
        self.c_priceperf, self.c_classes, self.c_crossmx = C(10, 5), C(6, 3.5), C(6, 3.5)
        self.c_elbow, self.c_sil, self.c_dendro = C(9, 3.5), C(9, 4), C(10, 5)
        self.c_corr, self.c_weights, self.c_train, self.c_ablation = C(7, 6), C(5, 5), C(10, 4), C(9, 4)
        self.profile_model = PandasModel()
        self.algo_model, self.weights_exp_model = PandasModel(), PandasModel()
        self.ablation_model, self.weights_model = PandasModel(), PandasModel()

        sub = QTabWidget()
        sub.addTab(split(Qt.Vertical, self.c_clusters, make_table(self.profile_model), sizes=[520, 200]),
                   "Карта кластеров")
        sub.addTab(split(Qt.Horizontal, self.c_radar, self.c_pricebox), "Профили кластеров")
        sub.addTab(self.c_priceperf, "Цена и мощность")
        sub.addTab(split(Qt.Horizontal, self.c_classes, self.c_crossmx), "Классы")
        sub.addTab(split(Qt.Vertical, self.c_elbow, self.c_sil), "Выбор k")
        sub.addTab(self.c_dendro, "Дендрограмма")
        sub.addTab(split(Qt.Horizontal, self.c_corr, self.c_weights,
                         split(Qt.Vertical, make_table(self.weights_model), make_table(self.weights_exp_model),
                               make_table(self.algo_model)), sizes=[600, 400, 400]),
                   "Признаки и веса")
        sub.addTab(self.c_train, "Нейросеть")
        sub.addTab(split(Qt.Vertical, self.c_ablation, make_table(self.ablation_model), sizes=[420, 260]),
                   "Сравнение моделей")
        lay.addWidget(sub, 1)
        return w

    def refresh_analytics(self):
        p = self.pipe
        charts.plot_clusters(self.c_clusters, p)
        self.profile_model.set_df(clustering.cluster_profile_table(p.phones_, p.clusters_, p.cluster_names_)
                                  .reset_index())
        charts.plot_cluster_radar(self.c_radar, p)
        charts.plot_price_box(self.c_pricebox, p)
        charts.plot_price_perf(self.c_priceperf, p)
        charts.plot_classes(self.c_classes, p)
        charts.plot_class_cluster_matrix(self.c_crossmx, p)
        charts.plot_elbow_silhouette(self.c_elbow, clustering.elbow(p.x_), clustering.silhouette_by_k(p.x_), p.k_)
        charts.plot_silhouette(self.c_sil, clustering.silhouette_values(p.x_, p.clusters_), p.clusters_,
                               p.cluster_names_)
        charts.plot_dendrogram(self.c_dendro, clustering.hierarchical_linkage(p.x_), p.phones_["model"].tolist())
        charts.plot_corr(self.c_corr, p)
        wt = p.space_.weights_table()
        self.weights_model.set_df(wt)
        charts.plot_weights(self.c_weights, wt)
        charts.plot_training(self.c_train, p)
        t = p.metrics_["test"]
        self.metrics_label.setText(
            f"Нейросеть на тесте: accuracy {t['accuracy']:.3f} · F1 {t['f1_macro']:.3f} · "
            f"ROC AUC {t['roc_auc_ovr']:.3f} · кластеров: {p.k_} · отзывов учтено: {p.n_feedback_}")
        self.refresh_experiments()

    def refresh_experiments(self):
        if not self.exp:
            return
        charts.plot_ablation(self.c_ablation, self.exp["ablation"])
        self.ablation_model.set_df(self.exp["ablation"].round(4))
        self.algo_model.set_df(self.exp["algorithms"].round(4))
        self.weights_exp_model.set_df(self.exp["weights"].round(4))

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
        task = Task(lambda: (params, PhonePipeline(**params).fit(db.get_phones(), db.get_liked_feedback())))
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
        self.model_status.setText("Обучаем модель: метки → кластеризация → нейросеть…")
        task = Task(lambda: PhonePipeline().fit(db.get_phones(), db.get_liked_feedback()))
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

        data = QGroupBox("База данных")
        dl = QVBoxLayout(data)
        self.db_stats = QLabel()
        b_import = QPushButton("Импортировать телефоны из CSV…")
        b_import.clicked.connect(self.import_csv)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Здесь появятся сообщения об импорте и обновлении цен")
        dl.addWidget(self.db_stats)
        dl.addWidget(b_import)
        dl.addWidget(self.log, 1)
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
        phones = db.get_phones()
        providers = [p for p in all_providers() if p.is_configured()]
        if not providers:
            self.log.append("Ключи не заполнены — цены остаются из локальной базы.")
            return
        for prov in providers:
            try:
                found = prov.match_prices(phones)
            except ProviderError as e:
                self.log.append(f"Не удалось: {e}")
                continue
            for phone_id, price in found.items():
                db.update_price(phone_id, price, prov.name)
            self.log.append(f"{prov.name}: обновлено цен — {len(found)}")
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
        self.refresh_catalog()
        self.refresh_db_stats()

    def refresh_db_stats(self):
        s = db.stats()
        self.db_stats.setText(f"Телефонов: <b>{s['phones']}</b> · анкет: {s['survey_answers']} · "
                              f"отзывов: {s['feedback']} · изменений цен: {s['price_history']}")
