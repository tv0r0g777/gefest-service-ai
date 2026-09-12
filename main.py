import csv
import math
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

APP_TITLE = "GEFEST Service AI — прогноз длительности ремонта"
RANDOM_STATE = 42
SAMPLE_COUNT = 720

BG = "#F5F8FC"
SURFACE = "#FFFFFF"
SURFACE_2 = "#EFF4F9"
BORDER = "#DCE5EF"
TEXT = "#183247"
MUTED = "#6F879B"
TEAL = "#159A92"
TEAL_DARK = "#0E7D77"
TEAL_SOFT = "#E3F6F3"
BLUE_SOFT = "#EAF2FF"
PURPLE_SOFT = "#F2EBFF"
GREEN_SOFT = "#E8F7EF"
ORANGE_SOFT = "#FFF2E1"
GREEN = "#1C9A69"
YELLOW = "#D6921B"
RED = "#D65B56"
ORANGE = "#F0A04B"
WHITE = "#FFFFFF"


@dataclass
class PredictionRecord:
    timestamp: str
    age: float
    repairs: int
    complexity: int
    parts_available: int
    priority: int
    predicted_hours: float
    reserve_hours: float
    category: str


def build_dataset(n=SAMPLE_COUNT, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)

    age_years = rng.uniform(0.5, 12.0, n)
    previous_repairs = rng.integers(0, 9, n)
    complexity = rng.integers(1, 11, n)
    parts_available = rng.integers(0, 2, n)
    priority = rng.integers(1, 4, n)
    noise = rng.normal(0, 1.6, n)

    repair_hours = (
        2.2
        + 0.42 * age_years
        + 1.10 * previous_repairs
        + 1.95 * complexity
        + (1 - parts_available) * 5.2
        - priority * 0.8
        + 0.07 * complexity * previous_repairs
        + noise
    )

    repair_hours = np.clip(repair_hours, 0.5, None)

    X = np.column_stack(
        [
            age_years,
            previous_repairs,
            complexity,
            parts_available,
            priority,
        ]
    )

    return X, repair_hours


class RepairPredictor:
    def __init__(self):
        self.X, self.y = build_dataset()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

        X_train_scaled = self.x_scaler.fit_transform(self.X_train)
        X_test_scaled = self.x_scaler.transform(self.X_test)

        y_train_scaled = self.y_scaler.fit_transform(
            self.y_train.reshape(-1, 1)
        ).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=(18, 10),
            activation="relu",
            solver="adam",
            learning_rate_init=0.0025,
            max_iter=4000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=RANDOM_STATE,
        )

        self.model.fit(
            X_train_scaled,
            y_train_scaled,
        )

        test_scaled = self.model.predict(
            X_test_scaled
        )

        self.y_pred = self.y_scaler.inverse_transform(
            test_scaled.reshape(-1, 1)
        ).ravel()

        self.mae = mean_absolute_error(
            self.y_test,
            self.y_pred,
        )

        self.rmse = math.sqrt(
            mean_squared_error(
                self.y_test,
                self.y_pred,
            )
        )

        self.r2 = r2_score(
            self.y_test,
            self.y_pred,
        )

    def predict(
        self,
        age,
        repairs,
        complexity,
        parts_available,
        priority,
    ):
        row = np.array(
            [
                [
                    age,
                    repairs,
                    complexity,
                    parts_available,
                    priority,
                ]
            ],
            dtype=float,
        )

        scaled = self.x_scaler.transform(
            row
        )

        prediction_scaled = self.model.predict(
            scaled
        )

        prediction = self.y_scaler.inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )[0, 0]

        return max(
            0.5,
            float(prediction),
        )


class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(
            parent,
            bg=BG,
        )

        self.canvas = tk.Canvas(
            self,
            bg=BG,
            highlightthickness=0,
        )

        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.inner = tk.Frame(
            self.canvas,
            bg=BG,
        )

        self.window_id = self.canvas.create_window(
            (0, 0),
            window=self.inner,
            anchor="nw",
        )

        self.inner.bind(
            "<Configure>",
            lambda event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )

        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(
                self.window_id,
                width=event.width,
            ),
        )

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True,
        )

        self.scrollbar.pack(
            side="right",
            fill="y",
        )

        self.canvas.bind_all(
            "<MouseWheel>",
            self._mousewheel,
        )

    def _mousewheel(self, event):
        try:
            self.canvas.yview_scroll(
                int(-event.delta / 120),
                "units",
            )
        except tk.TclError:
            pass


class App:
    def __init__(self, root):
        self.root = root

        self.root.title(
            APP_TITLE
        )

        self.root.geometry(
            "1380x870"
        )

        self.root.minsize(
            1180,
            740,
        )

        self.root.configure(
            bg=BG
        )

        self.predictor = RepairPredictor()

        self.history = []

        self.setup_styles()

        self.build_ui()

    def setup_styles(self):
        style = ttk.Style()

        style.theme_use(
            "clam"
        )

        style.configure(
            "TNotebook",
            background=BG,
            borderwidth=0,
        )

        style.configure(
            "TNotebook.Tab",
            background=SURFACE_2,
            foreground=MUTED,
            padding=(20, 11),
            font=("Segoe UI", 10),
            borderwidth=0,
        )

        style.map(
            "TNotebook.Tab",
            background=[
                ("selected", SURFACE),
                ("active", "#E7EEF5"),
            ],
            foreground=[
                ("selected", TEAL_DARK),
                ("active", TEXT),
            ],
        )

        style.configure(
            "TEntry",
            fieldbackground=SURFACE,
            foreground=TEXT,
            padding=8,
        )

        style.configure(
            "TCombobox",
            fieldbackground=SURFACE,
            background=SURFACE,
            foreground=TEXT,
            arrowcolor=MUTED,
            padding=7,
        )

        style.map(
            "TCombobox",
            fieldbackground=[
                ("readonly", SURFACE)
            ],
            foreground=[
                ("readonly", TEXT)
            ],
            selectbackground=[
                ("readonly", SURFACE)
            ],
            selectforeground=[
                ("readonly", TEXT)
            ],
        )

        style.configure(
            "Treeview",
            background=SURFACE,
            foreground=TEXT,
            fieldbackground=SURFACE,
            rowheight=31,
            font=("Segoe UI", 9),
            borderwidth=0,
        )

        style.configure(
            "Treeview.Heading",
            background=SURFACE_2,
            foreground=TEXT,
            font=("Segoe UI Semibold", 9),
            padding=8,
            borderwidth=0,
        )

        style.map(
            "Treeview",
            background=[
                ("selected", "#D7EFED")
            ],
            foreground=[
                ("selected", TEXT)
            ],
        )

    def build_ui(self):
        main = tk.Frame(
            self.root,
            bg=BG,
        )

        main.pack(
            fill="both",
            expand=True,
        )

        self.build_header(
            main
        )

        self.build_kpis(
            main
        )

        self.build_tabs(
            main
        )

    def build_header(self, parent):
        header = tk.Frame(
            parent,
            bg=BG,
        )

        header.pack(
            fill="x",
            padx=24,
            pady=(18, 10),
        )

        left = tk.Frame(
            header,
            bg=BG,
        )

        left.pack(
            side="left",
            fill="x",
            expand=True,
        )

        tk.Label(
            left,
            text="GEFEST SERVICE AI",
            bg=BG,
            fg=TEAL_DARK,
            font=("Segoe UI Black", 12),
        ).pack(
            anchor="w"
        )

        tk.Label(
            left,
            text="Прогноз длительности ремонта оборудования",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 24, "bold"),
        ).pack(
            anchor="w",
            pady=(4, 0),
        )

        tk.Label(
            left,
            text=(
                "Учебная система на базе нейронной сети MLP "
                "для оценки трудоёмкости ремонтной заявки"
            ),
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w",
            pady=(5, 0),
        )

        status = tk.Frame(
            header,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
        )

        status.pack(
            side="right",
            padx=(18, 0),
            pady=10,
        )

        tk.Label(
            status,
            text="●  Модель обучена",
            bg=SURFACE,
            fg=GREEN,
            font=("Segoe UI Semibold", 10),
            padx=16,
            pady=11,
        ).pack()

    def build_kpis(self, parent):
        row = tk.Frame(
            parent,
            bg=BG,
        )

        row.pack(
            fill="x",
            padx=24,
            pady=(0, 13),
        )

        cards = [
            (
                "Обучающая выборка",
                str(len(self.predictor.X_train)),
                "наблюдений",
                BLUE_SOFT,
            ),
            (
                "Тестовая выборка",
                str(len(self.predictor.X_test)),
                "наблюдений",
                PURPLE_SOFT,
            ),
            (
                "Средняя ошибка",
                f"{self.predictor.mae:.2f}",
                "MAE, часов",
                GREEN_SOFT,
            ),
            (
                "Качество модели",
                f"{self.predictor.r2:.3f}",
                "коэффициент R²",
                ORANGE_SOFT,
            ),
        ]

        for index, card_data in enumerate(cards):
            card = tk.Frame(
                row,
                bg=card_data[3],
                height=105,
            )

            card.pack(
                side="left",
                fill="x",
                expand=True,
                padx=(
                    0 if index == 0 else 8,
                    0 if index == len(cards) - 1 else 8,
                ),
            )

            card.pack_propagate(
                False
            )

            tk.Label(
                card,
                text=card_data[0],
                bg=card_data[3],
                fg=TEXT,
                font=("Segoe UI Semibold", 10),
            ).pack(
                anchor="w",
                padx=18,
                pady=(14, 2),
            )

            tk.Label(
                card,
                text=card_data[1],
                bg=card_data[3],
                fg=TEXT,
                font=("Segoe UI", 22, "bold"),
            ).pack(
                anchor="w",
                padx=18,
            )

            tk.Label(
                card,
                text=card_data[2],
                bg=card_data[3],
                fg=MUTED,
                font=("Segoe UI", 9),
            ).pack(
                anchor="w",
                padx=18,
                pady=(0, 10),
            )

    def build_tabs(self, parent):
        self.notebook = ttk.Notebook(
            parent
        )

        self.notebook.pack(
            fill="both",
            expand=True,
            padx=24,
            pady=(0, 20),
        )

        self.build_forecast_tab()
        self.build_data_tab()
        self.build_logic_tab()
        self.build_model_tab()
        self.build_history_tab()

    def make_card(
        self,
        parent,
        title=None,
        subtitle=None,
    ):
        card = tk.Frame(
            parent,
            bg=SURFACE,
            bd=0,
            highlightbackground=BORDER,
            highlightthickness=1,
        )

        if title:
            tk.Label(
                card,
                text=title,
                bg=SURFACE,
                fg=TEXT,
                font=("Segoe UI", 15, "bold"),
            ).pack(
                anchor="w",
                padx=22,
                pady=(
                    18,
                    3 if subtitle else 15,
                ),
            )

        if subtitle:
            tk.Label(
                card,
                text=subtitle,
                bg=SURFACE,
                fg=MUTED,
                font=("Segoe UI", 9),
            ).pack(
                anchor="w",
                padx=22,
                pady=(0, 14),
            )

        return card

    def build_forecast_tab(self):
        tab = tk.Frame(
            self.notebook,
            bg=BG,
        )

        self.notebook.add(
            tab,
            text="  Прогноз  ",
        )

        scroll = ScrollFrame(
            tab
        )

        scroll.pack(
            fill="both",
            expand=True,
            pady=8,
        )

        content = scroll.inner

        form = self.make_card(
            content,
            "Параметры заявки",
            "Все исходные значения собраны в одной компактной панели",
        )

        form.pack(
            fill="x",
            pady=(0, 12),
        )

        grid = tk.Frame(
            form,
            bg=SURFACE,
        )

        grid.pack(
            fill="x",
            padx=22,
            pady=(0, 12),
        )

        self.age_entry = self.make_entry(
            grid,
            "Возраст оборудования",
            "лет",
            0,
            0,
        )

        self.repairs_entry = self.make_entry(
            grid,
            "Предыдущие ремонты",
            "раз",
            0,
            1,
        )

        self.complexity_entry = self.make_entry(
            grid,
            "Сложность неисправности",
            "1–10",
            0,
            2,
        )

        self.parts_combo = self.make_combo(
            grid,
            "Наличие запчастей",
            [
                "Запчасти есть",
                "Запчастей нет",
            ],
            1,
            0,
            0,
        )

        self.priority_combo = self.make_combo(
            grid,
            "Приоритет заявки",
            [
                "Низкий",
                "Средний",
                "Высокий",
            ],
            1,
            1,
            1,
        )

        action = tk.Frame(
            grid,
            bg=SURFACE,
        )

        action.grid(
            row=1,
            column=2,
            sticky="nsew",
            padx=(12, 0),
            pady=(12, 4),
        )

        tk.Label(
            action,
            text="Действия",
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w",
            pady=(0, 6),
        )

        action_row = tk.Frame(
            action,
            bg=SURFACE,
        )

        action_row.pack(
            fill="x"
        )

        tk.Button(
            action_row,
            text="Рассчитать прогноз",
            command=self.run_prediction,
            bg=TEAL,
            fg=WHITE,
            activebackground=TEAL_DARK,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            padx=18,
            pady=10,
            cursor="hand2",
            font=("Segoe UI Semibold", 10),
        ).pack(
            side="left",
            fill="x",
            expand=True,
        )

        tk.Button(
            action_row,
            text="Пример",
            command=self.fill_demo,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground="#E1E9F1",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=16,
            pady=10,
            cursor="hand2",
            font=("Segoe UI", 10),
        ).pack(
            side="left",
            padx=(8, 0),
        )

        result = self.make_card(
            content,
            "Результат прогнозирования",
            "Оценка ожидаемой длительности ремонта по данным нейросети",
        )

        result.pack(
            fill="x",
            pady=(0, 12),
        )

        result_row = tk.Frame(
            result,
            bg=SURFACE,
        )

        result_row.pack(
            fill="x",
            padx=22,
            pady=(0, 18),
        )

        forecast_box = tk.Frame(
            result_row,
            bg=BLUE_SOFT,
            height=145,
        )

        forecast_box.pack(
            side="left",
            fill="both",
            expand=True,
        )

        forecast_box.pack_propagate(
            False
        )

        tk.Label(
            forecast_box,
            text="ПРОГНОЗИРУЕМОЕ ВРЕМЯ",
            bg=BLUE_SOFT,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(
            anchor="w",
            padx=18,
            pady=(18, 3),
        )

        forecast_value_row = tk.Frame(
            forecast_box,
            bg=BLUE_SOFT,
        )

        forecast_value_row.pack(
            anchor="w",
            padx=18,
        )

        self.prediction_value = tk.Label(
            forecast_value_row,
            text="—",
            bg=BLUE_SOFT,
            fg=TEXT,
            font=("Segoe UI", 34, "bold"),
        )

        self.prediction_value.pack(
            side="left"
        )

        tk.Label(
            forecast_value_row,
            text=" часов",
            bg=BLUE_SOFT,
            fg=TEXT,
            font=("Segoe UI", 13),
        ).pack(
            side="left",
            pady=(15, 0),
        )

        self.category_label = tk.Label(
            forecast_box,
            text="Ожидает расчёта",
            bg=BLUE_SOFT,
            fg=MUTED,
            font=("Segoe UI", 9),
        )

        self.category_label.pack(
            anchor="w",
            padx=18,
            pady=(2, 0),
        )

        reserve_box = tk.Frame(
            result_row,
            bg=GREEN_SOFT,
            width=245,
            height=145,
        )

        reserve_box.pack(
            side="left",
            fill="y",
            padx=(12, 0),
        )

        reserve_box.pack_propagate(
            False
        )

        tk.Label(
            reserve_box,
            text="РЕЗЕРВ ВРЕМЕНИ",
            bg=GREEN_SOFT,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(
            anchor="w",
            padx=18,
            pady=(18, 5),
        )

        self.reserve_value = tk.Label(
            reserve_box,
            text="—",
            bg=GREEN_SOFT,
            fg=TEXT,
            font=("Segoe UI", 24, "bold"),
        )

        self.reserve_value.pack(
            anchor="w",
            padx=18,
        )

        tk.Label(
            reserve_box,
            text="плановый запас +15%",
            bg=GREEN_SOFT,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            padx=18,
            pady=(3, 0),
        )

        recommendation_box = tk.Frame(
            result_row,
            bg=ORANGE_SOFT,
            width=340,
            height=145,
        )

        recommendation_box.pack(
            side="left",
            fill="y",
            padx=(12, 0),
        )

        recommendation_box.pack_propagate(
            False
        )

        tk.Label(
            recommendation_box,
            text="РЕКОМЕНДАЦИЯ",
            bg=ORANGE_SOFT,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(
            anchor="w",
            padx=18,
            pady=(18, 5),
        )

        self.recommendation_label = tk.Label(
            recommendation_box,
            text="После расчёта здесь появится краткая рекомендация.",
            bg=ORANGE_SOFT,
            fg=TEXT,
            wraplength=290,
            justify="left",
            font=("Segoe UI", 9),
        )

        self.recommendation_label.pack(
            anchor="w",
            padx=18,
        )

        bottom = tk.Frame(
            content,
            bg=BG,
        )

        bottom.pack(
            fill="x"
        )

        last = self.make_card(
            bottom,
            "Последний расчёт",
            "Параметры текущего прогноза",
        )

        last.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 6),
        )

        self.summary_text = tk.Label(
            last,
            text=(
                "Возраст: —\n"
                "Ремонты: —\n"
                "Сложность: —\n"
                "Запчасти: —\n"
                "Приоритет: —"
            ),
            bg=SURFACE,
            fg=MUTED,
            justify="left",
            anchor="nw",
            font=("Segoe UI", 10),
        )

        self.summary_text.pack(
            anchor="w",
            padx=22,
            pady=(0, 18),
        )

        quality = self.make_card(
            bottom,
            "Модель",
            "Краткие показатели качества",
        )

        quality.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(6, 0),
        )

        metric_grid = tk.Frame(
            quality,
            bg=SURFACE,
        )

        metric_grid.pack(
            fill="x",
            padx=22,
            pady=(0, 12),
        )

        self.make_small_metric(
            metric_grid,
            "MAE",
            f"{self.predictor.mae:.2f} ч",
            0,
        )

        self.make_small_metric(
            metric_grid,
            "RMSE",
            f"{self.predictor.rmse:.2f} ч",
            1,
        )

        self.make_small_metric(
            metric_grid,
            "R²",
            f"{self.predictor.r2:.3f}",
            2,
        )

        tk.Label(
            quality,
            text=(
                "Подробная схема работы и график обучения "
                "вынесены в отдельные вкладки «Логика» и «Модель»."
            ),
            bg=SURFACE,
            fg=MUTED,
            wraplength=520,
            justify="left",
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            padx=22,
            pady=(0, 18),
        )

    def make_entry(
        self,
        parent,
        title,
        unit,
        row,
        column,
    ):
        box = tk.Frame(
            parent,
            bg=SURFACE,
        )

        box.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=(
                0 if column == 0 else 12,
                0,
            ),
            pady=(
                0 if row == 0 else 12,
                4,
            ),
        )

        parent.grid_columnconfigure(
            column,
            weight=1,
        )

        tk.Label(
            box,
            text=title,
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w"
        )

        input_row = tk.Frame(
            box,
            bg=SURFACE,
        )

        input_row.pack(
            fill="x",
            pady=(6, 0),
        )

        entry = ttk.Entry(
            input_row,
            font=("Segoe UI", 10),
        )

        entry.pack(
            side="left",
            fill="x",
            expand=True,
        )

        tk.Label(
            input_row,
            text=unit,
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(
            side="right",
            padx=(8, 0),
        )

        return entry

    def make_combo(
        self,
        parent,
        title,
        values,
        row,
        column,
        initial,
    ):
        box = tk.Frame(
            parent,
            bg=SURFACE,
        )

        box.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=(
                0 if column == 0 else 12,
                0,
            ),
            pady=(
                0 if row == 0 else 12,
                4,
            ),
        )

        parent.grid_columnconfigure(
            column,
            weight=1,
        )

        tk.Label(
            box,
            text=title,
            bg=SURFACE,
            fg=TEXT,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w"
        )

        combo = ttk.Combobox(
            box,
            state="readonly",
            values=values,
            font=("Segoe UI", 10),
        )

        combo.current(
            initial
        )

        combo.pack(
            fill="x",
            pady=(6, 0),
        )

        return combo

    def make_small_metric(
        self,
        parent,
        title,
        value,
        column,
    ):
        box = tk.Frame(
            parent,
            bg=SURFACE_2,
        )

        box.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(
                0 if column == 0 else 4,
                0 if column == 2 else 4,
            ),
        )

        parent.grid_columnconfigure(
            column,
            weight=1,
        )

        tk.Label(
            box,
            text=title,
            bg=SURFACE_2,
            fg=MUTED,
            font=("Segoe UI Semibold", 8),
        ).pack(
            pady=(9, 2),
        )

        tk.Label(
            box,
            text=value,
            bg=SURFACE_2,
            fg=TEXT,
            font=("Segoe UI", 12, "bold"),
        ).pack(
            pady=(0, 9),
        )

    def build_data_tab(self):
        tab = tk.Frame(
            self.notebook,
            bg=BG,
        )

        self.notebook.add(
            tab,
            text="  Данные  ",
        )

        scroll = ScrollFrame(
            tab
        )

        scroll.pack(
            fill="both",
            expand=True,
            pady=8,
        )

        content = scroll.inner

        intro = self.make_card(
            content,
            "Демонстрационная обучающая выборка",
            (
                "Данные сформированы программно и не содержат "
                "конфиденциальной информации предприятия"
            ),
        )

        intro.pack(
            fill="x",
            pady=(0, 12),
        )

        facts = tk.Frame(
            intro,
            bg=SURFACE,
        )

        facts.pack(
            fill="x",
            padx=22,
            pady=(0, 18),
        )

        fact_data = [
            (
                "720",
                "наблюдений",
                BLUE_SOFT,
            ),
            (
                "5",
                "входных признаков",
                GREEN_SOFT,
            ),
            (
                "1",
                "целевой показатель",
                ORANGE_SOFT,
            ),
        ]

        for index, item in enumerate(fact_data):
            box = tk.Frame(
                facts,
                bg=item[2],
            )

            box.pack(
                side="left",
                fill="x",
                expand=True,
                padx=(
                    0 if index == 0 else 5,
                    0 if index == 2 else 5,
                ),
            )

            tk.Label(
                box,
                text=item[0],
                bg=item[2],
                fg=TEXT,
                font=("Segoe UI", 18, "bold"),
            ).pack(
                pady=(9, 0),
            )

            tk.Label(
                box,
                text=item[1],
                bg=item[2],
                fg=MUTED,
                font=("Segoe UI", 9),
            ).pack(
                pady=(0, 9),
            )

        table = self.make_card(
            content,
            "Фрагмент набора данных",
            "Первые 45 строк демонстрационной выборки",
        )

        table.pack(
            fill="both",
            expand=True,
            pady=(0, 12),
        )

        holder = tk.Frame(
            table,
            bg=SURFACE,
        )

        holder.pack(
            fill="both",
            expand=True,
            padx=22,
            pady=(0, 18),
        )

        columns = (
            "age",
            "repairs",
            "complexity",
            "parts",
            "priority",
            "hours",
        )

        self.data_tree = ttk.Treeview(
            holder,
            columns=columns,
            show="headings",
            height=14,
        )

        headings = {
            "age": "Возраст, лет",
            "repairs": "Ремонты",
            "complexity": "Сложность",
            "parts": "Запчасти",
            "priority": "Приоритет",
            "hours": "Время, ч",
        }

        for column in columns:
            self.data_tree.heading(
                column,
                text=headings[column],
            )

            self.data_tree.column(
                column,
                width=145,
                anchor="center",
            )

        scrollbar = ttk.Scrollbar(
            holder,
            orient="vertical",
            command=self.data_tree.yview,
        )

        self.data_tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.data_tree.pack(
            side="left",
            fill="both",
            expand=True,
        )

        scrollbar.pack(
            side="right",
            fill="y",
        )

        for index in range(
            min(
                45,
                len(self.predictor.X),
            )
        ):
            row = self.predictor.X[
                index
            ]

            priority_text = {
                1: "Низкий",
                2: "Средний",
                3: "Высокий",
            }[
                int(row[4])
            ]

            self.data_tree.insert(
                "",
                "end",
                values=(
                    f"{row[0]:.1f}",
                    int(row[1]),
                    int(row[2]),
                    (
                        "Есть"
                        if int(row[3]) == 1
                        else "Нет"
                    ),
                    priority_text,
                    f"{self.predictor.y[index]:.1f}",
                ),
            )

        descriptions = self.make_card(
            content,
            "Описание признаков",
        )

        descriptions.pack(
            fill="x"
        )

        items = [
            (
                "Возраст оборудования",
                (
                    "Отражает общий срок эксплуатации и косвенно "
                    "характеризует степень износа устройства."
                ),
            ),
            (
                "Предыдущие ремонты",
                (
                    "Показывают историю обслуживания и возможную "
                    "склонность техники к повторным неисправностям."
                ),
            ),
            (
                "Сложность неисправности",
                (
                    "Оценка от 1 до 10, характеризующая предполагаемую "
                    "трудоёмкость диагностики и восстановления."
                ),
            ),
            (
                "Наличие запчастей",
                (
                    "Позволяет учитывать дополнительное время, "
                    "если необходимой комплектующей нет в наличии."
                ),
            ),
            (
                "Приоритет заявки",
                (
                    "Характеризует срочность обработки обращения "
                    "и организационную приоритетность ремонта."
                ),
            ),
        ]

        for title, description in items:
            row = tk.Frame(
                descriptions,
                bg=SURFACE,
            )

            row.pack(
                fill="x",
                padx=22,
                pady=6,
            )

            tk.Label(
                row,
                text="●",
                bg=SURFACE,
                fg=TEAL,
                font=("Segoe UI", 9),
            ).pack(
                side="left",
                anchor="n",
                padx=(0, 8),
            )

            body = tk.Frame(
                row,
                bg=SURFACE,
            )

            body.pack(
                side="left",
                fill="x",
                expand=True,
            )

            tk.Label(
                body,
                text=title,
                bg=SURFACE,
                fg=TEXT,
                font=("Segoe UI Semibold", 10),
            ).pack(
                anchor="w"
            )

            tk.Label(
                body,
                text=description,
                bg=SURFACE,
                fg=MUTED,
                wraplength=1050,
                justify="left",
                font=("Segoe UI", 9),
            ).pack(
                anchor="w"
            )

        tk.Frame(
            descriptions,
            bg=SURFACE,
            height=10,
        ).pack()

    def build_logic_tab(self):
        tab = tk.Frame(
            self.notebook,
            bg=BG,
        )

        self.notebook.add(
            tab,
            text="  Логика  ",
        )

        scroll = ScrollFrame(
            tab
        )

        scroll.pack(
            fill="both",
            expand=True,
            pady=8,
        )

        content = scroll.inner

        process = self.make_card(
            content,
            "Как формируется прогноз",
            "Последовательность обработки одной ремонтной заявки",
        )

        process.pack(
            fill="x",
            pady=(0, 12),
        )

        steps_holder = tk.Frame(
            process,
            bg=SURFACE,
        )

        steps_holder.pack(
            fill="x",
            padx=22,
            pady=(0, 20),
        )

        steps = [
            (
                "01",
                "Исходные данные",
                (
                    "Система получает пять параметров "
                    "оборудования и заявки."
                ),
                BLUE_SOFT,
            ),
            (
                "02",
                "Стандартизация",
                (
                    "StandardScaler приводит числовые значения "
                    "к сопоставимому масштабу."
                ),
                PURPLE_SOFT,
            ),
            (
                "03",
                "Нейросеть MLP",
                (
                    "Архитектура 5 → 18 → 10 → 1 формирует "
                    "ожидаемую длительность ремонта."
                ),
                GREEN_SOFT,
            ),
            (
                "04",
                "Результат",
                (
                    "Интерфейс показывает прогноз, резерв времени "
                    "и сохраняет расчёт в истории."
                ),
                ORANGE_SOFT,
            ),
        ]

        for index, item in enumerate(steps):
            box = tk.Frame(
                steps_holder,
                bg=item[3],
                height=170,
            )

            box.pack(
                side="left",
                fill="both",
                expand=True,
                padx=(
                    0 if index == 0 else 6,
                    0 if index == 3 else 6,
                ),
            )

            box.pack_propagate(
                False
            )

            tk.Label(
                box,
                text=item[0],
                bg=TEAL,
                fg=WHITE,
                font=("Segoe UI", 10, "bold"),
                padx=8,
                pady=5,
            ).pack(
                anchor="w",
                padx=16,
                pady=(15, 10),
            )

            tk.Label(
                box,
                text=item[1],
                bg=item[3],
                fg=TEXT,
                font=("Segoe UI", 11, "bold"),
            ).pack(
                anchor="w",
                padx=16,
            )

            tk.Label(
                box,
                text=item[2],
                bg=item[3],
                fg=MUTED,
                justify="left",
                wraplength=245,
                font=("Segoe UI", 9),
            ).pack(
                anchor="w",
                padx=16,
                pady=(8, 10),
            )

        architecture = self.make_card(
            content,
            "Архитектура нейронной сети",
            (
                "Пять входных признаков, два скрытых слоя "
                "и одно значение на выходе"
            ),
        )

        architecture.pack(
            fill="x",
            pady=(0, 12),
        )

        chain = tk.Frame(
            architecture,
            bg=SURFACE,
        )

        chain.pack(
            pady=(4, 20),
        )

        nodes = [
            (
                "Вход",
                "5",
                "признаков",
            ),
            (
                "Скрытый слой",
                "18",
                "нейронов",
            ),
            (
                "Скрытый слой",
                "10",
                "нейронов",
            ),
            (
                "Выход",
                "1",
                "значение",
            ),
        ]

        for index, node in enumerate(nodes):
            box = tk.Frame(
                chain,
                bg=TEAL_SOFT,
                width=165,
                height=105,
            )

            box.pack(
                side="left",
                padx=7,
            )

            box.pack_propagate(
                False
            )

            tk.Label(
                box,
                text=node[0],
                bg=TEAL_SOFT,
                fg=TEAL_DARK,
                font=("Segoe UI Semibold", 9),
            ).pack(
                pady=(12, 0),
            )

            tk.Label(
                box,
                text=node[1],
                bg=TEAL_SOFT,
                fg=TEAL_DARK,
                font=("Segoe UI", 24, "bold"),
            ).pack()

            tk.Label(
                box,
                text=node[2],
                bg=TEAL_SOFT,
                fg=MUTED,
                font=("Segoe UI", 8),
            ).pack()

            if index < len(nodes) - 1:
                tk.Label(
                    chain,
                    text="→",
                    bg=SURFACE,
                    fg=TEAL_DARK,
                    font=("Segoe UI", 24, "bold"),
                ).pack(
                    side="left",
                    padx=4,
                )

        explanation = self.make_card(
            content,
            "Почему используется MLP",
        )

        explanation.pack(
            fill="x"
        )

        tk.Label(
            explanation,
            text=(
                "Задача является регрессионной: требуется получить "
                "конкретное числовое значение длительности ремонта. "
                "Многослойный персептрон способен учитывать совместное "
                "и нелинейное влияние нескольких признаков, при этом "
                "остаётся достаточно простым для учебного проекта, "
                "быстро обучается на обычном компьютере и допускает "
                "дальнейшее дообучение при появлении новой истории "
                "ремонтных заявок."
            ),
            bg=SURFACE,
            fg=MUTED,
            justify="left",
            wraplength=1100,
            font=("Segoe UI", 10),
        ).pack(
            anchor="w",
            padx=22,
            pady=(0, 20),
        )

    def build_model_tab(self):
        tab = tk.Frame(
            self.notebook,
            bg=BG,
        )

        self.notebook.add(
            tab,
            text="  Модель  ",
        )

        scroll = ScrollFrame(
            tab
        )

        scroll.pack(
            fill="both",
            expand=True,
            pady=8,
        )

        content = scroll.inner

        metrics = self.make_card(
            content,
            "Качество модели",
            "Метрики рассчитаны на отложенной тестовой выборке",
        )

        metrics.pack(
            fill="x",
            pady=(0, 12),
        )

        row = tk.Frame(
            metrics,
            bg=SURFACE,
        )

        row.pack(
            fill="x",
            padx=22,
            pady=(0, 18),
        )

        data = [
            (
                "MAE",
                f"{self.predictor.mae:.3f} ч",
                "средняя абсолютная ошибка",
                GREEN_SOFT,
            ),
            (
                "RMSE",
                f"{self.predictor.rmse:.3f} ч",
                "среднеквадратичная ошибка",
                BLUE_SOFT,
            ),
            (
                "R²",
                f"{self.predictor.r2:.3f}",
                "коэффициент детерминации",
                ORANGE_SOFT,
            ),
            (
                "Итерации",
                str(self.predictor.model.n_iter_),
                "ранняя остановка",
                PURPLE_SOFT,
            ),
        ]

        for index, item in enumerate(data):
            box = tk.Frame(
                row,
                bg=item[3],
                height=115,
            )

            box.pack(
                side="left",
                fill="x",
                expand=True,
                padx=(
                    0 if index == 0 else 5,
                    0 if index == 3 else 5,
                ),
            )

            box.pack_propagate(
                False
            )

            tk.Label(
                box,
                text=item[0],
                bg=item[3],
                fg=MUTED,
                font=("Segoe UI Semibold", 9),
            ).pack(
                anchor="w",
                padx=16,
                pady=(13, 2),
            )

            tk.Label(
                box,
                text=item[1],
                bg=item[3],
                fg=TEXT,
                font=("Segoe UI", 20, "bold"),
            ).pack(
                anchor="w",
                padx=16,
            )

            tk.Label(
                box,
                text=item[2],
                bg=item[3],
                fg=MUTED,
                font=("Segoe UI", 8),
            ).pack(
                anchor="w",
                padx=16,
                pady=(2, 8),
            )

        config = self.make_card(
            content,
            "Параметры обучения",
        )

        config.pack(
            fill="x",
            pady=(0, 12),
        )

        config_grid = tk.Frame(
            config,
            bg=SURFACE,
        )

        config_grid.pack(
            fill="x",
            padx=22,
            pady=(0, 18),
        )

        config_items = [
            (
                "Архитектура",
                "5 → 18 → 10 → 1",
            ),
            (
                "Функция активации",
                "ReLU",
            ),
            (
                "Оптимизатор",
                "Adam",
            ),
            (
                "Learning rate",
                "0.0025",
            ),
            (
                "Validation fraction",
                "0.15",
            ),
            (
                "Early stopping",
                "включён",
            ),
        ]

        for index, item in enumerate(config_items):
            row_index = index // 3

            column_index = index % 3

            box = tk.Frame(
                config_grid,
                bg=SURFACE_2,
            )

            box.grid(
                row=row_index,
                column=column_index,
                sticky="nsew",
                padx=5,
                pady=5,
            )

            config_grid.grid_columnconfigure(
                column_index,
                weight=1,
            )

            tk.Label(
                box,
                text=item[0],
                bg=SURFACE_2,
                fg=MUTED,
                font=("Segoe UI", 9),
            ).pack(
                anchor="w",
                padx=14,
                pady=(10, 0),
            )

            tk.Label(
                box,
                text=item[1],
                bg=SURFACE_2,
                fg=TEXT,
                font=("Segoe UI Semibold", 11),
            ).pack(
                anchor="w",
                padx=14,
                pady=(2, 10),
            )

        chart = self.make_card(
            content,
            "Кривая функции потерь",
            "Снижение ошибки в процессе обучения",
        )

        chart.pack(
            fill="both",
            expand=True,
        )

        self.loss_canvas = tk.Canvas(
            chart,
            bg=SURFACE,
            highlightthickness=0,
            height=330,
        )

        self.loss_canvas.pack(
            fill="both",
            expand=True,
            padx=22,
            pady=(0, 18),
        )

        self.loss_canvas.bind(
            "<Configure>",
            lambda event: self.draw_loss_curve(),
        )

    def draw_loss_curve(self):
        canvas = self.loss_canvas

        canvas.delete(
            "all"
        )

        width = canvas.winfo_width()

        height = canvas.winfo_height()

        if width < 160 or height < 160:
            return

        losses = self.predictor.model.loss_curve_

        if not losses:
            return

        left = 65
        right = 20
        top = 20
        bottom = 45

        chart_width = width - left - right

        chart_height = height - top - bottom

        minimum = min(
            losses
        )

        maximum = max(
            losses
        )

        if maximum == minimum:
            maximum += 1

        for step in range(5):
            y = (
                top
                + chart_height
                * step
                / 4
            )

            canvas.create_line(
                left,
                y,
                left + chart_width,
                y,
                fill="#EDF2F7",
            )

        canvas.create_line(
            left,
            top,
            left,
            top + chart_height,
            fill=BORDER,
            width=2,
        )

        canvas.create_line(
            left,
            top + chart_height,
            left + chart_width,
            top + chart_height,
            fill=BORDER,
            width=2,
        )

        points = []

        for index, loss in enumerate(losses):
            x = (
                left
                + index
                * chart_width
                / max(
                    1,
                    len(losses) - 1,
                )
            )

            y = (
                top
                + chart_height
                - (
                    loss
                    - minimum
                )
                / (
                    maximum
                    - minimum
                )
                * chart_height
            )

            points.append(
                (
                    x,
                    y,
                )
            )

        for index in range(
            len(points) - 1
        ):
            canvas.create_line(
                points[index][0],
                points[index][1],
                points[index + 1][0],
                points[index + 1][1],
                fill=ORANGE,
                width=3,
                smooth=True,
            )

        if points:
            canvas.create_oval(
                points[0][0] - 4,
                points[0][1] - 4,
                points[0][0] + 4,
                points[0][1] + 4,
                fill=ORANGE,
                outline="",
            )

            canvas.create_oval(
                points[-1][0] - 5,
                points[-1][1] - 5,
                points[-1][0] + 5,
                points[-1][1] + 5,
                fill=TEAL,
                outline="",
            )

        canvas.create_text(
            12,
            top,
            text=f"{maximum:.3f}",
            anchor="w",
            fill=MUTED,
            font=("Segoe UI", 9),
        )

        canvas.create_text(
            12,
            top + chart_height,
            text=f"{minimum:.3f}",
            anchor="w",
            fill=MUTED,
            font=("Segoe UI", 9),
        )

        canvas.create_text(
            left,
            height - 14,
            text="1",
            anchor="s",
            fill=MUTED,
            font=("Segoe UI", 9),
        )

        canvas.create_text(
            left + chart_width,
            height - 14,
            text=str(
                len(losses)
            ),
            anchor="s",
            fill=MUTED,
            font=("Segoe UI", 9),
        )

        canvas.create_text(
            left + chart_width / 2,
            height - 3,
            text="Итерации обучения",
            anchor="s",
            fill=MUTED,
            font=("Segoe UI", 9),
        )

    def build_history_tab(self):
        tab = tk.Frame(
            self.notebook,
            bg=BG,
        )

        self.notebook.add(
            tab,
            text="  История  ",
        )

        top = tk.Frame(
            tab,
            bg=BG,
        )

        top.pack(
            fill="x",
            pady=(8, 10),
        )

        left = tk.Frame(
            top,
            bg=BG,
        )

        left.pack(
            side="left",
            fill="x",
            expand=True,
        )

        tk.Label(
            left,
            text="История прогнозов",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(
            anchor="w"
        )

        tk.Label(
            left,
            text="Результаты текущего сеанса работы программы",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(
            anchor="w",
            pady=(3, 0),
        )

        tk.Button(
            top,
            text="Экспорт CSV",
            command=self.export_history,
            bg=TEAL,
            fg=WHITE,
            activebackground=TEAL_DARK,
            activeforeground=WHITE,
            relief="flat",
            bd=0,
            padx=18,
            pady=9,
            cursor="hand2",
            font=("Segoe UI Semibold", 9),
        ).pack(
            side="right",
            padx=(8, 0),
        )

        tk.Button(
            top,
            text="Очистить",
            command=self.clear_history,
            bg=SURFACE_2,
            fg=TEXT,
            activebackground="#E2EAF2",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=18,
            pady=9,
            cursor="hand2",
            font=("Segoe UI", 9),
        ).pack(
            side="right"
        )

        card = tk.Frame(
            tab,
            bg=SURFACE,
            highlightbackground=BORDER,
            highlightthickness=1,
        )

        card.pack(
            fill="both",
            expand=True,
            pady=(0, 8),
        )

        columns = (
            "time",
            "age",
            "repairs",
            "complexity",
            "parts",
            "priority",
            "hours",
            "reserve",
        )

        self.history_tree = ttk.Treeview(
            card,
            columns=columns,
            show="headings",
        )

        headings = {
            "time": "Время",
            "age": "Возраст",
            "repairs": "Ремонты",
            "complexity": "Сложность",
            "parts": "Запчасти",
            "priority": "Приоритет",
            "hours": "Прогноз, ч",
            "reserve": "Резерв, ч",
        }

        widths = {
            "time": 120,
            "age": 90,
            "repairs": 90,
            "complexity": 100,
            "parts": 110,
            "priority": 100,
            "hours": 110,
            "reserve": 110,
        }

        for column in columns:
            self.history_tree.heading(
                column,
                text=headings[column],
            )

            self.history_tree.column(
                column,
                width=widths[column],
                anchor="center",
            )

        scrollbar = ttk.Scrollbar(
            card,
            orient="vertical",
            command=self.history_tree.yview,
        )

        self.history_tree.configure(
            yscrollcommand=scrollbar.set
        )

        self.history_tree.pack(
            side="left",
            fill="both",
            expand=True,
            padx=8,
            pady=8,
        )

        scrollbar.pack(
            side="right",
            fill="y",
            padx=(0, 8),
            pady=8,
        )

    def fill_demo(self):
        self.age_entry.delete(
            0,
            tk.END,
        )

        self.repairs_entry.delete(
            0,
            tk.END,
        )

        self.complexity_entry.delete(
            0,
            tk.END,
        )

        self.age_entry.insert(
            0,
            "4",
        )

        self.repairs_entry.insert(
            0,
            "2",
        )

        self.complexity_entry.insert(
            0,
            "6",
        )

        self.parts_combo.current(
            0
        )

        self.priority_combo.current(
            1
        )

    def run_prediction(self):
        try:
            age = float(
                self.age_entry
                .get()
                .strip()
                .replace(
                    ",",
                    ".",
                )
            )

            repairs = int(
                self.repairs_entry
                .get()
                .strip()
            )

            complexity = int(
                self.complexity_entry
                .get()
                .strip()
            )

        except ValueError:
            messagebox.showerror(
                "Ошибка ввода",
                "Проверьте числовые значения в форме.",
            )

            return

        if not 0.1 <= age <= 20:
            messagebox.showerror(
                "Ошибка ввода",
                "Возраст оборудования должен быть от 0,1 до 20 лет.",
            )

            return

        if not 0 <= repairs <= 20:
            messagebox.showerror(
                "Ошибка ввода",
                "Количество предыдущих ремонтов должно быть от 0 до 20.",
            )

            return

        if not 1 <= complexity <= 10:
            messagebox.showerror(
                "Ошибка ввода",
                "Сложность неисправности должна быть от 1 до 10.",
            )

            return

        parts_available = (
            1
            if self.parts_combo.current() == 0
            else 0
        )

        priority = (
            self.priority_combo.current()
            + 1
        )

        hours = self.predictor.predict(
            age,
            repairs,
            complexity,
            parts_available,
            priority,
        )

        reserve_hours = (
            hours
            * 1.15
        )

        if hours < 10:
            category = (
                "Короткий ремонт"
            )

            color = GREEN

            recommendation = (
                "Ожидается сравнительно небольшая трудоёмкость. "
                "Заявку можно планировать в стандартном рабочем окне."
            )

        elif hours < 20:
            category = (
                "Средняя длительность"
            )

            color = YELLOW

            recommendation = (
                "Стоит заранее зарезервировать рабочее время специалиста "
                "и проверить готовность необходимых комплектующих."
            )

        else:
            category = (
                "Повышенная длительность"
            )

            color = RED

            recommendation = (
                "Ремонт может быть трудоёмким. Желательно предусмотреть "
                "дополнительный резерв или временную замену оборудования."
            )

        self.prediction_value.config(
            text=f"{hours:.1f}",
            fg=color,
        )

        self.reserve_value.config(
            text=f"{reserve_hours:.1f} ч",
            fg=color,
        )

        self.category_label.config(
            text=category,
            fg=color,
        )

        self.recommendation_label.config(
            text=recommendation,
            fg=TEXT,
        )

        self.summary_text.config(
            text=(
                f"Возраст: {age:.1f} лет\n"
                f"Ремонты: {repairs}\n"
                f"Сложность: {complexity}/10\n"
                f"Запчасти: {self.parts_combo.get()}\n"
                f"Приоритет: {self.priority_combo.get()}"
            ),
            fg=TEXT,
        )

        record = PredictionRecord(
            timestamp=datetime.now().strftime(
                "%H:%M:%S"
            ),
            age=age,
            repairs=repairs,
            complexity=complexity,
            parts_available=parts_available,
            priority=priority,
            predicted_hours=hours,
            reserve_hours=reserve_hours,
            category=category,
        )

        self.history.append(
            record
        )

        self.add_history_row(
            record
        )

    def add_history_row(self, record):
        priority_text = {
            1: "Низкий",
            2: "Средний",
            3: "Высокий",
        }[
            record.priority
        ]

        self.history_tree.insert(
            "",
            "end",
            values=(
                record.timestamp,
                f"{record.age:.1f}",
                record.repairs,
                record.complexity,
                (
                    "Есть"
                    if record.parts_available
                    else "Нет"
                ),
                priority_text,
                f"{record.predicted_hours:.1f}",
                f"{record.reserve_hours:.1f}",
            ),
        )

    def clear_history(self):
        if not self.history:
            return

        if not messagebox.askyesno(
            "Очистка истории",
            "Удалить все прогнозы текущего сеанса?",
        ):
            return

        self.history.clear()

        for item in self.history_tree.get_children():
            self.history_tree.delete(
                item
            )

    def export_history(self):
        if not self.history:
            messagebox.showwarning(
                "Экспорт",
                "История пока пуста.",
            )

            return

        default_name = (
            "gefest_service_predictions_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M"
            )
            + ".csv"
        )

        path = filedialog.asksaveasfilename(
            title="Сохранить историю прогнозов",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[
                (
                    "CSV",
                    "*.csv",
                ),
                (
                    "Все файлы",
                    "*.*",
                ),
            ],
        )

        if not path:
            return

        try:
            with open(
                path,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:
                writer = csv.writer(
                    file,
                    delimiter=";",
                )

                writer.writerow(
                    [
                        "Время",
                        "Возраст, лет",
                        "Предыдущие ремонты",
                        "Сложность",
                        "Запчасти",
                        "Приоритет",
                        "Прогноз, ч",
                        "Резерв, ч",
                        "Категория",
                    ]
                )

                for record in self.history:
                    writer.writerow(
                        [
                            record.timestamp,
                            f"{record.age:.1f}",
                            record.repairs,
                            record.complexity,
                            (
                                "Есть"
                                if record.parts_available
                                else "Нет"
                            ),
                            {
                                1: "Низкий",
                                2: "Средний",
                                3: "Высокий",
                            }[
                                record.priority
                            ],
                            f"{record.predicted_hours:.2f}",
                            f"{record.reserve_hours:.2f}",
                            record.category,
                        ]
                    )

            messagebox.showinfo(
                "Экспорт",
                (
                    "История сохранена:\n"
                    f"{path}"
                ),
            )

        except OSError as error:
            messagebox.showerror(
                "Ошибка экспорта",
                (
                    "Не удалось сохранить файл:\n"
                    f"{error}"
                ),
            )


def main():
    root = tk.Tk()

    App(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
