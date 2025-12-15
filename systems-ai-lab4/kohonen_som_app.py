"""
Самоорганизующиеся карты Кохонена (СОК)
Алгоритм "Победитель забирает всё" с механизмом утомляемости нейронов
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QDoubleSpinBox, QTextEdit, QGroupBox, QGridLayout,
                             QTabWidget, QFileDialog, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QSplitter)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
import time
from datetime import datetime


class KohonenSOM:
    """Самоорганизующаяся карта Кохонена"""

    def __init__(self, n_neurons, input_dim, learning_rate_init=0.5, seed=42):
        np.random.seed(seed)
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.learning_rate_init = learning_rate_init
        self.weights = np.random.randn(n_neurons, input_dim)
        self.initial_weights = self.weights.copy()
        self.weights = self._normalize_weights(self.weights)
        self.history = {'weights': [], 'lr': [], 'winners': []}
        self.fatigue_counters = np.zeros(n_neurons)

    def _normalize_weights(self, weights):
        """Нормировка векторов весов"""
        norms = np.linalg.norm(weights, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return weights / norms

    def _find_winner(self, x, use_fatigue=False, fatigue_penalty=0.1):
        """Поиск нейрона-победителя"""
        x_norm = x / np.linalg.norm(x)
        distances = np.linalg.norm(self.weights - x_norm, axis=1)

        if use_fatigue:
            distances = distances + fatigue_penalty * self.fatigue_counters

        winner_idx = np.argmin(distances)
        return winner_idx

    def _update_weights(self, winner_idx, x, learning_rate):
        """Обновление весов нейрона-победителя"""
        x_norm = x / np.linalg.norm(x)
        self.weights[winner_idx] += learning_rate * (x_norm - self.weights[winner_idx])
        self.weights[winner_idx] = self.weights[winner_idx] / np.linalg.norm(self.weights[winner_idx])

    def train(self, X, n_epochs=100, lr_decay='linear', use_fatigue=False, 
              fatigue_penalty=0.1, fatigue_decay=0.95):
        """Обучение СОК Кохонена"""
        start_time = time.time()

        if use_fatigue:
            self.fatigue_counters = np.zeros(self.n_neurons)

        for epoch in range(n_epochs):
            if lr_decay == 'linear':
                learning_rate = self.learning_rate_init * (1 - epoch / n_epochs)
            else:
                learning_rate = self.learning_rate_init * np.exp(-epoch / (n_epochs / 5))

            indices = np.random.permutation(len(X))
            epoch_winners = []

            for idx in indices:
                x = X[idx]
                winner_idx = self._find_winner(x, use_fatigue, fatigue_penalty)
                epoch_winners.append(winner_idx)
                self._update_weights(winner_idx, x, learning_rate)

                if use_fatigue:
                    self.fatigue_counters[winner_idx] += 1
                    self.fatigue_counters *= fatigue_decay

            if epoch % 10 == 0 or epoch == n_epochs - 1:
                self.history['weights'].append(self.weights.copy())
                self.history['lr'].append(learning_rate)
                self.history['winners'].append(epoch_winners)

        training_time = time.time() - start_time
        return training_time

    def predict(self, X):
        """Предсказание кластеров для данных"""
        predictions = []
        for x in X:
            winner_idx = self._find_winner(x)
            predictions.append(winner_idx)
        return np.array(predictions)

    def get_cluster_centers(self, X):
        """Получение центров кластеров (среднее положение точек)"""
        predictions = self.predict(X)
        centers = np.zeros((self.n_neurons, self.input_dim))

        for neuron_idx in range(self.n_neurons):
            cluster_points = X[predictions == neuron_idx]

            if len(cluster_points) > 0:
                centers[neuron_idx] = np.mean(cluster_points, axis=0)
            else:
                avg_norm = np.mean(np.linalg.norm(X, axis=1))
                centers[neuron_idx] = self.weights[neuron_idx] * avg_norm

        return centers


class MplCanvas(FigureCanvas):
    """Холст для отображения matplotlib графиков"""

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor='white')
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class KohonenApp(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.data = None
        self.som_classic = None
        self.som_fatigue = None
        self.time_classic = 0
        self.time_fatigue = 0
        self.predictions_classic = None
        self.predictions_fatigue = None
        self.initUI()
        self.load_default_data()

    def initUI(self):
        self.setWindowTitle('Самоорганизующиеся карты Кохонена')
        self.setGeometry(100, 100, 1600, 950)

        # Единый профессиональный стиль в синей палитре
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f9fa;
            }
            QWidget {
                background-color: white;
                color: #2c3e50;
                font-family: 'Segoe UI', 'Arial', sans-serif;
                font-size: 10pt;
            }
            QGroupBox {
                font-weight: 600;
                font-size: 11pt;
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                margin-top: 18px;
                padding-top: 20px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px;
                color: #495057;
                background-color: white;
            }
            QPushButton {
                background-color: #4a90e2;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                color: white;
                font-weight: 600;
                font-size: 10pt;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2868a8;
            }
            QPushButton:disabled {
                background-color: #cbd5e0;
                color: #a0aec0;
            }
            QLabel {
                color: #2c3e50;
                background-color: transparent;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: white;
                border: 2px solid #e1e8ed;
                border-radius: 5px;
                padding: 8px 12px;
                color: #2c3e50;
                font-size: 10pt;
                min-height: 20px;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #4a90e2;
                background-color: #f8f9fa;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                width: 20px;
                border-left: 1px solid #e1e8ed;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 20px;
                border-left: 1px solid #e1e8ed;
            }
            QTextEdit {
                background-color: #f8f9fa;
                border: 2px solid #e1e8ed;
                border-radius: 6px;
                padding: 10px;
                color: #2c3e50;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 9pt;
            }
            QTabWidget::pane {
                border: 2px solid #e1e8ed;
                border-radius: 8px;
                background-color: white;
                top: -2px;
            }
            QTabBar::tab {
                background-color: #f8f9fa;
                border: 2px solid #e1e8ed;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 12px 25px;
                color: #6c757d;
                font-weight: 600;
                margin-right: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #4a90e2;
                border-color: #e1e8ed;
                border-bottom-color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e9ecef;
                color: #495057;
            }
            QTableWidget {
                background-color: white;
                border: 2px solid #e1e8ed;
                border-radius: 6px;
                gridline-color: #f1f3f5;
            }
            QTableWidget::item {
                padding: 8px;
            }
            QTableWidget::item:selected {
                background-color: #e3f2fd;
                color: #2c3e50;
            }
            QHeaderView::section {
                background-color: #4a90e2;
                color: white;
                padding: 10px;
                border: none;
                font-weight: 600;
                font-size: 10pt;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)

        right_panel = self.create_visualization_panel()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

    def create_control_panel(self):
        """Создание панели управления"""
        panel = QWidget()
        panel.setMaximumWidth(450)
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        header_widget = QWidget()
        header_widget.setStyleSheet('background-color: #4a90e2; border-radius: 10px; padding: 20px;')
        header_layout = QVBoxLayout(header_widget)

        title = QLabel('СОК Кохонена')
        title_font = QFont('Segoe UI', 20, QFont.Weight.Bold)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet('color: white;')
        header_layout.addWidget(title)

        subtitle = QLabel('Самоорганизующиеся карты')
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet('color: white; font-size: 11pt; opacity: 0.9;')
        header_layout.addWidget(subtitle)

        layout.addWidget(header_widget)

        # Группа: Данные
        data_group = QGroupBox('Управление данными')
        data_layout = QVBoxLayout()
        data_layout.setSpacing(12)

        btn_default = QPushButton('Загрузить данные по умолчанию')
        btn_default.clicked.connect(self.load_default_data)
        data_layout.addWidget(btn_default)

        btn_load = QPushButton('Открыть файл с данными...')
        btn_load.clicked.connect(self.load_data_from_file)
        data_layout.addWidget(btn_load)

        self.lbl_data_info = QLabel('Данных нет')
        self.lbl_data_info.setStyleSheet('''
            padding: 12px; 
            background-color: #f8f9fa; 
            border: 2px solid #e1e8ed;
            border-radius: 6px;
            font-weight: 600;
            color: #6c757d;
        ''')
        self.lbl_data_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        data_layout.addWidget(self.lbl_data_info)

        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        params_group = QGroupBox('Параметры обучения')
        params_layout = QGridLayout()
        params_layout.setSpacing(18)
        params_layout.setColumnStretch(1, 1)
        params_layout.setVerticalSpacing(15)

        row = 0
        lbl = QLabel('Количество нейронов:')
        lbl.setStyleSheet('font-weight: 500; font-size: 10pt;')
        params_layout.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignLeft)
        self.spin_neurons = QSpinBox()
        self.spin_neurons.setRange(2, 10)
        self.spin_neurons.setValue(3)
        self.spin_neurons.setMinimumWidth(120)
        params_layout.addWidget(self.spin_neurons, row, 1)

        row += 1
        lbl = QLabel('Количество эпох:')
        lbl.setStyleSheet('font-weight: 500; font-size: 10pt;')
        params_layout.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignLeft)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(10, 1000)
        self.spin_epochs.setValue(100)
        self.spin_epochs.setMinimumWidth(120)
        params_layout.addWidget(self.spin_epochs, row, 1)

        row += 1
        lbl = QLabel('Скорость обучения:')
        lbl.setStyleSheet('font-weight: 500; font-size: 10pt;')
        params_layout.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignLeft)
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.1, 1.0)
        self.spin_lr.setSingleStep(0.1)
        self.spin_lr.setValue(0.5)
        self.spin_lr.setMinimumWidth(120)
        params_layout.addWidget(self.spin_lr, row, 1)

        row += 1
        lbl = QLabel('Random seed:')
        lbl.setStyleSheet('font-weight: 500; font-size: 10pt;')
        params_layout.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignLeft)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(42)
        self.spin_seed.setMinimumWidth(120)
        params_layout.addWidget(self.spin_seed, row, 1)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        train_group = QGroupBox('Запуск обучения')
        train_layout = QVBoxLayout()
        train_layout.setSpacing(12)

        btn_train_classic = QPushButton('Обучить классический алгоритм')
        btn_train_classic.clicked.connect(self.train_classic)
        train_layout.addWidget(btn_train_classic)

        btn_train_fatigue = QPushButton('Обучить с утомляемостью')
        btn_train_fatigue.clicked.connect(self.train_fatigue)
        train_layout.addWidget(btn_train_fatigue)

        btn_train_both = QPushButton('Обучить оба алгоритма')
        btn_train_both.clicked.connect(self.train_both)
        train_layout.addWidget(btn_train_both)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        export_group = QGroupBox('Экспорт результатов')
        export_layout = QVBoxLayout()

        btn_save_report = QPushButton('Сохранить отчет')
        btn_save_report.clicked.connect(self.save_report)
        export_layout.addWidget(btn_save_report)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()

        return panel

    def create_visualization_panel(self):
        """Создание панели визуализации"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        # Вкладка 1: Исходные данные
        self.canvas_data = MplCanvas(self, width=9, height=6, dpi=100)
        self.tabs.addTab(self.canvas_data, 'Исходные данные')

        tab_classic = QWidget()
        layout_classic = QVBoxLayout(tab_classic)
        self.canvas_classic = MplCanvas(self, width=9, height=5, dpi=100)
        layout_classic.addWidget(self.canvas_classic, stretch=2)

        self.table_classic = self.create_results_table()
        layout_classic.addWidget(self.table_classic, stretch=1)

        self.tabs.addTab(tab_classic, 'Классический алгоритм')

        tab_fatigue = QWidget()
        layout_fatigue = QVBoxLayout(tab_fatigue)
        self.canvas_fatigue = MplCanvas(self, width=9, height=5, dpi=100)
        layout_fatigue.addWidget(self.canvas_fatigue, stretch=2)

        self.table_fatigue = self.create_results_table()
        layout_fatigue.addWidget(self.table_fatigue, stretch=1)

        self.tabs.addTab(tab_fatigue, 'С утомляемостью')

        tab_comparison = QWidget()
        layout_comparison = QVBoxLayout(tab_comparison)
        self.canvas_comparison = MplCanvas(self, width=9, height=5, dpi=100)
        layout_comparison.addWidget(self.canvas_comparison, stretch=2)

        self.text_comparison = QTextEdit()
        self.text_comparison.setReadOnly(True)
        self.text_comparison.setMaximumHeight(150)
        layout_comparison.addWidget(self.text_comparison, stretch=1)

        self.tabs.addTab(tab_comparison, 'Сравнение')

        layout.addWidget(self.tabs)

        return panel

    def create_results_table(self):
        """Создание таблицы результатов"""
        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(['Кластер', 'Количество', 'Центр X', 'Центр Y', 'Номера точек'])

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        return table

    def update_results_table(self, table, predictions, centers):
        """Обновление таблицы результатов"""
        table.setRowCount(self.spin_neurons.value())

        for i in range(self.spin_neurons.value()):
            count = np.sum(predictions == i)
            points = np.where(predictions == i)[0] + 1

            item = QTableWidgetItem(f'Кластер {i}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if count == 0:
                item.setBackground(QColor('#fee'))
            else:
                item.setBackground(QColor('#e8f5e9'))
            item.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
            table.setItem(i, 0, item)

            item = QTableWidgetItem(str(count))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setFont(QFont('Segoe UI', 10, QFont.Weight.Bold))
            table.setItem(i, 1, item)

            item = QTableWidgetItem(f'{centers[i, 0]:.2f}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 2, item)

            item = QTableWidgetItem(f'{centers[i, 1]:.2f}')
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 3, item)

            if count > 0:
                points_str = ', '.join(map(str, points))
            else:
                points_str = 'Пустой кластер'
            item = QTableWidgetItem(points_str)
            item.setFont(QFont('Consolas', 9))
            table.setItem(i, 4, item)

    def load_default_data(self):
        """Загрузка данных по умолчанию"""
        self.data = np.array([
            [12, 8], [16, -5], [12, 20], [12, 17], [16, 2],
            [12, -11], [15, 8], [16, 1], [17, 4], [16, -8],
            [18, -6], [15, 8], [15, 2], [11, 5]
        ])
        self.lbl_data_info.setText(f'Загружено: {len(self.data)} точек')
        self.lbl_data_info.setStyleSheet('''
            padding: 12px; 
            background-color: #e8f5e9; 
            border: 2px solid #4caf50;
            border-radius: 6px;
            font-weight: 600;
            color: #2e7d32;
        ''')
        self.plot_data()
        QMessageBox.information(self, 'Данные загружены', 
                               f'Загружено {len(self.data)} точек данных по умолчанию.')

    def load_data_from_file(self):
        """Загрузка данных из файла"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Открыть файл с данными', '', 
            'CSV Files (*.csv);;Text Files (*.txt);;All Files (*.*)'
        )
        if filename:
            try:
                self.data = np.loadtxt(filename, delimiter=',')
                self.lbl_data_info.setText(f'Загружено: {len(self.data)} точек')
                self.lbl_data_info.setStyleSheet('''
                    padding: 12px; 
                    background-color: #e8f5e9; 
                    border: 2px solid #4caf50;
                    border-radius: 6px;
                    font-weight: 600;
                    color: #2e7d32;
                ''')
                self.plot_data()
                QMessageBox.information(self, 'Успех', 
                                       f'Данные успешно загружены из файла:\n{filename}')
            except Exception as e:
                QMessageBox.critical(self, 'Ошибка загрузки', 
                                   f'Не удалось загрузить файл:\n{str(e)}')

    def plot_data(self):
        """Отображение исходных данных"""
        self.canvas_data.axes.clear()

        # Рисуем точки
        scatter = self.canvas_data.axes.scatter(self.data[:, 0], self.data[:, 1], 
                                               c='#4a90e2', s=180, alpha=0.8, 
                                               edgecolors='#2c3e50', linewidths=2.5,
                                               zorder=3)

        # Добавляем номера точек
        for i, (x, y) in enumerate(self.data):
            self.canvas_data.axes.annotate(f'{i+1}', (x, y), fontsize=8, 
                                          ha='center', va='center', 
                                          fontweight='bold', color='white', zorder=4)

        self.canvas_data.axes.set_xlabel('Признак 1', fontsize=13, fontweight='600')
        self.canvas_data.axes.set_ylabel('Признак 2', fontsize=13, fontweight='600')
        self.canvas_data.axes.set_title('Исходные данные для кластеризации', 
                                       fontsize=15, fontweight='bold', pad=20)
        self.canvas_data.axes.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        self.canvas_data.axes.set_facecolor('#fafbfc')

        # Добавляем рамку
        for spine in self.canvas_data.axes.spines.values():
            spine.set_edgecolor('#e1e8ed')
            spine.set_linewidth(2)

        self.canvas_data.figure.tight_layout()
        self.canvas_data.draw()

    def train_classic(self):
        """Обучение классического алгоритма"""
        if self.data is None:
            QMessageBox.warning(self, 'Предупреждение', 'Сначала загрузите данные!')
            return

        self.som_classic = KohonenSOM(n_neurons=self.spin_neurons.value(), 
                                     input_dim=2, 
                                     learning_rate_init=self.spin_lr.value(),
                                     seed=self.spin_seed.value())

        self.time_classic = self.som_classic.train(self.data, 
                                                   n_epochs=self.spin_epochs.value(),
                                                   use_fatigue=False)

        self.predictions_classic = self.som_classic.predict(self.data)
        centers = self.som_classic.get_cluster_centers(self.data)

        self.plot_classic_results()
        self.update_results_table(self.table_classic, self.predictions_classic, centers)
        self.tabs.setCurrentIndex(1)

        QMessageBox.information(self, 'Обучение завершено', 
                               f'Классический алгоритм обучен за {self.time_classic:.4f} сек')

    def train_fatigue(self):
        """Обучение с утомляемостью"""
        if self.data is None:
            QMessageBox.warning(self, 'Предупреждение', 'Сначала загрузите данные!')
            return

        self.som_fatigue = KohonenSOM(n_neurons=self.spin_neurons.value(), 
                                     input_dim=2, 
                                     learning_rate_init=self.spin_lr.value(),
                                     seed=self.spin_seed.value())

        self.time_fatigue = self.som_fatigue.train(self.data, 
                                                   n_epochs=self.spin_epochs.value(),
                                                   use_fatigue=True,
                                                   fatigue_penalty=0.15,
                                                   fatigue_decay=0.95)

        self.predictions_fatigue = self.som_fatigue.predict(self.data)
        centers = self.som_fatigue.get_cluster_centers(self.data)

        self.plot_fatigue_results()
        self.update_results_table(self.table_fatigue, self.predictions_fatigue, centers)
        self.tabs.setCurrentIndex(2)

        QMessageBox.information(self, 'Обучение завершено', 
                               f'Алгоритм с утомляемостью обучен за {self.time_fatigue:.4f} сек')

    def train_both(self):
        """Обучение обоих алгоритмов"""
        self.train_classic()
        self.train_fatigue()
        self.plot_comparison()
        self.tabs.setCurrentIndex(3)

    def plot_classic_results(self):
        """Визуализация результатов классического алгоритма"""
        self.canvas_classic.axes.clear()

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        predictions = self.predictions_classic
        centers = self.som_classic.get_cluster_centers(self.data)

        legend_elements = []
        center_added = False
        weight_vector_added = False
        empty_added = False

        for i in range(self.spin_neurons.value()):
            mask = predictions == i
            cluster_points = self.data[mask]

            if len(cluster_points) > 0:
                from scipy.spatial import ConvexHull
                if len(cluster_points) >= 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        for simplex in hull.simplices:
                            self.canvas_classic.axes.plot(cluster_points[simplex, 0], 
                                                         cluster_points[simplex, 1], 
                                                         colors[i % len(colors)], 
                                                         alpha=0.3, linewidth=1.5, zorder=1)
                    except:
                        pass

                self.canvas_classic.axes.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                                c=colors[i % len(colors)], s=160, alpha=0.8,
                                                edgecolors='#2c3e50', linewidths=2, 
                                                label=f'Кластер {i} ({len(cluster_points)} точек)',
                                                zorder=3)

                if not center_added:
                    self.canvas_classic.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='*', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10)
                    legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                                 markerfacecolor='gray', markersize=10,
                                                 markeredgecolor='#2c3e50', markeredgewidth=1.5,
                                                 label='Центр кластера', linestyle='None'))
                    center_added = True
                else:
                    self.canvas_classic.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='*', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10)
            else:
                if not empty_added:
                    self.canvas_classic.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='X', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10, alpha=0.5)
                    legend_elements.append(Line2D([0], [0], marker='X', color='w', 
                                                 markerfacecolor='gray', markersize=9,
                                                 markeredgecolor='#2c3e50', markeredgewidth=1.5,
                                                 label='Пустой кластер', linestyle='None', alpha=0.5))
                    empty_added = True
                else:
                    self.canvas_classic.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='X', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10, alpha=0.5)

        max_norm = np.max(np.linalg.norm(self.data, axis=1))
        for i in range(self.spin_neurons.value()):
            w = self.som_classic.weights[i] * max_norm
            if not weight_vector_added:
                arrow = FancyArrowPatch((0, 0), (w[0], w[1]),
                                       arrowstyle='->', mutation_scale=15, 
                                       linewidth=2, color='#34495e', 
                                       alpha=0.5, zorder=5, linestyle='--')
                self.canvas_classic.axes.add_patch(arrow)
                legend_elements.append(Line2D([0], [0], color='#34495e', linewidth=2,
                                             linestyle='--', alpha=0.5,
                                             label='Вектор весов'))
                weight_vector_added = True
            else:
                arrow = FancyArrowPatch((0, 0), (w[0], w[1]),
                                       arrowstyle='->', mutation_scale=15, 
                                       linewidth=2, color='#34495e', 
                                       alpha=0.5, zorder=5, linestyle='--')
                self.canvas_classic.axes.add_patch(arrow)

        self.canvas_classic.axes.set_xlabel('Признак 1', fontsize=12, fontweight='600')
        self.canvas_classic.axes.set_ylabel('Признак 2', fontsize=12, fontweight='600')
        self.canvas_classic.axes.set_title('Классический алгоритм "Победитель забирает всё"', 
                                          fontsize=14, fontweight='bold', pad=15)

        handles, labels = self.canvas_classic.axes.get_legend_handles_labels()
        all_handles = handles + legend_elements
        self.canvas_classic.axes.legend(handles=all_handles, loc='best', fontsize=9, 
                                       framealpha=0.95, edgecolor='#e1e8ed')

        self.canvas_classic.axes.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        self.canvas_classic.axes.axhline(y=0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.4)
        self.canvas_classic.axes.axvline(x=0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.4)
        self.canvas_classic.axes.set_facecolor('#fafbfc')

        # Рамка
        for spine in self.canvas_classic.axes.spines.values():
            spine.set_edgecolor('#e1e8ed')
            spine.set_linewidth(2)

        self.canvas_classic.figure.tight_layout()
        self.canvas_classic.draw()

    def plot_fatigue_results(self):
        """Визуализация результатов алгоритма с утомляемостью"""
        self.canvas_fatigue.axes.clear()

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']
        predictions = self.predictions_fatigue
        centers = self.som_fatigue.get_cluster_centers(self.data)

        legend_elements = []
        center_added = False
        weight_vector_added = False
        empty_added = False

        for i in range(self.spin_neurons.value()):
            mask = predictions == i
            cluster_points = self.data[mask]

            if len(cluster_points) > 0:
                # Рисуем окружность вокруг кластера
                from scipy.spatial import ConvexHull
                if len(cluster_points) >= 3:
                    try:
                        hull = ConvexHull(cluster_points)
                        for simplex in hull.simplices:
                            self.canvas_fatigue.axes.plot(cluster_points[simplex, 0], 
                                                         cluster_points[simplex, 1], 
                                                         colors[i % len(colors)], 
                                                         alpha=0.3, linewidth=1.5, zorder=1)
                    except:
                        pass

                self.canvas_fatigue.axes.scatter(cluster_points[:, 0], cluster_points[:, 1],
                                                c=colors[i % len(colors)], s=160, alpha=0.8,
                                                edgecolors='#2c3e50', linewidths=2, 
                                                label=f'Кластер {i} ({len(cluster_points)} точек)',
                                                zorder=3)

                if not center_added:
                    self.canvas_fatigue.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='*', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10)
                    legend_elements.append(Line2D([0], [0], marker='*', color='w', 
                                                 markerfacecolor='gray', markersize=10,
                                                 markeredgecolor='#2c3e50', markeredgewidth=1.5,
                                                 label='Центр кластера', linestyle='None'))
                    center_added = True
                else:
                    self.canvas_fatigue.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='*', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10)
            else:
                if not empty_added:
                    self.canvas_fatigue.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='X', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10, alpha=0.5)
                    legend_elements.append(Line2D([0], [0], marker='X', color='w', 
                                                 markerfacecolor='gray', markersize=9,
                                                 markeredgecolor='#2c3e50', markeredgewidth=1.5,
                                                 label='Пустой кластер', linestyle='None', alpha=0.5))
                    empty_added = True
                else:
                    self.canvas_fatigue.axes.scatter([centers[i, 0]], [centers[i, 1]], 
                                                    c=colors[i % len(colors)],
                                                    s=300, marker='X', edgecolors='#2c3e50', 
                                                    linewidths=2, zorder=10, alpha=0.5)

        max_norm = np.max(np.linalg.norm(self.data, axis=1))
        for i in range(self.spin_neurons.value()):
            w = self.som_fatigue.weights[i] * max_norm
            if not weight_vector_added:
                arrow = FancyArrowPatch((0, 0), (w[0], w[1]),
                                       arrowstyle='->', mutation_scale=15, 
                                       linewidth=2, color='#c0392b', 
                                       alpha=0.5, zorder=5, linestyle='--')
                self.canvas_fatigue.axes.add_patch(arrow)
                legend_elements.append(Line2D([0], [0], color='#c0392b', linewidth=2,
                                             linestyle='--', alpha=0.5,
                                             label='Вектор весов'))
                weight_vector_added = True
            else:
                arrow = FancyArrowPatch((0, 0), (w[0], w[1]),
                                       arrowstyle='->', mutation_scale=15, 
                                       linewidth=2, color='#c0392b', 
                                       alpha=0.5, zorder=5, linestyle='--')
                self.canvas_fatigue.axes.add_patch(arrow)

        self.canvas_fatigue.axes.set_xlabel('Признак 1', fontsize=12, fontweight='600')
        self.canvas_fatigue.axes.set_ylabel('Признак 2', fontsize=12, fontweight='600')
        self.canvas_fatigue.axes.set_title('Алгоритм с механизмом утомляемости нейронов', 
                                          fontsize=14, fontweight='bold', pad=15)

        handles, labels = self.canvas_fatigue.axes.get_legend_handles_labels()
        all_handles = handles + legend_elements
        self.canvas_fatigue.axes.legend(handles=all_handles, loc='best', fontsize=9, 
                                       framealpha=0.95, edgecolor='#e1e8ed')

        self.canvas_fatigue.axes.grid(True, alpha=0.2, linestyle='--', linewidth=1)
        self.canvas_fatigue.axes.axhline(y=0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.4)
        self.canvas_fatigue.axes.axvline(x=0, color='#95a5a6', linestyle='-', linewidth=1, alpha=0.4)
        self.canvas_fatigue.axes.set_facecolor('#fafbfc')

        # Рамка
        for spine in self.canvas_fatigue.axes.spines.values():
            spine.set_edgecolor('#e1e8ed')
            spine.set_linewidth(2)

        self.canvas_fatigue.figure.tight_layout()
        self.canvas_fatigue.draw()

    def plot_comparison(self):
        """Визуализация сравнения алгоритмов"""
        if self.som_classic is None or self.som_fatigue is None:
            return

        self.canvas_comparison.axes.clear()

        x = np.arange(self.spin_neurons.value())
        width = 0.35

        counts_classic = [np.sum(self.predictions_classic == i) for i in range(self.spin_neurons.value())]
        counts_fatigue = [np.sum(self.predictions_fatigue == i) for i in range(self.spin_neurons.value())]

        bars1 = self.canvas_comparison.axes.bar(x - width/2, counts_classic, width, 
                                               label='Классический', color='#4a90e2', 
                                               edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        bars2 = self.canvas_comparison.axes.bar(x + width/2, counts_fatigue, width, 
                                               label='С утомляемостью', color='#e74c3c', 
                                               edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                self.canvas_comparison.axes.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                                                f'{int(height)}',
                                                ha='center', va='bottom', fontweight='bold', fontsize=11)

        self.canvas_comparison.axes.set_xlabel('Номер кластера', fontsize=12, fontweight='600')
        self.canvas_comparison.axes.set_ylabel('Количество точек', fontsize=12, fontweight='600')
        self.canvas_comparison.axes.set_title('Сравнение распределения точек по кластерам', 
                                             fontsize=14, fontweight='bold', pad=15)
        self.canvas_comparison.axes.set_xticks(x)
        self.canvas_comparison.axes.set_xticklabels([f'Кластер {i}' for i in range(self.spin_neurons.value())])
        self.canvas_comparison.axes.legend(fontsize=11, framealpha=0.95, edgecolor='#e1e8ed')
        self.canvas_comparison.axes.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=1)
        self.canvas_comparison.axes.set_facecolor('#fafbfc')

        max_count = max(max(counts_classic), max(counts_fatigue))
        self.canvas_comparison.axes.set_ylim(0, max_count + 2)

        for spine in self.canvas_comparison.axes.spines.values():
            spine.set_edgecolor('#e1e8ed')
            spine.set_linewidth(2)

        self.canvas_comparison.figure.tight_layout()
        self.canvas_comparison.draw()

        self.text_comparison.clear()
        self.text_comparison.append('╔' + '='*68 + '╗')
        self.text_comparison.append('║' + ' '*18 + 'СРАВНЕНИЕ АЛГОРИТМОВ' + ' '*30 + '║')
        self.text_comparison.append('╠' + '='*68 + '╣')
        self.text_comparison.append(f'║ Время обучения классического:     {self.time_classic:8.4f} сек' + ' '*18 + '║')
        self.text_comparison.append(f'║ Время обучения с утомляемостью:   {self.time_fatigue:8.4f} сек' + ' '*18 + '║')
        diff_pct = abs(self.time_classic - self.time_fatigue) / self.time_classic * 100
        self.text_comparison.append(f'║ Разница во времени:               {diff_pct:8.1f} %' + ' '*20 + '║')
        self.text_comparison.append('╠' + '='*68 + '╣')

        std_classic = np.std(counts_classic)
        std_fatigue = np.std(counts_fatigue)
        self.text_comparison.append(f'║ Стандартное отклонение (классич.): {std_classic:7.2f}' + ' '*23 + '║')
        self.text_comparison.append(f'║ Стандартное отклонение (утомл.):   {std_fatigue:7.2f}' + ' '*23 + '║')

        empty_classic = sum(1 for c in counts_classic if c == 0)
        empty_fatigue = sum(1 for c in counts_fatigue if c == 0)
        self.text_comparison.append(f'║ Пустых кластеров (классич.):       {empty_classic:7d}' + ' '*25 + '║')
        self.text_comparison.append(f'║ Пустых кластеров (утомл.):         {empty_fatigue:7d}' + ' '*25 + '║')

        self.text_comparison.append('╚' + '='*68 + '╝')

    def save_report(self):
        """Сохранение отчета о результатах обучения"""
        if self.data is None:
            QMessageBox.warning(self, 'Предупреждение', 'Нет данных для сохранения!')
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить отчет', 
            f'kohonen_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            'Text Files (*.txt);;All Files (*.*)'
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write('='*80 + '\n')
                    f.write(' '*20 + 'ОТЧЕТ ПО ОБУЧЕНИЮ СОК КОХОНЕНА\n')
                    f.write('='*80 + '\n\n')
                    f.write(f'Дата и время: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')

                    f.write('ПАРАМЕТРЫ ОБУЧЕНИЯ:\n')
                    f.write('-'*80 + '\n')
                    f.write(f'Количество точек данных: {len(self.data)}\n')
                    f.write(f'Количество нейронов: {self.spin_neurons.value()}\n')
                    f.write(f'Количество эпох: {self.spin_epochs.value()}\n')
                    f.write(f'Начальная скорость обучения: {self.spin_lr.value()}\n')
                    f.write(f'Random seed: {self.spin_seed.value()}\n\n')

                    f.write('ИСХОДНЫЕ ДАННЫЕ:\n')
                    f.write('-'*80 + '\n')
                    f.write(f'{"№":>4}  {"Признак 1":>12}  {"Признак 2":>12}\n')
                    for i, (x, y) in enumerate(self.data, 1):
                        f.write(f'{i:4d}  {x:12.2f}  {y:12.2f}\n')
                    f.write('\n')

                    if self.som_classic is not None:
                        f.write('КЛАССИЧЕСКИЙ АЛГОРИТМ "ПОБЕДИТЕЛЬ ЗАБИРАЕТ ВСЁ":\n')
                        f.write('-'*80 + '\n')
                        f.write(f'Время обучения: {self.time_classic:.4f} секунд\n\n')

                        centers = self.som_classic.get_cluster_centers(self.data)
                        f.write(f'{"Кластер":<12} {"Точек":<8} {"Центр X":<12} {"Центр Y":<12} {"Номера точек"}\n')
                        f.write('-'*80 + '\n')
                        for i in range(self.spin_neurons.value()):
                            count = np.sum(self.predictions_classic == i)
                            points = np.where(self.predictions_classic == i)[0] + 1
                            points_str = ', '.join(map(str, points)) if count > 0 else 'Пустой'
                            f.write(f'{i:<12} {count:<8} {centers[i, 0]:<12.2f} {centers[i, 1]:<12.2f} {points_str}\n')
                        f.write('\n')

                    if self.som_fatigue is not None:
                        f.write('АЛГОРИТМ С МЕХАНИЗМОМ УТОМЛЯЕМОСТИ НЕЙРОНОВ:\n')
                        f.write('-'*80 + '\n')
                        f.write(f'Время обучения: {self.time_fatigue:.4f} секунд\n\n')

                        centers = self.som_fatigue.get_cluster_centers(self.data)
                        f.write(f'{"Кластер":<12} {"Точек":<8} {"Центр X":<12} {"Центр Y":<12} {"Номера точек"}\n')
                        f.write('-'*80 + '\n')
                        for i in range(self.spin_neurons.value()):
                            count = np.sum(self.predictions_fatigue == i)
                            points = np.where(self.predictions_fatigue == i)[0] + 1
                            points_str = ', '.join(map(str, points)) if count > 0 else 'Пустой'
                            f.write(f'{i:<12} {count:<8} {centers[i, 0]:<12.2f} {centers[i, 1]:<12.2f} {points_str}\n')
                        f.write('\n')

                    if self.som_classic is not None and self.som_fatigue is not None:
                        f.write('СРАВНЕНИЕ АЛГОРИТМОВ:\n')
                        f.write('-'*80 + '\n')
                        f.write(f'Время классического: {self.time_classic:.4f} сек\n')
                        f.write(f'Время с утомляемостью: {self.time_fatigue:.4f} сек\n')
                        diff_pct = abs(self.time_classic - self.time_fatigue) / self.time_classic * 100
                        f.write(f'Разница: {diff_pct:.1f}%\n\n')

                    f.write('='*80 + '\n')
                    f.write('Конец отчета\n')
                    f.write('='*80 + '\n')

                QMessageBox.information(self, 'Успех', f'Отчет успешно сохранен:\n{filename}')

            except Exception as e:
                QMessageBox.critical(self, 'Ошибка', f'Не удалось сохранить отчет:\n{str(e)}')


def main():
    app = QApplication(sys.argv)
    window = KohonenApp()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
