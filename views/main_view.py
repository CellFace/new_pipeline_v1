# coding=utf-8

# Standard Library Imports
import os
import sys
import json
import ctypes

# Third Party Imports
import pythoncom
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSizePolicy, QSplitter,
                            QGroupBox, QMenu, QCheckBox, QAction, QStatusBar, QLabel, QLineEdit, QFormLayout, QFileDialog)
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Adjusting sys.path to include parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local Application/Library Specific Imports
from base.main import MainApp
from base.settings import LoggerMixin

# Ensure the application is initialized in STA mode
if not ctypes.windll.ole32.CoInitializeEx(None, 2):  # 2 means COINIT_APARTMENTTHREADED
    raise RuntimeError("Failed to initialize COM library")


class MyMplCanvas(FigureCanvas):
    
    def __init__(self):
        self.fig = Figure()
        self.axes = self.fig.add_subplot(111)
        super(MyMplCanvas, self).__init__(self.fig)

    def plot(self, data):
        self.axes.imshow(data, cmap="gray", vmin=0, vmax=255)
        self.draw()


class ConfigWindow(QMainWindow):
    
    def __init__(self, config_path):
        super().__init__()

        self.config_path = config_path

        self.setWindowTitle("Configuration")
        self.setGeometry(200, 200, 300, 300)

        self.layout = QFormLayout()

        with open(self.config_path, "r") as f:
            self.configs = json.load(f)

        self.textboxes = {}

        for key, value in self.configs.items():
            textbox = QLineEdit(str(value))
            self.layout.addRow(
                QLabel(key), textbox
            )
            self.textboxes[key] = textbox

        save_button = QPushButton("Save Configurations")
        save_button.clicked.connect(self.save_configurations)
        self.layout.addRow("", save_button)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

    def save_configurations(self):
        for key, textbox in self.textboxes.items():
            self.configs[key] = type(self.configs[key])(textbox.text())

        with open(self.config_path, "w") as f:
            json.dump(self.configs, f)


class Worker(QObject):
    
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, file_path, ai_reconstruction, png_upload, post_data_processing, config):
        super().__init__()
        self.file_path = file_path
        self.ai_reconstruction = ai_reconstruction
        self.png_upload = png_upload
        self.post_data_processing = post_data_processing
        self.config = config
        self.app = MainApp()

    def run(self):
        pythoncom.CoInitialize()
        try:
            self.app.process_data(
                h5_file_path=self.file_path,
                ai_reconstruction=self.ai_reconstruction,
                png_upload=self.png_upload,
                post_data_processing=self.post_data_processing,
                config=self.config,
            )
        finally:
            pythoncom.CoUninitialize()
        self.finished.emit()


class ProcessH5Form(QMainWindow, LoggerMixin):
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Process Form")
        self.setGeometry(500, 500, 500, 500)

        self.layout = QVBoxLayout()

        self.file_path = QLineEdit()
        self.layout.addWidget(self.file_path)

        self.select_file_button = QPushButton("Select File")
        self.select_file_button.clicked.connect(self.select_file)
        self.layout.addWidget(self.select_file_button)

        self.process_button = QPushButton("Process File")
        self.process_button.clicked.connect(self.process_file)
        self.layout.addWidget(self.process_button)

        self.ai_checkbox = QCheckBox("AI Reconstruction")
        self.layout.addWidget(self.ai_checkbox)

        self.png_checkbox = QCheckBox("Upload PNG Images")
        self.layout.addWidget(self.png_checkbox)

        self.postdata_process_checkbox = QCheckBox("Post Data Process")
        self.postdata_process_checkbox.stateChanged.connect(
            self.toggle_other_checkboxes
        )
        self.layout.addWidget(self.postdata_process_checkbox)

        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        self.app = MainApp()

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "All Files (*);;Python Files (*.py)",
            options=options,
        )
        if file_path:
            self.file_path.setText(file_path)

    def process_file(self):
        file_path = self.file_path.text()
        ai_reconstruction = self.ai_checkbox.isChecked()
        png_upload = self.png_checkbox.isChecked()
        post_data_processing = self.postdata_process_checkbox.isChecked()

        self.thread = QThread()
        self.worker = Worker(
            file_path,
            ai_reconstruction,
            png_upload,
            post_data_processing,
            self.config,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

        self.process_button.setEnabled(False)
        self.worker.finished.connect(lambda: self.process_button.setEnabled(True))

    def toggle_other_checkboxes(self, state):
        is_checked = state == Qt.Checked
        self.ovizio_checkbox.setDisabled(is_checked)
        self.ai_checkbox.setDisabled(is_checked)
        self.png_checkbox.setDisabled(is_checked)


class ProcessingWorker(QObject):
    finished  = pyqtSignal()
    message   = pyqtSignal(str)          # progress / status text

    def __init__(self, main_app: MainApp):
        super().__init__()
        self.main_app = main_app

    def run(self):
        # emit a first message
        self.message.emit("Processing started …")

        try:
            for _ in self.main_app.run(action="Processing"):
                # if your generator can yield progress you may emit it here
                pass
        except Exception as e:
            self.message.emit(f"Error: {e}")
        else:
            self.message.emit("Processing finished.")
        finally:
            self.finished.emit()


class MainWindow(QMainWindow, LoggerMixin):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Acquisition App")
        self.setGeometry(100, 100, 1250, 900)

        self.canvas = MyMplCanvas()

        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.createButtonLayout())
        self.splitter.addWidget(self.canvas)

        layout = QVBoxLayout()
        layout.addWidget(self.splitter)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

        self.createMenuBar()

        self.app = MainApp()

    def createButtonLayout(self):
        button_widget = QWidget()
        vbox = QVBoxLayout(button_widget)

        self.acquisitionGroup = QGroupBox("Acquisition")
        self.acquisitionLayout = QVBoxLayout()

        self.acquire_button = self.createButton("Preview Images", self.start_preview)
        self.acquisitionLayout.addWidget(self.acquire_button)

        self.stop_button = self.createButton("Stop Preview", self.stop_preview)
        self.acquisitionLayout.addWidget(self.stop_button)

        self.acquisitionGroup.setLayout(self.acquisitionLayout)

        self.processingGroup = QGroupBox("Processing")
        self.processingLayout = QVBoxLayout()

        self.start_process_button = self.createButton(
            "Start Processing", self.start_processing
        )
        self.processingLayout.addWidget(self.start_process_button)

        self.process_capture_file_button = self.createButton(
            "Process Capture File", self.start_process_file
        )
        self.processingLayout.addWidget(self.process_capture_file_button)

        self.config_button = self.createButton("Configuration", self.openConfigWindow)
        self.processingLayout.addWidget(self.config_button)

        self.close_button = self.createButton("Close", self.close_app)
        self.processingLayout.addWidget(self.close_button)

        self.processingGroup.setLayout(self.processingLayout)

        vbox.addWidget(self.acquisitionGroup)
        vbox.addWidget(self.processingGroup)

        return button_widget

    def openConfigWindow(self):
        self.config_window = ConfigWindow("./config/config.json")
        self.config_window.show()

    def createButton(self, text, slot):
        button = QPushButton(text)
        button.clicked.connect(slot)
        button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        button.setMinimumSize(100, 50)
        button.setMaximumSize(150, 50)
        return button

    def createMenuBar(self):
        menuBar = self.menuBar()

        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)

        exitAction = QAction("E&xit", self)
        exitAction.triggered.connect(self.close_app)
        fileMenu.addAction(exitAction)

    def start_processing(self):
        # Disable buttons so the user can’t start two jobs
        self.start_process_button.setEnabled(False)
        self.acquire_button.setEnabled(False)

        # Create worker + thread
        self.thread  = QThread(self)
        self.worker  = ProcessingWorker(self.app)
        self.worker.moveToThread(self.thread)

        # Wire signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Show status messages
        self.worker.message.connect(self.statusBar.showMessage)

        # Re-enable buttons when done
        self.worker.finished.connect(
            lambda: self.start_process_button.setEnabled(True))
        self.worker.finished.connect(
            lambda: self.acquire_button.setEnabled(True))

        # Start
        self.thread.start()

    def start_preview(self):
        result_gen = self.app.run(action="Previewing")
        try:
            result = next(result_gen)
        except StopIteration:
            self.logger.info("Stop Preview - StopIteration")

    def stop_preview(self):
        if self.app.acquisition is not None:
            self.app.acquisition.continue_recording = False
            self.app.running = False
        else:
            self.logger.error("Acquisition instance not initialized or not found.")

    def start_process_file(self):
        self.process_h5_form = ProcessH5Form()
        self.process_h5_form.show()

    def closeEvent(self, event):
        self.close_app()

    def close_app(self):
        self.stop_preview()
        self.close()
        QApplication.quit()
        self.logger.info("Close the App Successful")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
