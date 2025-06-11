import sys
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit,
                             QFileDialog, QMessageBox, QSlider, QSizeGrip, QFrame, QStyle)
from PyQt6.QtCore import pyqtSignal, QThread, Qt, QPoint
from PyQt6.QtGui import QMouseEvent, QIcon

import config
from services import TTSWorker, OpenAIWorker

class ClickableLabel(QLabel):
    """A QLabel that emits a 'clicked' signal when pressed."""
    clicked = pyqtSignal()
    def __init__(self, text, parent=None, objectName=""):
        super().__init__(text, parent)
        if objectName:
            self.setObjectName(objectName)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    def mousePressEvent(self, event: QMouseEvent):
        self.clicked.emit()

class CustomTitleBar(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("TitleBar")
        layout = QHBoxLayout()
        layout.setContentsMargins(15, 8, 15, 0)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        self.title_label = QLabel("F5 TTS & LLM Client", self, objectName="TitleLabel")
        layout.addWidget(self.title_label)
        
        layout.addStretch()

        self.minimize_button = ClickableLabel("—", self, objectName="MinimizeButton")
        self.close_button = ClickableLabel("✕", self, objectName="CloseButton")
        
        self.minimize_button.clicked.connect(parent.showMinimized)
        self.close_button.clicked.connect(parent.close)

        layout.addWidget(self.minimize_button)
        layout.addWidget(self.close_button)
        self.setLayout(layout)
        self.start_move_pos = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.start_move_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.start_move_pos:
            delta = event.globalPosition().toPoint() - self.start_move_pos
            self.parent.move(self.parent.pos() + delta)
            self.start_move_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent): self.start_move_pos = None

class TTSClientGUI(QWidget):
    # --- Custom Signals ---
    # Signal to start the LLM streaming process with a given prompt
    startLlmStream = pyqtSignal(str)
    # Signal to request the immediate stop of any ongoing generation
    stopGeneration = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.llm_stream_finished = False
        self.is_job_running = False
        self.is_stopping = False
        
        self._setup_workers()
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle('F5 TTS & LLM Client')
        
        style = self.style()
        icon = style.standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setWindowIcon(icon)

        self.background_frame = QFrame(self, objectName="BackgroundFrame")
        self.main_layout = QVBoxLayout(self.background_frame)
        self.main_layout.setContentsMargins(1, 1, 1, 1); self.main_layout.setSpacing(0)
        
        outer_layout = QVBoxLayout(self); outer_layout.setContentsMargins(0,0,0,0)
        outer_layout.addWidget(self.background_frame)
        
        self.title_bar = CustomTitleBar(self)
        self.main_layout.addWidget(self.title_bar)

        content_widget = QWidget()
        self.init_ui(content_widget)
        self.main_layout.addWidget(content_widget)
        
        grip = QSizeGrip(self)
        self.main_layout.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

    def init_ui(self, parent_widget):
        layout = QVBoxLayout(parent_widget)
        layout.setSpacing(15); layout.setContentsMargins(20, 20, 20, 20)

        # --- TTS Server Settings ---
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel('服务器地址:')); self.host_input = QLineEdit(config.DEFAULT_HOST)
        server_layout.addWidget(self.host_input); server_layout.addWidget(QLabel('端口:'))
        self.port_input = QLineEdit(str(config.DEFAULT_PORT)); self.port_input.setFixedWidth(60)
        server_layout.addWidget(self.port_input); layout.addLayout(server_layout)

        ref_audio_layout = QHBoxLayout()
        ref_audio_layout.addWidget(QLabel('参考音频:')); self.ref_audio_input = QLineEdit(config.DEFAULT_REF_AUDIO)
        ref_audio_layout.addWidget(self.ref_audio_input)
        self.browse_button = QPushButton('浏览'); self.browse_button.clicked.connect(self.browse_file)
        ref_audio_layout.addWidget(self.browse_button); layout.addLayout(ref_audio_layout)
        
        ref_text_layout = QHBoxLayout()
        ref_text_layout.addWidget(QLabel('参考文本:')); self.ref_text_input = QLineEdit(config.DEFAULT_REF_TEXT)
        ref_text_layout.addWidget(self.ref_text_input); layout.addLayout(ref_text_layout)

        self.connect_button = QPushButton('连接TTS服务器'); self.connect_button.clicked.connect(self.toggle_tts_connection)
        layout.addWidget(self.connect_button)

        # --- OpenAI Settings ---
        openai_layout = QVBoxLayout(); openai_layout.setSpacing(10)
        
        api_base_layout = QHBoxLayout()
        api_base_layout.addWidget(QLabel('OpenAI Base URL:')); self.api_base_input = QLineEdit(config.DEFAULT_API_BASE)
        api_base_layout.addWidget(self.api_base_input)
        openai_layout.addLayout(api_base_layout)

        api_key_layout = QHBoxLayout()
        api_key_layout.addWidget(QLabel('OpenAI API Key:')); self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        api_key_layout.addWidget(self.api_key_input)
        openai_layout.addLayout(api_key_layout)
        
        model_name_layout = QHBoxLayout()
        model_name_layout.addWidget(QLabel('Model Name:')); self.model_name_input = QLineEdit("qwen2.5:7b")
        model_name_layout.addWidget(self.model_name_input)
        openai_layout.addLayout(model_name_layout)
        
        layout.addLayout(openai_layout)

        # --- Text Input and Controls ---
        layout.addWidget(QLabel('LLM Prompt:'))
        self.text_input = QTextEdit(); self.text_input.setPlaceholderText('在此输入文本以发送给LLM...'); self.text_input.setFixedHeight(100)
        layout.addWidget(self.text_input)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel('语速:')); self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(5); self.speed_slider.setMaximum(20); self.speed_slider.setValue(10)
        speed_layout.addWidget(self.speed_slider); self.speed_label = QLabel('1.0x'); self.speed_label.setFixedWidth(40)
        self.speed_slider.valueChanged.connect(lambda v: self.speed_label.setText(f'{v/10.0:.1f}x'))
        speed_layout.addWidget(self.speed_label); layout.addLayout(speed_layout)

        chunk_size_layout = QHBoxLayout()
        chunk_size_layout.addWidget(QLabel('分段大小:')); self.chunk_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.chunk_size_slider.setMinimum(0); self.chunk_size_slider.setMaximum(len(config.CHUNK_SIZES) - 1)
        try: default_index = config.CHUNK_SIZES.index(config.DEFAULT_CHUNK_SIZE)
        except ValueError: default_index = 0
        self.chunk_size_slider.setValue(default_index); chunk_size_layout.addWidget(self.chunk_size_slider)
        self.chunk_size_label = QLabel(str(config.CHUNK_SIZES[default_index])); self.chunk_size_label.setFixedWidth(40)
        self.chunk_size_slider.valueChanged.connect(lambda v: self.chunk_size_label.setText(str(config.CHUNK_SIZES[v])))
        chunk_size_layout.addWidget(self.chunk_size_label); layout.addLayout(chunk_size_layout)

        self.send_button = QPushButton('发送到LLM并播报'); self.send_button.clicked.connect(self.do_send_text_to_llm)
        self.send_button.setEnabled(False)
        layout.addWidget(self.send_button)

        self.status_label = QLabel('状态: 未连接'); self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

    def browse_file(self):
        fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', "WAV 文件 (*.wav)")
        if fname: self.ref_audio_input.setText(fname)

    def _setup_workers(self):
        """Initializes and connects all worker threads and objects."""
        # --- TTS Worker Setup ---
        self.tts_worker_thread = QThread()
        self.tts_worker = TTSWorker(None, None, None, None) # Initialized empty
        self.tts_worker.moveToThread(self.tts_worker_thread)
        self.tts_worker.status_update.connect(self.update_status)
        self.tts_worker.error.connect(self.show_error)
        self.tts_worker.connection_established.connect(self.on_tts_connection_established)
        self.tts_worker.connection_lost.connect(self.on_tts_connection_lost)
        self.tts_worker.job_finished.connect(self.on_tts_job_finished)
        self.tts_worker_thread.start()

        # --- OpenAI Worker Setup ---
        self.openai_worker_thread = QThread()
        self.openai_worker = OpenAIWorker(None, None, None) # Initialized empty
        self.openai_worker.moveToThread(self.openai_worker_thread)
        self.openai_worker.text_chunk_received.connect(self.on_llm_chunk_received)
        self.openai_worker.finished.connect(self.on_llm_finished)
        self.openai_worker.error.connect(self.show_error)
        
        # --- Cross-thread Signal Connections ---
        self.startLlmStream.connect(self.openai_worker.start_streaming)
        self.stopGeneration.connect(self.openai_worker.stop)
        self.stopGeneration.connect(self.tts_worker.stop_generation)
        
        self.openai_worker_thread.start()

    def toggle_tts_connection(self):
        if not self.tts_worker.is_running:
            # Update worker with fresh data from UI before connecting
            self.tts_worker.host = self.host_input.text()
            self.tts_worker.port = int(self.port_input.text())
            self.tts_worker.ref_audio = self.ref_audio_input.text()
            self.tts_worker.ref_text = self.ref_text_input.text()
            
            # Use invokeMethod to call the slot on the worker's thread
            QApplication.instance().postEvent(self.tts_worker, config.ConnectEvent())
            self.connect_button.setText('连接中...'); self.connect_button.setEnabled(False)
        else:
            QApplication.instance().postEvent(self.tts_worker, config.DisconnectEvent())

    def on_tts_connection_established(self):
        self.connect_button.setText('断开TTS连接'); self.connect_button.setEnabled(True)
        self.send_button.setEnabled(True); self.set_tts_inputs_enabled(False)

    def on_tts_connection_lost(self):
        self.connect_button.setText('连接TTS服务器'); self.connect_button.setEnabled(True)
        self.send_button.setEnabled(False); self.set_tts_inputs_enabled(True)
        self.update_status('未连接')

    def set_tts_inputs_enabled(self, enabled):
        for w in [self.host_input, self.port_input, self.ref_audio_input, self.browse_button, self.ref_text_input]:
            w.setEnabled(enabled)

    def do_send_text_to_llm(self):
        if self.is_job_running:
            # Prevent user from clicking stop multiple times while a stop is in progress
            if not self.is_stopping:
                self.update_status("Stopping generation...")
                self.is_stopping = True
                self.stopGeneration.emit()
            return

        if self.tts_worker.is_running:
            prompt = self.text_input.toPlainText().strip()
            if not prompt:
                self.show_error("Prompt cannot be empty.")
                return
            
            # Update OpenAI worker with latest settings from UI
            self.openai_worker.base_url = self.api_base_input.text()
            self.openai_worker.api_key = self.api_key_input.text()
            self.openai_worker.model_name = self.model_name_input.text()
            if not self.openai_worker.model_name:
                self.show_error("Model Name cannot be empty.")
                return

            self.is_job_running = True
            self.is_stopping = False  # Reset for the new job
            self.llm_stream_finished = False
            self.send_button.setText("停止生成")
            self.text_input.clear()

            # Tell the server to reset its state for this new job.
            self.tts_worker.signal_new_job()
            
            # Emit signal to start the stream on the worker's thread
            self.startLlmStream.emit(prompt)

    def on_llm_chunk_received(self, chunk):
        """Directly forwards any received text chunk to the TTS worker."""
        # Prevent forwarding new chunks if a stop has been requested.
        if self.is_job_running and not self.is_stopping and self.tts_worker and chunk:
            speed = self.speed_slider.value() / 10.0
            chunk_size = config.CHUNK_SIZES[self.chunk_size_slider.value()]
            self.tts_worker.send_text_to_server(chunk, speed, chunk_size)

    def on_llm_finished(self):
        """Called when the LLM stream is complete."""
        # If a stop was requested, the stop signal has already been sent.
        # Don't send a flush command, just mark the stream as finished and exit.
        if self.is_stopping:
            self.llm_stream_finished = True
            self.update_status("LLM stream stopped, waiting for TTS to confirm.")
            return

        if self.tts_worker:
            self.tts_worker.send_text_to_server("__FLUSH_AUDIO__", 0, 0)
        
        self.llm_stream_finished = True
        self.update_status("LLM stream finished, waiting for TTS...")

    def on_tts_job_finished(self):
        """
        Called when the TTS server confirms a job is fully processed.
        This is the definitive signal to reset the UI to a ready state.
        """
        self.is_job_running = False
        self.is_stopping = False # A finished job can no longer be in a "stopping" state.
        self.llm_stream_finished = False
        self.send_button.setText("发送到LLM并播报")
        self.update_status("Job finished.")

    def update_status(self, message): self.status_label.setText(f'状态: {message}')
    def show_error(self, message): QMessageBox.critical(self, '错误', message)
    
    def closeEvent(self, event):
        # Gracefully shut down the worker threads
        self.tts_worker.stop()
        self.openai_worker.stop()
        self.tts_worker_thread.quit()
        self.tts_worker_thread.wait()
        self.openai_worker_thread.quit()
        self.openai_worker_thread.wait()
        event.accept()