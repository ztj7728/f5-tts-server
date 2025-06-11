# Default configuration settings

# TTS Server Settings
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 7300
DEFAULT_REF_AUDIO = 'test.wav'
DEFAULT_REF_TEXT = '贝加尔湖是世界上最古老、最深的淡水湖泊，位于俄罗斯西伯利亚地区， 湖水极其清澈透明，是世界上最纯净的湖泊之一。'

# UI Settings
CHUNK_SIZES = [32, 64, 128, 512, 1024, 2048]
DEFAULT_CHUNK_SIZE = 64

# OpenAI API Settings
DEFAULT_API_BASE = "http://127.0.0.1:11434/v1"
# --- Custom Event Types ---
# Used for safe, cross-thread communication with the TTSWorker
from PyQt6.QtCore import QEvent

class ConnectEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 1))

class DisconnectEvent(QEvent):
    def __init__(self):
        super().__init__(QEvent.Type(QEvent.Type.User + 2))

# Add the event types to QEvent for easy access
QEvent.Connect = QEvent.Type.User + 1
QEvent.Disconnect = QEvent.Type.User + 2