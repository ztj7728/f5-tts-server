import sys
from PyQt6.QtWidgets import QApplication
from ui import TTSClientGUI

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Load the stylesheet
    try:
        with open('style.qss', 'r') as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        print("Stylesheet 'style.qss' not found. Running with default styles.")
    
    # Create and show the main window
    gui = TTSClientGUI()
    gui.show()
    
    # Start the application event loop
    sys.exit(app.exec())