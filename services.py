import socket
import struct
import json
import os
import pyaudio
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from openai import OpenAI
import threading

class TTSWorker(QObject):
    """Handles the connection and communication with the TTS server."""
    status_update = pyqtSignal(str)
    error = pyqtSignal(str)
    connection_established = pyqtSignal()
    connection_lost = pyqtSignal()
    job_finished = pyqtSignal()

    def __init__(self, host=None, port=None, ref_audio=None, ref_text=None):
        super().__init__()
        self.host, self.port, self.ref_audio, self.ref_text = host, port, ref_audio, ref_text
        self.client_socket, self.is_running = None, False
        self.pyaudio_instance, self.pyaudio_stream, self.audio_buffer = None, None, b''
        self.is_stopping = False
        self.network_thread = None
    
    def customEvent(self, event):
        """Handles custom events posted from other threads."""
        if event.type() == event.Connect:
            self.connect_to_server()
        elif event.type() == event.Disconnect:
            self.stop()

    def connect_to_server(self):
        if self.is_running:
            return
        self.is_running = True
        try:
            if not os.path.exists(self.ref_audio):
                raise FileNotFoundError(f"Reference audio file not found: {self.ref_audio}")
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.status_update.emit(f"Connected to {self.host}:{self.port}")
            self._send_reference_data()
            ready_signal = self.client_socket.recv(5)
            if ready_signal != b"READY":
                raise ConnectionAbortedError(f"Server not ready. Received: {ready_signal}")
            self.status_update.emit("Server is ready, you can start sending text.")
            self.connection_established.emit()
            
            self.network_thread = threading.Thread(target=self._listen_for_data)
            self.network_thread.daemon = True
            self.network_thread.start()
        except Exception as e:
            self.error.emit(f"Connection failed: {e}")
            self.stop()

    def _send_reference_data(self):
        with open(self.ref_audio, 'rb') as f:
            ref_audio_bytes = f.read()
        header = {'ref_text': self.ref_text, 'ref_audio_len': len(ref_audio_bytes)}
        header_bytes = json.dumps(header).encode('utf-8')
        self.client_socket.sendall(struct.pack('!I', len(header_bytes)))
        self.client_socket.sendall(header_bytes)
        self.client_socket.sendall(ref_audio_bytes)
        self.status_update.emit("Reference data sent, waiting for server response...")

    def _listen_for_data(self):
        """
        The core network listener loop. This is the definitive, correct version.
        It robustly handles the network buffer to ensure the "END" marker is never
        accidentally discarded, which was the root cause of the UI state bug.
        """
        self._init_pyaudio()
        network_buffer = b''
        while self.is_running:
            try:
                data = self.client_socket.recv(8192)
                if not data:
                    break  # Connection closed by server

                network_buffer += data

                # Process all complete jobs (demarcated by "END") in the buffer.
                while b"END" in network_buffer:
                    end_index = network_buffer.find(b"END")
                    job_data = network_buffer[:end_index]
                    
                    # The remainder of the buffer belongs to the next job.
                    network_buffer = network_buffer[end_index + 3:]

                    # A job has officially finished. Process its final audio chunk.
                    if not self.is_stopping:
                        self._play_audio_chunk(job_data)
                    
                    # Flush the audio player's internal buffer to play any remaining samples.
                    if self.audio_buffer:
                        padding = 4 - (len(self.audio_buffer) % 4)
                        if padding != 4: self._play_audio_chunk(b'\x00' * padding)

                    # The job is over. Signal the UI, reset state, and close the stream for this job.
                    self.job_finished.emit()
                    self.is_stopping = False
                    self._close_pyaudio_stream()

                # After the loop, any data left in the buffer is an intermediate chunk
                # from a job still in progress.
                if network_buffer:
                    if not self.is_stopping:
                        # If we're not stopping, play it.
                        self._play_audio_chunk(network_buffer)
                    
                    # The buffer must be cleared to prevent deadlocks.
                    # If we played the chunk, the data is now safe in the audio player's buffer.
                    # If we were stopping, this correctly discards the unwanted audio.
                    network_buffer = b''

            except (socket.error, OSError):
                break  # Connection lost or closed
            except Exception as e:
                if self.is_running:
                    self.error.emit(f"Error in network listener: {e}")
                break
        
        self.connection_lost.emit()

    def send_text_to_server(self, text, speed, chunk_size):
        if self.client_socket and self.is_running:
            try:
                payload = {"text": text, "speed": speed, "chunk_size": chunk_size}
                payload_bytes = json.dumps(payload).encode('utf-8')
                # Pack the length of the payload as a 4-byte unsigned integer and send it first
                self.client_socket.sendall(struct.pack('!I', len(payload_bytes)))
                # Send the actual payload
                self.client_socket.sendall(payload_bytes)
                self.status_update.emit(f"Sent: {text[:20]}... (Speed: {speed}x, Chunk: {chunk_size})")
            except Exception as e:
                self.error.emit(f"Error sending text: {e}")
                self.stop()

    def signal_new_job(self):
        """Signals the server to reset its state for a new job."""
        if self.client_socket and self.is_running:
            try:
                payload = {"is_new_job": True}
                payload_bytes = json.dumps(payload).encode('utf-8')
                self.client_socket.sendall(struct.pack('!I', len(payload_bytes)))
                self.client_socket.sendall(payload_bytes)
            except Exception as e:
                self.error.emit(f"Error signaling new job: {e}")

    def stop_generation(self):
        """Stops the current audio playback and signals the server to stop generating."""
        self.is_stopping = True # Set the flag to start ignoring audio chunks.
        
        if self.pyaudio_stream:
            self.pyaudio_stream.stop_stream() # Stop playback immediately.
        
        self.audio_buffer = b'' # Clear any buffered audio.

        # Signal the server to stop the current job
        if self.client_socket and self.is_running:
            try:
                payload = {"text": "__STOP_GENERATION__"}
                payload_bytes = json.dumps(payload).encode('utf-8')
                self.client_socket.sendall(struct.pack('!I', len(payload_bytes)))
                self.client_socket.sendall(payload_bytes)
                self.status_update.emit("Sent stop signal to server.")
            except Exception as e:
                self.error.emit(f"Error sending stop signal: {e}")

    def _init_pyaudio(self):
        if self.pyaudio_instance is None:
            self.pyaudio_instance = pyaudio.PyAudio()
        
        # Always open a new stream. This is called when a new job's audio arrives.
        # First, ensure any previous stream is properly closed.
        self._close_pyaudio_stream()
        self.pyaudio_stream = self.pyaudio_instance.open(format=pyaudio.paFloat32, channels=1, rate=24000, output=True)

    def _play_audio_chunk(self, chunk):
        """
        A stateful audio player that uses an internal buffer to ensure
        that only perfectly aligned 4-byte chunks are sent to the audio device.
        It also handles re-initializing the audio stream if it was previously closed.
        """
        if not chunk:
            return

        try:
            # If the stream is gone or was stopped, re-initialize it for the new job.
            if self.pyaudio_stream is None or not self.pyaudio_stream.is_active():
                self._init_pyaudio()

            self.audio_buffer += chunk
            
            # Calculate how many full 4-byte chunks we have
            num_bytes = (len(self.audio_buffer) // 4) * 4
            
            if num_bytes > 0:
                # Extract the playable data and leave the remainder in the buffer
                data_to_play = self.audio_buffer[:num_bytes]
                self.audio_buffer = self.audio_buffer[num_bytes:]
                
                audio_array = np.frombuffer(data_to_play, dtype=np.float32)
                if self.pyaudio_stream and self.pyaudio_stream.is_active():
                    self.pyaudio_stream.write(audio_array.tobytes())

        except Exception as e:
            self.error.emit(f"PyAudio playback error: {e}")

    def stop(self):
        """Stops the connection and cleans up all resources."""
        if not self.is_running:
            return
        self.is_running = False
        
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass # Ignore if socket is already closed
            self.client_socket.close()
            self.client_socket = None

        if self.network_thread and self.network_thread.is_alive():
            self.network_thread.join(timeout=1.0)
        self.network_thread = None

        self._close_pyaudio_stream()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
        
        self.audio_buffer = b''
        self.status_update.emit("Disconnected.")
        self.connection_lost.emit() # Ensure the UI always cleans up.

    def _close_pyaudio_stream(self):
        """Safely closes and releases the PyAudio stream."""
        if self.pyaudio_stream:
            try:
                if self.pyaudio_stream.is_active():
                    self.pyaudio_stream.stop_stream()
                self.pyaudio_stream.close()
            except Exception:
                # Ignore errors during cleanup
                pass
            finally:
                self.pyaudio_stream = None

class OpenAIWorker(QObject):
    """Handles streaming text from an OpenAI-compatible API."""
    text_chunk_received = pyqtSignal(str)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, base_url=None, api_key=None, model_name=None):
        super().__init__()
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.is_running = False
        self.client = None

    def start_streaming(self, prompt):
        self.is_running = True
        try:
            # If the api_key is empty, use a placeholder as the library requires a non-empty string.
            api_key = self.api_key if self.api_key else "none"
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            for chunk in stream:
                if not self.is_running:
                    break
                content = chunk.choices[0].delta.content
                if content:
                    self.text_chunk_received.emit(content)
        except Exception as e:
            self.error.emit(f"OpenAI streaming error: {e}")
        finally:
            self.is_running = False
            self.finished.emit()

    def stop(self):
        self.is_running = False