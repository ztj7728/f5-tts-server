import argparse
import gc
import logging
import queue
import re
import socket
import struct
import threading
import traceback
import os
import json
import tempfile
import importlib.resources

import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    chunk_text,
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

# --- Configuration ---
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "F5TTS_v1_Base", "model_1250000.safetensors")
DEFAULT_VOCODER_PATH = os.path.join(BASE_DIR, "checkpoints", "vocos-mel-24khz")
MODEL_CKPT_PATH = os.environ.get("F5_TTS_MODEL_PATH", DEFAULT_MODEL_PATH)
VOCODER_PATH = os.environ.get("F5_TTS_VOCODER_PATH", DEFAULT_VOCODER_PATH)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Device Configuration ---
def get_device():
    """Determines the appropriate device for inference and prints feedback."""
    device_env = os.environ.get("DEVICE", "auto").lower()
    if device_env == "auto":
        if torch.cuda.is_available():
            logger.info("CUDA device found. Using GPU for acceleration.")
            return "cuda", torch.float16
        else:
            logger.warning("----------------------------------------------------------")
            logger.warning("WARNING: CUDA device not found.")
            logger.warning("Falling back to CPU. Performance will be much slower.")
            logger.warning("----------------------------------------------------------")
            return "cpu", torch.float32
    elif device_env == "cuda":
        if torch.cuda.is_available():
            logger.info("Forcing CUDA device as per environment variable.")
            return "cuda", torch.float16
        else:
            logger.error("FATAL: DEVICE is set to 'cuda' but no CUDA device is available.")
            logger.error("Please check your NVIDIA drivers and CUDA installation.")
            raise RuntimeError("CUDA not available, but was explicitly requested.")
    else:
        logger.info("Using CPU as per environment variable.")
        return "cpu", torch.float32

DEVICE, DTYPE = get_device()

# Special marker to signal the end of a single synthesis job (e.g., one sentence)
JOB_END = object()

class PreloadedModels:
    """Loads and holds the core TTS models in memory for maximum performance."""
    def __init__(self, model_name, ckpt_file, vocab_file, vocoder_path, device, dtype):
        logger.info(f"Loading core TTS models into memory on {device} with {dtype}...")
        self.device = device
        self.dtype = dtype
        
        model_cfg = OmegaConf.load(str(importlib.resources.files("f5_tts").joinpath(f"configs/{model_name}.yaml")))
        self.model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model_arc = model_cfg.model.arch
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate

        self.model = load_model(
            self.model_cls, self.model_arc, ckpt_path=ckpt_file,
            mel_spec_type=self.mel_spec_type, vocab_file=vocab_file,
            ode_method="euler", use_ema=True, device=self.device,
        ).to(self.device, dtype=self.dtype)
        
        self.vocoder = load_vocoder(
            vocoder_name=self.mel_spec_type, is_local=True, 
            local_path=vocoder_path, device=self.device
        )
        
        if self.device == 'cuda':
            logger.info("Compiling model for PyTorch 2.x for further speedup...")
            self.model = torch.compile(self.model)

        logger.info("Core models loaded and compiled successfully.")


class TTSClientSession:
    """
    Manages a single client's session using a high-performance, perpetual pipeline
    with dedicated threads for inference and network sending.
    """
    def __init__(self, preloaded_models, ref_audio_path, ref_text, client_socket):
        self.models = preloaded_models
        self.device = self.models.device
        self.client_socket = client_socket
        
        self.AUDIO_BUFFER_SECONDS = 1.5
        self.AUDIO_BUFFER_THRESHOLD = int(self.models.sampling_rate * 4 * self.AUDIO_BUFFER_SECONDS)

        self.is_running = False
        self.stop_requested = threading.Event()
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.inference_thread = None
        self.send_thread = None

        self.update_reference(ref_audio_path, ref_text)
        self._warm_up()

    def update_reference(self, ref_audio_path, ref_text):
        """Processes and sets the reference audio for this specific session."""
        self.ref_audio_path, self.ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        self.audio, self.sr = torchaudio.load(self.ref_audio_path)
        
        ref_audio_duration = self.audio.shape[-1] / self.sr
        ref_text_byte_len = len(self.ref_text.encode("utf-8"))
        base_chars = ref_text_byte_len / ref_audio_duration * (25 - ref_audio_duration)
        self.max_chars = int(base_chars)
        self.few_chars = int(base_chars / 2)

    @torch.inference_mode()
    def _warm_up(self):
        """Warms up the model with this session's specific reference audio."""
        logger.info("Warming up the model for the new session...")
        gen_text = "Warm-up text."
        with torch.amp.autocast("cuda", enabled=(self.device == 'cuda')):
            for _ in infer_batch_process(
                (self.audio, self.sr), self.ref_text, [gen_text],
                self.models.model, self.models.vocoder, progress=None, 
                device=self.device, streaming=True,
            ):
                pass
        logger.info("Session warm-up completed.")

    def start(self):
        """Starts the perpetual inference and sending threads."""
        if self.is_running: return
        logger.info("Starting session threads...")
        self.is_running = True
        self.inference_thread = threading.Thread(target=self._inference_worker)
        self.send_thread = threading.Thread(target=self._send_worker)
        self.inference_thread.daemon = True
        self.send_thread.daemon = True
        self.inference_thread.start()
        self.send_thread.start()
        logger.info("Session threads started.")

    def stop(self):
        """Stops the perpetual threads and cleans up."""
        if not self.is_running: return
        logger.info("Stopping session threads...")
        self.is_running = False
        self.stop_requested.set()  # Ensure all loops are aware of the stop
        self.text_queue.put(None)  # Unblock inference worker
        self.audio_queue.put(None) # Unblock send worker
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2)
        if self.send_thread and self.send_thread.is_alive():
            self.send_thread.join(timeout=2)
        logger.info("Session threads stopped.")

    def add_text_to_queue(self, payload):
        """Adds a text payload from the client to the processing queue."""
        if self.is_running:
            self.text_queue.put(payload)

    def prepare_for_new_job(self):
        """Resets the session state for a new synthesis job by clearing the stop flag."""
        logger.info("Preparing for new job, clearing stop request flag.")
        self.stop_requested.clear()

    def request_stop(self):
        """
        A thread-safe method to request the immediate stop of the current job.
        This works by setting a stop flag, clearing pending work, and then
        queueing a JOB_END marker to signal the send_worker.
        """
        if not self.stop_requested.is_set():
            logger.info("Stop requested. Clearing queues and signaling job end.")
            self.stop_requested.set()

            # Clear any unprocessed text to prevent the inference worker from starting new work.
            while not self.text_queue.empty():
                try:
                    self.text_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Clear any audio chunks that have been generated but not yet sent.
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Signal the sender thread that the job is over.
            # This will cause it to send the b"END" marker to the client.
            self.audio_queue.put(JOB_END)
            logger.info("Stop signal processed. JOB_END queued.")

    @torch.inference_mode()
    def _inference_worker(self):
        """
        The perpetual, GPU-bound worker. Pulls text from the central queue,
        runs inference, and puts the resulting audio into the central audio queue.
        """
        text_buffer = ""
        last_speed = 1.0
        last_chunk_size = 2048

        while self.is_running:
            request = self.text_queue.get()
            if request is None:
                self.is_running = False
                break

            # If a stop was requested while we were waiting for text,
            # discard the text we just received and reset the buffer.
            if self.stop_requested.is_set():
                logger.info("Inference worker: Stop detected after getting new text. Discarding.")
                text_buffer = ""
                continue
            
            text = request.get("text")
            if text == "__FLUSH_AUDIO__":
                logger.info("Flush signal received. Processing remaining buffer.")
                if text_buffer:
                    self._run_inference_on_batch(text_buffer, last_speed, last_chunk_size)
                    text_buffer = ""
                self.audio_queue.put(JOB_END) # Signal end of this specific job
                continue

            text_buffer += text
            last_speed = request["speed"]
            last_chunk_size = request.get("chunk_size", 2048)
            
            # Use regex to split sentences after punctuation, keeping the punctuation.
            # This ensures we only process complete sentences.
            sentences = re.split(r'(?<=[，。？！,?!.\n])', text_buffer)
            
            # The last element of the split might be an incomplete sentence.
            # If it is, we put it back in the buffer for the next round.
            text_buffer = ""
            if sentences and sentences[-1] and sentences[-1][-1] not in "，。？！,?!.\n":
                text_buffer = sentences.pop()

            # If we have complete sentences, process them as a single batch.
            if sentences:
                full_batch_text = "".join(sentences)
                self._run_inference_on_batch(full_batch_text, last_speed, last_chunk_size)

        if text_buffer:
            self._run_inference_on_batch(text_buffer, last_speed, last_chunk_size)
        
        self.audio_queue.put(None) # Signal end of entire session
        logger.info("Inference worker finished.")

    def _run_inference_on_batch(self, text, speed, chunk_size):
        """Helper function to run inference and put audio chunks into the central queue."""
        if not text.strip(): return
        logger.info(f"Synthesizing batch: '{text.strip()}'")
        text_batches = chunk_text(text, max_chars=self.max_chars)
        if text_batches:
            first_batch = text_batches.pop(0)
            appetizer_batches = chunk_text(first_batch, max_chars=self.few_chars)
            text_batches = appetizer_batches + text_batches

        with torch.amp.autocast("cuda", enabled=(self.device == 'cuda')):
            audio_stream = infer_batch_process(
                (self.audio, self.sr), self.ref_text, text_batches,
                self.models.model, self.models.vocoder, progress=None,
                device=self.device, streaming=True, chunk_size=chunk_size, speed=speed
            )
            for audio_chunk, _ in audio_stream:
                if not self.is_running or self.stop_requested.is_set():
                    if self.stop_requested.is_set():
                        logger.info("Inference for batch interrupted by stop signal inside generation loop.")
                    break
                if len(audio_chunk) > 0:
                    clipped_chunk = np.clip(audio_chunk, -1.0, 1.0)
                    self.audio_queue.put(clipped_chunk)

    def _send_worker(self):
        """
        The perpetual, Network-bound worker. A simplified, stateless loop that
        handles audio chunks and job control signals.
        """
        while self.is_running:
            try:
                # Block here waiting for ANY item from the inference worker.
                chunk = self.audio_queue.get(block=True)

                if chunk is None:  # End of session signal
                    self.is_running = False
                    break

                # If it's the end-of-job marker (sent on natural completion or manual stop)
                if chunk is JOB_END:
                    logger.info("Job finished or stopped. Sending END marker to client.")
                    self.client_socket.sendall(b"END")
                    continue  # Loop back to wait for the next job

                # If we get here, it's an audio chunk.
                # If a stop was requested, we are in "draining" mode and discard it.
                if self.stop_requested.is_set():
                    continue

                # Otherwise, send the audio chunk to the client.
                self.client_socket.sendall(chunk.astype(np.float32).tobytes())

            except (socket.error, BrokenPipeError):
                logger.warning("Client disconnected during stream.")
                self.is_running = False
                break  # Exit the main while loop
            except Exception as e:
                logger.error(f"Send worker error: {e}", exc_info=True)
                self.is_running = False
                break
        
        logger.info("Send worker finished.")


def recv_all(conn, length):
    data = b""
    while len(data) < length:
        packet = conn.recv(length - len(data))
        if not packet: return None
        data += packet
    return data

def handle_client(conn, addr, preloaded_models):
    ref_audio_file, session = None, None
    try:
        with conn:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            logger.info(f"Handling new client from {addr}")

            header_len_bytes = recv_all(conn, 4)
            if not header_len_bytes: return
            header_len = struct.unpack('!I', header_len_bytes)[0]
            header_bytes = recv_all(conn, header_len)
            if not header_bytes: return
            header = json.loads(header_bytes.decode('utf-8'))
            
            ref_audio_bytes = recv_all(conn, header['ref_audio_len'])
            if not ref_audio_bytes: return

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                fp.write(ref_audio_bytes)
                ref_audio_file = fp.name
            
            session = TTSClientSession(preloaded_models, ref_audio_file, header['ref_text'], conn)
            session.start()
            conn.sendall(b"READY")
            logger.info(f"Session ready for {addr}. Now handling synthesis jobs.")

            while True:
                msg_len_bytes = recv_all(conn, 4)
                if not msg_len_bytes:
                    logger.info(f"Client {addr} disconnected gracefully (socket closed).")
                    break
                
                msg_len = struct.unpack('!I', msg_len_bytes)[0]
                payload_bytes = recv_all(conn, msg_len)
                if not payload_bytes:
                    logger.info(f"Client {addr} disconnected gracefully.")
                    break
                
                payload = json.loads(payload_bytes.decode("utf-8"))
                
                # Check for control signals before queueing text
                if payload.get("is_new_job"):
                    logger.info(f"Received new job signal from {addr}.")
                    if session:
                        session.prepare_for_new_job()

                if payload.get("text") == "__STOP_GENERATION__":
                    logger.info(f"Received __STOP_GENERATION__ command from {addr}.")
                    if session:
                        session.request_stop()
                elif "text" in payload: # Only queue if there's a text key.
                    logger.info(f"Received text from {addr}: '{payload.get('text', '')[:50]}...'")
                    if session:
                        session.add_text_to_queue(payload)

    except (socket.error, ConnectionResetError) as e:
        logger.warning(f"Connection with {addr} lost: {e}")
    except Exception as e:
        logger.error(f"Error handling client {addr}: {e}")
        traceback.print_exc()
    finally:
        if session:
            session.stop()
        if ref_audio_file and os.path.exists(ref_audio_file): os.remove(ref_audio_file)
        logger.info(f"Connection with {addr} closed and resources cleaned up.")


def start_server(host, port, preloaded_models):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen()
        logger.info(f"Server started on {host}:{port}")
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr, preloaded_models))
            client_thread.daemon = True
            client_thread.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7300)
    args = parser.parse_args()

    try:
        models = PreloadedModels(
            model_name="F5TTS_v1_Base",
            ckpt_file=MODEL_CKPT_PATH,
            vocab_file="",
            vocoder_path=VOCODER_PATH,
            device=DEVICE,
            dtype=DTYPE,
        )
        start_server(args.host, args.port, models)
    except KeyboardInterrupt:
        logger.info("Server shutting down.")
        gc.collect()