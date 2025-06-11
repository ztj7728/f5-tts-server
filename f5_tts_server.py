import os
import uuid
import uvicorn
import logging
import requests
from urllib.parse import urlparse
from typing import Union, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from f5_tts.api import F5TTS

# Load environment variables from .env file
load_dotenv()

# --- Base Directory ---
# 获取脚本所在目录，确保所有相对路径都基于此
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Configuration ---
# Load from environment variables or .env file
SERVER_HOST = os.environ.get("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", 8000))
PUBLIC_HOSTNAME = os.environ.get("PUBLIC_HOSTNAME", f"http://127.0.0.1:{SERVER_PORT}")

# Model and device configuration
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "F5TTS_v1_Base", "model_1250000.safetensors")
DEFAULT_VOCODER_PATH = os.path.join(BASE_DIR, "checkpoints", "vocos-mel-24khz")
MODEL_CKPT_PATH = os.environ.get("F5_TTS_MODEL_PATH", DEFAULT_MODEL_PATH)
VOCODER_PATH = os.environ.get("F5_TTS_VOCODER_PATH", DEFAULT_VOCODER_PATH)
DEVICE = os.environ.get("DEVICE", "cuda")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="F5-TTS FastAPI Server",
    description="一个使用 F5-TTS 模型进行文本到语音合成的服务器。",
    version="1.0.0",
)

# --- Static File Serving ---
# Create a directory for generated audio files
PUBLIC_DIR = os.path.join(BASE_DIR, "generated_audio")
os.makedirs(PUBLIC_DIR, exist_ok=True)
# Mount the directory to serve files under the /audio path
app.mount("/audio", StaticFiles(directory=PUBLIC_DIR), name="audio")

# --- Global TTS Model ---
tts_model = None

@app.on_event("startup")
def load_model():
    """服务启动时加载 F5-TTS 模型。"""
    global tts_model
    logger.info("服务器启动，开始加载模型...")
    if not os.path.exists(MODEL_CKPT_PATH):
        logger.warning(f"未找到模型路径，将从 HF Hub 下载：{MODEL_CKPT_PATH}")
    if not os.path.exists(VOCODER_PATH):
        logger.warning(f"未找到声码器目录，将从 HF Hub 下载：{VOCODER_PATH}")
    try:
        logger.info(f"加载模型：{MODEL_CKPT_PATH}")
        logger.info(f"加载声码器：{VOCODER_PATH}")
        logger.info(f"使用设备：{DEVICE}")
        tts_model = F5TTS(
            ckpt_file=MODEL_CKPT_PATH,
            vocoder_local_path=VOCODER_PATH,
            device=DEVICE
        )
        logger.info("模型加载成功！")
    except Exception as e:
        logger.error(f"加载模型失败：{e}", exc_info=True)
        raise RuntimeError("无法加载 F5-TTS 模型，请检查路径和文件完整性。") from e

# --- Temporary File Storage for Reference Audio ---
TEMP_DIR = os.path.join(BASE_DIR, "temp_ref_audio")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Helper Functions ---
def _download_audio(url: str, save_path: str):
    """Downloads audio from a URL and saves it."""
    try:
        # Basic URL validation
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            raise ValueError("Invalid URL provided.")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Successfully downloaded audio from {url} to {save_path}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download audio from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {e}")
    except ValueError as e:
        logger.error(f"Invalid URL format: {url}: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- API Endpoints ---
@app.post("/synthesize/", response_class=JSONResponse)
async def synthesize_speech(
    ref_text: str = Form(..., description="参考音频中的转录文本。"),
    gen_text: str = Form(..., description="要合成的目标文本。"),
    ref_audio: Union[UploadFile, str] = Form(..., description="参考 WAV 音频文件（上传）或其 URL（字符串）。"),
    speed: float = Form(1.0, description="语速调节，大于 1 加快，小于 1 放慢。"),
    cfg_strength: float = Form(2.0, description="CFG 强度，控制与参考音频的相似度。"),
    nfe_step: int = Form(32, description="采样步数，影响生成质量和速度。"),
    seed: Optional[int] = Form(None, description="随机种子，用于复现结果。如果留空则随机。"),
):
    """
    使用参考音频和文本进行语音克隆并合成新语音。
    成功返回一个包含音频文件 URL 的 JSON 对象。
    """
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS 模型尚未加载。")

    ref_path = os.path.join(TEMP_DIR, f"ref_{uuid.uuid4()}.wav")
    
    # The generated file will be placed in the public directory
    output_filename = f"gen_{uuid.uuid4()}.wav"
    out_path = os.path.join(PUBLIC_DIR, output_filename)

    try:
        # Handle reference audio (URL or Upload)
        if isinstance(ref_audio, str):
            logger.info(f"从 URL 下载参考音频: {ref_audio}")
            _download_audio(ref_audio, ref_path)
        else:
            logger.info(f"从上传文件中保存参考音频: {ref_audio.filename}")
            with open(ref_path, "wb") as buf:
                buf.write(await ref_audio.read())

        # Perform inference
        logger.info(f"合成文本：{gen_text[:50]}... (语速: {speed}, CFG: {cfg_strength}, 步数: {nfe_step}, 种子: {seed})")
        tts_model.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            file_wave=out_path,
            speed=speed,
            cfg_strength=cfg_strength,
            nfe_step=nfe_step,
            seed=seed,
        )
        
        # Construct the public URL for the generated file
        audio_url = f"{PUBLIC_HOSTNAME.rstrip('/')}/audio/{output_filename}"
        logger.info(f"合成完成，音频 URL: {audio_url}")

        # Return the URL in a JSON response
        return JSONResponse(content={"audio_url": audio_url, "seed": tts_model.seed})

    except Exception as e:
        # Re-raise HTTP exceptions or wrap others
        if isinstance(e, HTTPException):
            raise
        else:
            logger.error(f"合成错误：{e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"合成失败：{e}")
    finally:
        # Clean up the temporary reference file
        if os.path.exists(ref_path):
            os.remove(ref_path)

@app.get("/", summary="状态检查")
def read_root():
    return {"message": "F5-TTS FastAPI 服务运行中。"}

# --- Main ---
if __name__ == "__main__":
    print("--- F5-TTS FastAPI 服务器 ---")
    print(f"模型路径: {MODEL_CKPT_PATH}")
    print(f"声码器路径: {VOCODER_PATH}")
    print(f"公开访问地址: {PUBLIC_HOSTNAME}")
    print(f"服务器监听地址: http://{SERVER_HOST}:{SERVER_PORT}")
    print("---")
    print("要修改配置，请编辑 .env 文件。")
    
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
