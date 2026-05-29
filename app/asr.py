import openai
import os
import tempfile
from pydub import AudioSegment

openai.api_key = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
client = openai.OpenAI(api_key=openai.api_key, base_url=OPENAI_BASE_URL)


async def transcribe_voice(audio_bytes: bytes, filename: str = "voice.ogg") -> str:
    """
    语音转文字，使用 Whisper API。
    Lark 语音格式可能是 aac/mp4/amr，先尝试直接传给 Whisper。
    """
    suffix = filename.split(".")[-1] if "." in filename else "ogg"
    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        with open(tmp_path, "rb") as audio_file:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh",  # 中文优先
            )
        return resp.text
    except Exception as e:
        # 如果格式不支持，尝试用 pydub 转 wav
        try:
            sound = AudioSegment.from_file(tmp_path)
            wav_path = tmp_path + ".wav"
            sound.export(wav_path, format="wav")
            with open(wav_path, "rb") as audio_file:
                resp = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="zh",
                )
            return resp.text
        except Exception as e2:
            raise Exception(f"ASR failed: {e}; retry failed: {e2}")
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except:
            pass
