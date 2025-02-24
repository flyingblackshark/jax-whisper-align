import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse,JSONResponse, StreamingResponse
from gradio_client import Client, handle_file
app = FastAPI()
transcribe_client = Client("http://localhost:7860")

@app.post("/api/transcribe")
async def upload_transcribe_audio(request: Request,
                                  audio: UploadFile = File(...)):
    audio_path = f"static/{audio.filename}"
    with open(audio_path, "wb+") as fp:
        fp.write(audio.file.read())
    try:
        transcript,csv_file,srt_file = transcribe_client.predict(handle_file(audio_path))
    finally:
        os.remove(audio_path)
    with open(csv_file, "r", encoding="utf-8") as csv_fp:
        with open(srt_file, "r", encoding="utf-8") as srt_fp:
            data = {
                "transcript": transcript,
                "csv": csv_fp.read(),
                "srt": srt_fp.read(),
            }
            return JSONResponse(data)