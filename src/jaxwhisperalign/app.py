"""Gradio web interface for JAX Whisper alignment."""

import jax
import gradio as gr
from typing import List, Dict, Tuple, Optional

from jaxwhisperalign import infererence
def transcribe_audio(audio_file: str) -> Tuple[List[Dict], str]:
    """Transcribe audio file and return segments with SRT text.
    
    Args:
        audio_file: Path to audio file
        
    Returns:
        Tuple of (output_segments, srt_text)
    """
    try:
        segments, detected_language = infererence.process_audio(audio_file)
        
        # Convert segments to output format
        output_segments = [
            {
                "start": segment["start"] / SAMPLE_RATE,
                "end": segment["end"] / SAMPLE_RATE,
                "text": segment["text"]
            }
            for segment in segments
        ]

        # Generate SRT text
        srt_text = ""
        for i, segment in enumerate(segments, start=1):
            start_time = format_time(segment['start'] / SAMPLE_RATE)
            end_time = format_time(segment['end'] / SAMPLE_RATE)
            srt_text += f"{i}\n{start_time} --> {end_time}\n{segment['text']}\n\n"

        return output_segments, srt_text
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {str(e)}")

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Constants
SAMPLE_RATE = 16000

if __name__ == "__main__":
    jax.distributed.initialize()
    # 创建 Gradio 界面
    with gr.Blocks() as demo:
        gr.Markdown("## 语音转文本工具")

        with gr.Row():
            audio_input = gr.Audio(label="上传音频", type="filepath")
            transcribe_button = gr.Button("开始转录")

        with gr.Row():
            output_text = gr.Textbox(label="识别结果", lines=10)

        with gr.Row():
            srt_output = gr.Textbox(label="SRT 格式", lines=10)

        def process_audio(audio_file: Optional[str]) -> Tuple[str, str]:
            """Process uploaded audio file and return transcription results.
            
            Args:
                audio_file: Path to uploaded audio file
                
            Returns:
                Tuple of (text_output, srt_text)
            """
            if not audio_file:
                return "请上传音频文件。", ""

            try:
                segments, srt_text = transcribe_audio(audio_file)
                text_output = "\n".join(
                    f"[{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}"
                    for segment in segments
                )
                return text_output, srt_text
            except Exception as e:
                error_msg = f"处理音频时出错: {str(e)}"
                return error_msg, ""

        transcribe_button.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[output_text, srt_output]
        )

    # Launch Gradio interface
    demo.queue()
    demo.launch(server_name='0.0.0.0', share=False)