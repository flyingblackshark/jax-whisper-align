import gradio as gr
import os
import csv
import jax
import gradio as gr
import infererence
def transcribe_audio(audio_file):

    segments,detected_language = infererence.process_audio(audio_file)

    output_segments = [
        {
            "start": segment["start"] / 16000,
            "end": segment["end"] / 16000,
            "text": segment["text"]
        }
        for segment in segments
    ]

    srt_text = ""
    for i, segment in enumerate(segments, start=1):
        srt_text += f"{i}\n"
        srt_text += f"{format_time(segment['start']/ 16000)} --> {format_time(segment['end']/ 16000)}\n"
        srt_text += f"{segment['text']}\n\n"

    return output_segments, srt_text

def format_time(seconds):
    # 将秒转换为 SRT 格式的时间
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def download_file(file_path):
    # 读取文件内容以提供下载
    with open(file_path, "rb") as file:
        return file.read()

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

        def process_audio(audio_file):
            if not audio_file:
                return "请上传音频文件。", ""

            segments, srt_text = transcribe_audio(audio_file)
            text_output = "\n".join(
                [f"[{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}" for segment in segments]
            )

            return text_output, srt_text

        transcribe_button.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[output_text, srt_output]
        )

    # 运行 Gradio 界面
    demo.queue()
    demo.launch(server_name='0.0.0.0')