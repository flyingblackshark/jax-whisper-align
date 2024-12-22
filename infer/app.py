import gradio as gr
import os
import csv
import jax
import infererence
def transcribe_audio(audio_file):
    # 加载 Whisper 模型


    # 使用模型进行语音转文本
    result,detected_language = infererence.process_audio(audio_file)

    # 获取转录的文本分段
    segments = result#["segments"]

    # 转换为分段的输出格式
    output_segments = [
        {
            "start": segment["start"] / 16000,
            "end": segment["end"] / 16000,
            "text": segment["text"]
        }
        for segment in segments
    ]

    # 创建 CSV 文件
    csv_file = "output.csv"
    with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for segment in result:
            start_time = segment["start"] / 16000
            end_time = segment["end"] / 16000
            transcript = segment["text"]
            writer.writerow([os.path.basename(audio_file), start_time, end_time, transcript, detected_language])
    srt_file = "output.srt"
    with open(srt_file, mode="w", encoding="utf-8") as f:
        for i, segment in enumerate(result, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(segment['start']/ 16000)} --> {format_time(segment['end']/ 16000)}\n")
            f.write(f"{segment['text']}\n\n")

    return output_segments, csv_file, srt_file

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
            download_csv = gr.File(label="下载 CSV", interactive=False)
            download_srt = gr.File(label="下载 SRT", interactive=False)

        def process_audio(audio_file):
            if not audio_file:
                return "请上传音频文件。", None, None

            segments, csv_file, srt_file = transcribe_audio(audio_file)
            text_output = "\n".join(
                [f"[{segment['start']:.2f}-{segment['end']:.2f}]: {segment['text']}" for segment in segments]
            )

            return text_output, csv_file, srt_file

        transcribe_button.click(
            fn=process_audio,
            inputs=audio_input,
            outputs=[output_text, download_csv, download_srt]
        )

    # 运行 Gradio 界面
    demo.queue()
    demo.launch(server_name='0.0.0.0')