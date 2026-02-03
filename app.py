# -*- coding: utf-8 -*-
"""基于 wav2vec2 与 IEMOCAP 的情感识别 WebUI 演示（不加载真实模型）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载模型，实际不下载权重，仅用于界面演示。"""
    return "模型状态：emotion-recognition-wav2vec2-IEMOCAP 已就绪（演示模式，未加载真实权重）"


def fake_emotion_recognition(audio_input):
    """模拟对音频进行情感识别并返回可视化描述。"""
    if audio_input is None:
        return "请上传或录制一段音频后进行情感识别。", ""
    duration = "（时长由实际音频决定）"
    lines = [
        "[演示] 已对音频进行情感识别（未加载真实模型）。",
        "识别结果示例（占位）：",
        "  预测情感：中性 / 高兴 / 悲伤 / 愤怒 等",
        "  置信度：0.xx",
        "",
        "加载真实 emotion-recognition-wav2vec2-IEMOCAP 模型后，将在此显示实际情感类别及概率分布。",
    ]
    return "\n".join(lines), "中性"  # 占位标签


def build_ui():
    with gr.Blocks(title="情感识别 wav2vec2-IEMOCAP WebUI") as demo:
        gr.Markdown("## 基于 wav2vec2 与 IEMOCAP 的情感识别 · WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示基于 wav2vec2 在 IEMOCAP 数据上微调的情感识别模型的典型使用流程，"
            "包括模型加载状态与音频情感识别结果的可视化展示。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("情感识别"):
                gr.Markdown("上传或录制一段音频，模型将预测其情感类别（如中性、高兴、悲伤、愤怒等）。")
                audio_in = gr.Audio(
                    label="输入音频",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                out_text = gr.Textbox(
                    label="识别结果说明",
                    lines=8,
                    interactive=False,
                )
                out_label = gr.Textbox(
                    label="预测情感标签",
                    interactive=False,
                )
                run_btn = gr.Button("执行情感识别（演示）")
                run_btn.click(
                    fn=fake_emotion_recognition,
                    inputs=[audio_in],
                    outputs=[out_text, out_label],
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载 emotion-recognition-wav2vec2-IEMOCAP 模型参数。*"
        )

    return demo


def main():
    app = build_ui()
    app.launch(server_name="127.0.0.1", server_port=8770, share=False)


if __name__ == "__main__":
    main()
