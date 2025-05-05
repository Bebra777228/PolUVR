import os
import re
import sys
import torch
import shutil
import logging
import subprocess
import gradio as gr

from PolUVR.separator import Separator
from UVR_resources import DEMUCS_v4_MODELS, VR_ARCH_MODELS, MDXNET_MODELS, MDX23C_MODELS, ROFORMER_MODELS

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

OUTPUT_FORMAT = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"]

def reset_stems():
    """Resets all audio components before new separation"""
    return [gr.update(value=None, visible=False) for _ in range(6)]

def print_message(input_file, model_name):
    """Prints information about the audio separation process."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print("\n")
    print("🎵 PolUVR 🎵")
    print("Input file:", base_name)
    print("Model used:", model_name)
    print("Audio separation in progress...")

def prepare_output_dir(input_file, output_dir):
    """Creates a directory to save the results and clears it if it already exists."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(output_dir, base_name)
    try:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    except Exception as e:
        raise RuntimeError(f"Error creating output directory {out_dir}: {e}") from e
    return out_dir

def rename_all_stems(audio, rename_stems, model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {"All Stems": rename_stems.replace("NAME", base_name).replace("STEM", "All Stems").replace("MODEL", model)}
    return stems

def leaderboard(list_filter, list_limit):
    try:
        result = subprocess.run(
            ["PolUVR", "-l", f"--list_filter={list_filter}", f"--list_limit={list_limit}"],
            capture_output=True,
            text=True,
            check=True,
        )
        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return "<table border='1'>" + "".join(
            f"<tr style='{'font-weight: bold; font-size: 1.2em;' if i == 0 else ''}'>" +
            "".join(f"<td>{cell}</td>" for cell in re.split(r"\s{2,}", line.strip())) +
            "</tr>"
            for i, line in enumerate(re.findall(r"^(?!-+)(.+)$", result.stdout.strip(), re.MULTILINE))
        ) + "</table>"

    except Exception as e:
        return f"Error: {e}"

def process_separation_results(separation, out_dir):
    """Process separation results and prepare outputs for UI components."""
    stems = [os.path.join(out_dir, file_name) for file_name in separation]

    outputs = []
    for i in range(6):
        if i < len(stems):
            outputs.append(gr.update(value=stems[i], visible=True, label=f"Stem {i+1} ({os.path.basename(stems[i])}"))
        else:
            outputs.append(gr.update(visible=False))

    return outputs

def create_stems_display():
    """Helper function to create 2-column stems display"""
    stems = []
    with gr.Column():
        for i in range(0, 6, 2):
            with gr.Row():
                stems.append(gr.Audio(visible=False, interactive=False, label=f"Stem {i+1}"))
                stems.append(gr.Audio(visible=False, interactive=False, label=f"Stem {i+2}"))
    return stems

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, rename_stems, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the Roformer model."""
    yield reset_stems()

    stemname = rename_all_stems(audio, rename_stems, model_key)
    print_message(audio, model_key)
    model = ROFORMER_MODELS[model_key]

    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        yield process_separation_results(separation, out_dir)
    except Exception as e:
        raise gr.Error(f"Error separating audio with Roformer: {e}") from e

def mdx23c_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, rename_stems, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the MDX23C model."""
    yield reset_stems()

    stemname = rename_all_stems(audio, rename_stems, model_key)
    print_message(audio, model_key)
    model = MDX23C_MODELS[model_key]

    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdxc_params={
                "segment_size": seg_size,
                "override_model_segment_size": override_seg_size,
                "batch_size": batch_size,
                "overlap": overlap,
                "pitch_shift": pitch_shift,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        yield process_separation_results(separation, out_dir)
    except Exception as e:
        raise gr.Error(f"Error separating audio with MDX23C: {e}") from e

def mdx_separator(audio, model_key, hop_length, seg_size, overlap, denoise, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, rename_stems, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the MDX-NET model."""
    yield reset_stems()

    stemname = rename_all_stems(audio, rename_stems, model_key)
    print_message(audio, model_key)
    model = MDXNET_MODELS[model_key]

    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            mdx_params={
                "hop_length": hop_length,
                "segment_size": seg_size,
                "overlap": overlap,
                "batch_size": batch_size,
                "enable_denoise": denoise,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        yield process_separation_results(separation, out_dir)
    except Exception as e:
        raise gr.Error(f"Error separating audio with MDX-NET: {e}") from e

def vr_separator(audio, model_key, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, rename_stems, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the VR ARCH model."""
    yield reset_stems()

    stemname = rename_all_stems(audio, rename_stems, model_key)
    print_message(audio, model_key)
    model = VR_ARCH_MODELS[model_key]

    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            vr_params={
                "batch_size": batch_size,
                "window_size": window_size,
                "aggression": aggression,
                "enable_tta": tta,
                "enable_post_process": post_process,
                "post_process_threshold": post_process_threshold,
                "high_end_process": high_end_process,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        yield process_separation_results(separation, out_dir)
    except Exception as e:
        raise gr.Error(f"Error separating audio with VR ARCH: {e}") from e

def demucs_separator(audio, model_key, seg_size, shifts, overlap, segments_enabled, model_dir, out_dir, out_format, norm_thresh, amp_thresh, rename_stems, progress=gr.Progress(track_tqdm=True)):
    """Performs audio separation using the Demucs model."""
    yield reset_stems()

    stemname = rename_all_stems(audio, rename_stems, model_key)
    print_message(audio, model_key)
    model = DEMUCS_v4_MODELS[model_key]

    try:
        out_dir = prepare_output_dir(audio, out_dir)
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=model_dir,
            output_dir=out_dir,
            output_format=out_format,
            normalization_threshold=norm_thresh,
            amplification_threshold=amp_thresh,
            use_autocast=use_autocast,
            demucs_params={
                "segment_size": seg_size,
                "shifts": shifts,
                "overlap": overlap,
                "segments_enabled": segments_enabled,
            },
        )

        progress(0.2, desc="Model loading...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separation...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        yield process_separation_results(separation, out_dir)
    except Exception as e:
        raise gr.Error(f"Error separating audio with Demucs: {e}") from e

def show_hide_params(param):
    """Updates the visibility of a parameter based on the checkbox state."""
    return gr.update(visible=param)

def clear_models(model_dir):
    """Deletes all model files from the specified directory."""
    try:
        for filename in os.listdir(model_dir):
            if filename.endswith((".th", ".pth", ".onnx", ".ckpt", ".json", ".yaml")):
                file_path = os.path.join(model_dir, filename)
                os.remove(file_path)
        return gr.Info("Models successfully cleared from memory.")
    except Exception as e:
        return gr.Error(f"Error deleting models: {e}")

def PolUVR_UI(default_model_file_dir="/tmp/PolUVR-models/", default_output_dir="output"):
    with gr.Tab("Roformer"):
        with gr.Group():
            with gr.Row():
                roformer_model = gr.Dropdown(value="MelBand Roformer Kim | Big Beta v5e FT by Unwa", label="Model", choices=list(ROFORMER_MODELS.keys()), scale=3)
                roformer_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", scale=1)
            with gr.Accordion("Separation Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        roformer_override_seg_size = gr.Checkbox(value=False, label="Override Segment Size", info="Use a custom segment size instead of the default value.")
                        with gr.Row():
                            roformer_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.", visible=False)
                            roformer_overlap = gr.Slider(minimum=2, maximum=10, step=1, value=8, label="Overlap", info="Decreasing overlap improves quality but slows down processing.")
                            roformer_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch Shift", info="Pitch shifting can improve separation for certain types of vocals.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            roformer_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            roformer_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            roformer_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            roformer_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            roformer_button = gr.Button("Start Separation", variant="primary")
        roformer_stems = create_stems_display()

    with gr.Tab("MDX23C"):
        with gr.Group():
            with gr.Row():
                mdx23c_model = gr.Dropdown(value="MDX23C InstVoc HQ", label="Model", choices=list(MDX23C_MODELS.keys()), scale=3)
                mdx23c_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", scale=1)
            with gr.Accordion("Separation Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        mdx23c_override_seg_size = gr.Checkbox(value=False, label="Override Segment Size", info="Use a custom segment size instead of the default value.")
                        with gr.Row():
                            mdx23c_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.", visible=False)
                            mdx23c_overlap = gr.Slider(minimum=2, maximum=50, step=1, value=8, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                            mdx23c_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch Shift", info="Pitch shifting can improve separation for certain types of vocals.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            mdx23c_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            mdx23c_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            mdx23c_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            mdx23c_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx23c_button = gr.Button("Start Separation", variant="primary")
        mdx23c_stems = create_stems_display()

    with gr.Tab("MDX-NET"):
        with gr.Group():
            with gr.Row():
                mdx_model = gr.Dropdown(value="UVR-MDX-NET Inst HQ 5", label="Model", choices=list(MDXNET_MODELS.keys()), scale=3)
                mdx_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", scale=1)
            with gr.Accordion("Separation Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        mdx_denoise = gr.Checkbox(value=False, label="Denoise", info="Enable denoising after separation.")
                        with gr.Row():
                            mdx_hop_length = gr.Slider(minimum=32, maximum=2048, step=32, value=1024, label="Hop Length", info="Parameter affecting separation accuracy.")
                            mdx_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Increasing the size can improve quality but requires more resources.")
                            mdx_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            mdx_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            mdx_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            mdx_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            mdx_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            mdx_button = gr.Button("Start Separation", variant="primary")
        mdx_stems = create_stems_display()

    with gr.Tab("VR ARCH"):
        with gr.Group():
            with gr.Row():
                vr_model = gr.Dropdown(value="1_HP-UVR", label="Model", choices=list(VR_ARCH_MODELS.keys()), scale=3)
                vr_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", scale=1)
            with gr.Accordion("Separation Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            vr_post_process = gr.Checkbox(value=False, label="Post-Process", info="Enable additional processing to improve separation quality.")
                            vr_tta = gr.Checkbox(value=False, label="TTA", info="Enable test-time augmentation for better quality.")
                            vr_high_end_process = gr.Checkbox(value=False, label="High-End Process", info="Restore missing high frequencies.")
                        with gr.Row():
                            vr_post_process_threshold = gr.Slider(minimum=0.1, maximum=0.3, step=0.1, value=0.2, label="Post-Process Threshold", info="Threshold for applying post-processing.", visible=False)
                            vr_window_size = gr.Slider(minimum=320, maximum=1024, step=32, value=512, label="Window Size", info="Decreasing window size improves quality but slows down processing.")
                            vr_aggression = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Aggression", info="Intensity of the main stem separation.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            vr_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Increasing batch size speeds up processing but requires more memory.")
                            vr_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            vr_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            vr_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            vr_button = gr.Button("Start Separation", variant="primary")
        vr_stems = create_stems_display()

    with gr.Tab("Demucs"):
        with gr.Group():
            with gr.Row():
                demucs_model = gr.Dropdown(value="htdemucs_ft", label="Model", choices=list(DEMUCS_v4_MODELS.keys()), scale=3)
                demucs_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output File Format", scale=1)
            with gr.Accordion("Separation Parameters", open=False):
                with gr.Column(variant="panel"):
                    with gr.Group():
                        demucs_segments_enabled = gr.Checkbox(value=True, label="Segment Processing", info="Enable processing audio in segments.")
                        with gr.Row():
                            demucs_seg_size = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Segment Size", info="Increasing segment size improves quality but slows down processing.")
                            demucs_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Increasing overlap improves quality but slows down processing.")
                            demucs_shifts = gr.Slider(minimum=0, maximum=20, step=1, value=2, label="Shifts", info="Increasing shifts improves quality but slows down processing.")
                with gr.Column(variant="panel"):
                    with gr.Group():
                        with gr.Row():
                            demucs_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization Threshold", info="Threshold for normalizing audio volume.")
                            demucs_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification Threshold", info="Threshold for amplifying quiet parts of the audio.")
        with gr.Row():
            demucs_audio = gr.Audio(label="Input Audio", type="filepath")
        with gr.Row():
            demucs_button = gr.Button("Start Separation", variant="primary")
        demucs_stems = create_stems_display()

    with gr.Tab("Settings"):
        with gr.Row():
            with gr.Column(variant="panel"):
                model_file_dir = gr.Textbox(value=default_model_file_dir, label="Model Directory", info="Specify the path to store model files.", placeholder="models/UVR_models")
                gr.HTML("""<div style="margin: -10px 0!important; text-align: center">The button below will delete all previously installed models from your device.</div>""")
                clear_models_button = gr.Button("Remove models from memory", variant="primary")
            with gr.Column(variant="panel"):
                output_dir = gr.Textbox(value=default_output_dir, label="Output Directory", info="Specify the path to save output files.", placeholder="output/UVR_output")

        with gr.Column():
            with gr.Group():
                gr.Markdown(
                    """
                    > Use keys to automatically format output file names.

                    > Available keys:
                    > * **NAME** - Input file name
                    > * **STEM** - Stem type (e.g., Vocals, Instrumental)
                    > * **MODEL** - Model name (e.g., BS-Roformer-Viperx-1297)

                    > Example:
                    > * **Template:** NAME_(STEM)_MODEL
                    > * **Result:** Music_(Vocals)_BS-Roformer-Viperx-1297
                    
                    <div style="color: red; font-weight: bold; background-color: #ffecec; padding: 10px; border-left: 3px solid red; margin: 10px 0;">
                    ⚠️ WARNING: This line changes the names of all output files at once. 
                    Use ONLY the specified keys (NAME, STEM, MODEL) to avoid corrupting the files. 
                    Do NOT add any extra text or characters outside these keys, or do so with caution.
                    </div>
                    """
                )
                rename_stems = gr.Textbox(value="NAME_(STEM)_MODEL", label="Rename Stems", placeholder="NAME_(STEM)_MODEL")

    with gr.Tab("Leaderboard"):
        with gr.Group():
            with gr.Row(equal_height=True):
                list_filter = gr.Dropdown(value="vocals", choices=["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"], label="Filter", info="Filter models by stem type.")
                list_limit = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Limit", info="Limit the number of displayed models.")
                list_button = gr.Button("Refresh List", variant="primary")

        output_list = gr.HTML(label="Leaderboard")

    # Event handlers
    roformer_override_seg_size.change(show_hide_params, inputs=[roformer_override_seg_size], outputs=[roformer_seg_size])
    mdx23c_override_seg_size.change(show_hide_params, inputs=[mdx23c_override_seg_size], outputs=[mdx23c_seg_size])
    vr_post_process.change(show_hide_params, inputs=[vr_post_process], outputs=[vr_post_process_threshold])
    list_button.click(leaderboard, inputs=[list_filter, list_limit], outputs=output_list)
    clear_models_button.click(clear_models, inputs=[model_file_dir])

    # Separation buttons
    roformer_button.click(
        roformer_separator,
        inputs=[
            roformer_audio,
            roformer_model,
            roformer_seg_size,
            roformer_override_seg_size,
            roformer_overlap,
            roformer_pitch_shift,
            model_file_dir,
            output_dir,
            roformer_output_format,
            roformer_norm_threshold,
            roformer_amp_threshold,
            roformer_batch_size,
            rename_stems
        ],
        outputs=roformer_stems,
        show_progress_on=roformer_audio,
        api_name=False
    )
    mdx23c_button.click(
        mdx23c_separator,
        inputs=[
            mdx23c_audio,
            mdx23c_model,
            mdx23c_seg_size,
            mdx23c_override_seg_size,
            mdx23c_overlap,
            mdx23c_pitch_shift,
            model_file_dir,
            output_dir,
            mdx23c_output_format,
            mdx23c_norm_threshold,
            mdx23c_amp_threshold,
            mdx23c_batch_size,
            rename_stems
        ],
        outputs=mdx23c_stems,
        show_progress_on=mdx23c_audio,
        api_name=False
    )
    mdx_button.click(
        mdx_separator,
        inputs=[
            mdx_audio,
            mdx_model,
            mdx_hop_length,
            mdx_seg_size,
            mdx_overlap,
            mdx_denoise,
            model_file_dir,
            output_dir,
            mdx_output_format,
            mdx_norm_threshold,
            mdx_amp_threshold,
            mdx_batch_size,
            rename_stems
        ],
        outputs=mdx_stems,
        show_progress_on=mdx_audio,
        api_name=False
    )
    vr_button.click(
        vr_separator,
        inputs=[
            vr_audio,
            vr_model,
            vr_window_size,
            vr_aggression,
            vr_tta,
            vr_post_process,
            vr_post_process_threshold,
            vr_high_end_process,
            model_file_dir,
            output_dir,
            vr_output_format,
            vr_norm_threshold,
            vr_amp_threshold,
            vr_batch_size,
            rename_stems
        ],
        outputs=vr_stems,
        show_progress_on=vr_audio,
        api_name=False
    )
    demucs_button.click(
        demucs_separator,
        inputs=[
            demucs_audio,
            demucs_model,
            demucs_seg_size,
            demucs_shifts,
            demucs_overlap,
            demucs_segments_enabled,
            model_file_dir,
            output_dir,
            demucs_output_format,
            demucs_norm_threshold,
            demucs_amp_threshold,
            rename_stems
        ],
        outputs=demucs_stems,
        show_progress_on=demucs_audio,
        api_name=False
    )


def main():
    with gr.Blocks(
        title="🎵 PolUVR 🎵",
        css="footer{display:none !important}",
        theme=gr.themes.Default(spacing_size="sm", radius_size="lg")
    ) as app:
        gr.HTML("<h1><center> 🎵 PolUVR 🎵 </center></h1>")
        PolUVR_UI()

    app.queue().launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        debug=True,
        show_error=True,
    )

if __name__ == "__main__":
    main()
