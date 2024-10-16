import os
import yt_dlp
import gradio as gr

from audio_separator.separator import Separator

ROFORMER_MODEL = {
  'BS-Roformer-Viperx-1297.ckpt': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
  'BS-Roformer-Viperx-1296.ckpt': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
  'BS-Roformer-Viperx-1053.ckpt': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
  'Mel-Roformer-Viperx-1143.ckpt': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
  'Mel-Roformer-Crowd-Aufr33-Viperx.ckpt': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
  'Mel-Roformer-Karaoke-Aufr33-Viperx.ckpt': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
  'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
  'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
}
MDC23C_MODEL = [
  'MDX23C_D1581.ckpt',
  'MDX23C-8KFFT-InstVoc_HQ.ckpt',
  'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
]
MDXNET_MODEL = [
  'UVR-MDX-NET-Inst_full_292.onnx',
  'UVR-MDX-NET_Inst_187_beta.onnx',
  'UVR-MDX-NET_Inst_82_beta.onnx',
  'UVR-MDX-NET_Inst_90_beta.onnx',
  'UVR-MDX-NET_Main_340.onnx',
  'UVR-MDX-NET_Main_390.onnx',
  'UVR-MDX-NET_Main_406.onnx',
  'UVR-MDX-NET_Main_427.onnx',
  'UVR-MDX-NET_Main_438.onnx',
  'UVR-MDX-NET-Inst_HQ_1.onnx',
  'UVR-MDX-NET-Inst_HQ_2.onnx',
  'UVR-MDX-NET-Inst_HQ_3.onnx',
  'UVR-MDX-NET-Inst_HQ_4.onnx',
  'UVR_MDXNET_Main.onnx',
  'UVR-MDX-NET-Inst_Main.onnx',
  'UVR_MDXNET_1_9703.onnx',
  'UVR_MDXNET_2_9682.onnx',
  'UVR_MDXNET_3_9662.onnx',
  'UVR-MDX-NET-Inst_1.onnx',
  'UVR-MDX-NET-Inst_2.onnx',
  'UVR-MDX-NET-Inst_3.onnx',
  'UVR_MDXNET_KARA.onnx',
  'UVR_MDXNET_KARA_2.onnx',
  'UVR_MDXNET_9482.onnx',
  'UVR-MDX-NET-Voc_FT.onnx',
  'Kim_Vocal_1.onnx',
  'Kim_Vocal_2.onnx',
  'Kim_Inst.onnx',
  'Reverb_HQ_By_FoxJoy.onnx',
  'UVR-MDX-NET_Crowd_HQ_1.onnx',
  'kuielab_a_vocals.onnx',
  'kuielab_a_other.onnx',
  'kuielab_a_bass.onnx',
  'kuielab_a_drums.onnx',
  'kuielab_b_vocals.onnx',
  'kuielab_b_other.onnx',
  'kuielab_b_bass.onnx',
  'kuielab_b_drums.onnx',
]
VR_ARCH_MODEL = [
  '1_HP-UVR.pth',
  '2_HP-UVR.pth',
  '3_HP-Vocal-UVR.pth',
  '4_HP-Vocal-UVR.pth',
  '5_HP-Karaoke-UVR.pth',
  '6_HP-Karaoke-UVR.pth',
  '7_HP2-UVR.pth',
  '8_HP2-UVR.pth',
  '9_HP2-UVR.pth',
  '10_SP-UVR-2B-32000-1.pth',
  '11_SP-UVR-2B-32000-2.pth',
  '12_SP-UVR-3B-44100.pth',
  '13_SP-UVR-4B-44100-1.pth',
  '14_SP-UVR-4B-44100-2.pth',
  '15_SP-UVR-MID-44100-1.pth',
  '16_SP-UVR-MID-44100-2.pth',
  '17_HP-Wind_Inst-UVR.pth',
  'UVR-DeEcho-DeReverb.pth',
  'UVR-De-Echo-Normal.pth',
  'UVR-De-Echo-Aggressive.pth',
  'UVR-DeNoise.pth',
  'UVR-DeNoise-Lite.pth',
  'UVR-BVE-4B_SN-44100-1.pth',
  'MGM_HIGHEND_v4.pth',
  'MGM_LOWEND_A_v4.pth',
  'MGM_LOWEND_B_v4.pth',
  'MGM_MAIN_v4.pth',
]
DEMUCS_MODEL = [
  'htdemucs_ft.yaml',
  'htdemucs_6s.yaml',
  'htdemucs.yaml',
  'hdemucs_mmi.yaml',
]

mdx_params={
  "batch_size": mdx_batch_size,
  "hop_length": mdx_hop_length,
  "segment_size": mdx_segment_size,
  "overlap": mdx_overlap,
  "enable_denoise": mdx_enable_denoise,
}
mdxc_params={
  "batch_size": mdxc_batch_size,
  "segment_size": mdxc_batch_size,
  "overlap": mdxc_batch_size,
}
vr_params={
  "batch_size": vr_batch_size,
  "window_size": vr_window_size,
  "aggression": vr_aggression,
  "enable_tta": vr_enable_tta,
  "enable_post_process": vr_enable_post_process,
  "post_process_threshold": vr_post_process_threshold,
  "high_end_process": vr_high_end_process,
}
demucs_params={
  "segment_size": demucs_segment_size,
  "shifts": demucs_shifts,
  "overlap": demucs_overlap,
  "segments_enabled": demucs_segments_enabled,
}

output_format = [
  'wav',
  'flac',
  'mp3',
]

def initialize_separator(model_file_dir, output_dir, output_format, normalization_threshold, amplification_threshold, mdx_params, vr_params, demucs_params, mdxc_params)
  return Separator(
    model_file_dir=model_file_dir,
    output_dir=output_dir,
    output_format=output_format,
    normalization_threshold=normalization_threshold,
    amplification_threshold=amplification_threshold,
    mdx_params=mdx_params,
    vr_params=vr_params,
    demucs_params=demucs_params,
    mdxc_params=mdxc_params,
  )


def roformer_separator(audio_path, roformer_model, roformer_output_format, roformer_overlap, roformer_segment_size):
  separator = initialize_separator(format, output_dir, vr_params, mdxc_params)

  roformer_separation = separator.separate(audio_path)
  
  stem1_file = roformer_separation[0]
  stem2_file = roformer_separation[1]

  return stem1_file, stem2_file


def mdxc_separator(mdx23c_audio, mdx23c_model, mdx23c_output_format, mdx23c_segment_size, mdx23c_overlap, mdx23c_denoise):
  separator = initialize_separator(format, output_dir, vr_params, mdxc_params)

  mdxc_separation = separator.separate(audio_path)

  stem1_file = mdxc_separation[0]
  stem2_file = mdxc_separation[1]

  return stem1_file, stem2_file


def mdxnet_separator(mdxnet_audio, mdxnet_model, mdxnet_output_format, mdxnet_segment_size, mdxnet_overlap, mdxnet_denoise):
  separator = initialize_separator(format, output_dir, vr_params, mdxc_params)

  mdxnet_separation = separator.separate(audio_path)

  stem1_file = mdxnet_separation[0]
  stem2_file = mdxnet_separation[1]

  return stem1_file, stem2_file


def vrarch_separator(vrarch_audio, vrarch_model, vrarch_output_format, vrarch_window_size, vrarch_agression, vrarch_tta, vrarch_high_end_process):
  separator = initialize_separator(format, output_dir, vr_params, mdxc_params)

  vrarch_separation = separator.separate(audio_path)

  stem1_file = vrarch_separation[0]
  stem2_file = vrarch_separation[1]

  return stem1_file, stem2_file


def demucs_separator(demucs_audio, demucs_model, demucs_output_format, demucs_shifts, demucs_overlap):
  separator = initialize_separator(format, output_dir, vr_params, mdxc_params)

  demucs_separation = separator.separate(audio_path)

  stem1_file = demucs_separation[0]
  stem2_file = demucs_separation[1]
  stem3_file = demucs_separation[2]
  stem4_file = demucs_separation[3]

  return stem1_file, stem2_file, stem3_file, stem4_file

with gr.Blocks(title="🎵 Audio Separator 🎵", css="footer{display:none !important}") as app:
    with gr.Tabs():
        with gr.TabItem("Roformer"):
            with gr.Row():
                roformer_model = gr.Dropdown(
                    label = "Select the Model",
                    choices=list(roformer_models.keys()),
                    interactive = True
                )
                roformer_output_format = gr.Dropdown(
                    label = "Select the Output Format",
                    choices = output_format,
                    interactive = True
                )
            with gr.Row():
                roformer_overlap = gr.Slider(
                    value = 4,
                    step = 1,
                    minimum = 2,
                    maximum = 4,
                    label = "Overlap",
                    info = "Amount of overlap between prediction windows.",
                    interactive = True
                )
                roformer_segment_size = gr.Slider(
                    value = 256,
                    step = 32,
                    minimum = 32,
                    maximum = 4000,
                    label = "Segment Size",
                    info = "Larger consumes more resources, but may give better results.",
                    interactive = True
                )
            with gr.Row():
                roformer_audio = gr.Audio(
                    label = "Input Audio",
                    type = "numpy",
                    interactive = True
                )
            with gr.Accordion("Separation by Link", open = False):
                with gr.Row():
                    roformer_link = gr.Textbox(
                    label = "Link",
                    placeholder = "Paste the link here",
                    interactive = True
                )
                with gr.Row():
                   gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    roformer_download_button = gr.Button(
                    "Download!",
                    variant = "primary"
                )

            roformer_download_button.click(download_audio, [roformer_link], [roformer_audio])

            with gr.Row():
                roformer_button = gr.Button("Separate!", variant = "primary")
            with gr.Row():
                roformer_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 1",
                    type = "filepath"
                )
                roformer_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 2",
                    type = "filepath"
                )

            roformer_button.click(roformer_separator, [roformer_audio, roformer_model, roformer_output_format, roformer_overlap, roformer_segment_size], [roformer_stem1, roformer_stem2])
        
        with gr.TabItem("MDX23C"):
            with gr.Row():
                mdx23c_model = gr.Dropdown(
                    label = "Select the Model",
                    choices = mdx23c_models,
                    interactive = True
                )
                mdx23c_output_format = gr.Dropdown(
                    label = "Select the Output Format",
                    choices = output_format,
                    interactive = True
                )
            with gr.Row():
                mdx23c_segment_size = gr.Slider(
                    minimum = 32,
                    maximum = 4000,
                    step = 32,
                    label = "Segment Size",
                    info = "Larger consumes more resources, but may give better results.",
                    value = 256,
                    interactive = True
                )
                mdx23c_overlap = gr.Slider(
                    minimum = 2,
                    maximum = 50,
                    step = 1,
                    label = "Overlap",
                    info = "Amount of overlap between prediction windows.",
                    value = 8,
                    interactive = True
                )
                mdx23c_denoise = gr.Checkbox(
                    label = "Denoise",
                    info = "Enable denoising during separation.",
                    value = False,
                    interactive = True
                )
            with gr.Row():
                mdx23c_audio = gr.Audio(
                    label = "Input Audio",
                    type = "numpy",
                    interactive = True
                )
            with gr.Accordion("Separation by Link", open = False):
                with gr.Row():
                    mdx23c_link = gr.Textbox(
                    label = "Link",
                    placeholder = "Paste the link here",
                    interactive = True
                )
                with gr.Row():
                   gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    mdx23c_download_button = gr.Button(
                    "Download!",
                    variant = "primary"
                )

            mdx23c_download_button.click(download_audio, [mdx23c_link], [mdx23c_audio])

            with gr.Row():
                mdx23c_button = gr.Button("Separate!", variant = "primary")
            with gr.Row():
                mdx23c_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 1",
                    type = "filepath"
                )
                mdx23c_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 2",
                    type = "filepath"
                )

            mdx23c_button.click(mdxc_separator, [mdx23c_audio, mdx23c_model, mdx23c_output_format, mdx23c_segment_size, mdx23c_overlap, mdx23c_denoise], [mdx23c_stem1, mdx23c_stem2])
        
        with gr.TabItem("MDX-NET"):
            with gr.Row():
                mdxnet_model = gr.Dropdown(
                    label = "Select the Model",
                    choices = mdxnet_models,
                    interactive = True
                )
                mdxnet_output_format = gr.Dropdown(
                    label = "Select the Output Format",
                    choices = output_format,
                    interactive = True
                )
            with gr.Row():
                mdxnet_segment_size = gr.Slider(
                    minimum = 32,
                    maximum = 4000,
                    step = 32,
                    label = "Segment Size",
                    info = "Larger consumes more resources, but may give better results.",
                    value = 256,
                    interactive = True
                )
                mdxnet_overlap = gr.Dropdown(
                        label = "Overlap",
                        choices = mdxnet_overlap_values,
                        value = mdxnet_overlap_values[0],
                        interactive = True
                )
                mdxnet_denoise = gr.Checkbox(
                    label = "Denoise",
                    info = "Enable denoising during separation.",
                    value = True,
                    interactive = True
                )
            with gr.Row():
                mdxnet_audio = gr.Audio(
                    label = "Input Audio",
                    type = "numpy",
                    interactive = True
                )
            with gr.Accordion("Separation by Link", open = False):
                with gr.Row():
                    mdxnet_link = gr.Textbox(
                    label = "Link",
                    placeholder = "Paste the link here",
                    interactive = True
                )
                with gr.Row():
                   gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    mdxnet_download_button = gr.Button(
                    "Download!",
                    variant = "primary"
                )

            mdxnet_download_button.click(download_audio, [mdxnet_link], [mdxnet_audio])

            with gr.Row():
                mdxnet_button = gr.Button("Separate!", variant = "primary")
            with gr.Row():
                mdxnet_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 1",
                    type = "filepath"
                )
                mdxnet_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    label = "Stem 2",
                    type = "filepath"
                )

            mdxnet_button.click(mdxnet_separator, [mdxnet_audio, mdxnet_model, mdxnet_output_format, mdxnet_segment_size, mdxnet_overlap, mdxnet_denoise], [mdxnet_stem1, mdxnet_stem2])

        with gr.TabItem("VR ARCH"):
            with gr.Row():
                vrarch_model = gr.Dropdown(
                    label = "Select the Model",
                    choices = vrarch_models,
                    interactive = True
                )
                vrarch_output_format = gr.Dropdown(
                    label = "Select the Output Format",
                    choices = output_format,
                    interactive = True
                )
            with gr.Row():
                vrarch_window_size = gr.Dropdown(
                    label = "Window Size",
                    choices = vrarch_window_size_values,
                    value = vrarch_window_size_values[0],
                    interactive = True
                )
                vrarch_agression = gr.Slider(
                    minimum = 1,
                    maximum = 50,
                    step = 1,
                    label = "Agression",
                    info = "Intensity of primary stem extraction.",
                    value = 5,
                    interactive = True
                )
                vrarch_tta = gr.Checkbox(
                    label = "TTA",
                    info = "Enable Test-Time-Augmentation; slow but improves quality.",
                    value = True,
                    visible = True,
                    interactive = True,
                )
                vrarch_high_end_process = gr.Checkbox(
                    label = "High End Process",
                    info = "Mirror the missing frequency range of the output.",
                    value = False,
                    visible = True,
                    interactive = True,
                )
            with gr.Row():
                vrarch_audio = gr.Audio(
                    label = "Input Audio",
                    type = "numpy",
                    interactive = True
                )
            with gr.Accordion("Separation by Link", open = False):
                with gr.Row():
                   vrarch_link = gr.Textbox(
                    label = "Link",
                    placeholder = "Paste the link here",
                    interactive = True
                )
                with gr.Row():
                   gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    vrarch_download_button = gr.Button(
                    "Download!",
                    variant = "primary"
                )

            vrarch_download_button.click(download_audio, [vrarch_link], [vrarch_audio])

            with gr.Row():
                vrarch_button = gr.Button("Separate!", variant = "primary")
            with gr.Row():
                vrarch_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 1"
                )
                vrarch_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 2"
                )

            vrarch_button.click(vrarch_separator, [vrarch_audio, vrarch_model, vrarch_output_format, vrarch_window_size, vrarch_agression, vrarch_tta, vrarch_high_end_process], [vrarch_stem1, vrarch_stem2])

        with gr.TabItem("Demucs"):
            with gr.Row():
                demucs_model = gr.Dropdown(
                    label = "Select the Model",
                    choices = demucs_models,
                    interactive = True
                )
                demucs_output_format = gr.Dropdown(
                    label = "Select the Output Format",
                    choices = output_format,
                    interactive = True
                )
            with gr.Row():
                demucs_shifts = gr.Slider(
                    minimum = 1,
                    maximum = 20,
                    step = 1,
                    label = "Shifts",
                    info = "Number of predictions with random shifts, higher = slower but better quality.",
                    value = 2,
                    interactive = True
                )
                demucs_overlap = gr.Dropdown(
                   label = "Overlap",
                   choices = demucs_overlap_values,
                   value = demucs_overlap_values[0],
                   interactive = True
                )
            with gr.Row():
                demucs_audio = gr.Audio(
                    label = "Input Audio",
                    type = "numpy",
                    interactive = True
                )
            with gr.Accordion("Separation by Link", open = False):
                with gr.Row():
                    demucs_link = gr.Textbox(
                    label = "Link",
                    placeholder = "Paste the link here",
                    interactive = True
                )
                with gr.Row():
                   gr.Markdown("You can paste the link to the video/audio from many sites, check the complete list [here](https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md)")
                with gr.Row():
                    demucs_download_button = gr.Button(
                    "Download!",
                    variant = "primary"
                )

            demucs_download_button.click(download_audio, [demucs_link], [demucs_audio])

            with gr.Row():
                demucs_button = gr.Button("Separate!", variant = "primary")
            with gr.Row():
                demucs_stem1 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 1"
                )
                demucs_stem2 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 2"
                )
            with gr.Row():
                demucs_stem3 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 3"
                )
                demucs_stem4 = gr.Audio(
                    show_download_button = True,
                    interactive = False,
                    type = "filepath",
                    label = "Stem 4"
                )
            
            demucs_button.click(demucs_separator, [demucs_audio, demucs_model, demucs_output_format, demucs_shifts, demucs_overlap], [demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4])

app.launch(share=True)
