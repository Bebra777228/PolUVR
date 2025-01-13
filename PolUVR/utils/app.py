import os
import re
import sys
import torch
import shutil
import logging
import subprocess
import gradio as gr
import yt_dlp

from PolUVR.separator import Separator

device = "cuda" if torch.cuda.is_available() else "cpu"
use_autocast = device == "cuda"

#=========================#
#     Roformer Models     #
#=========================#
ROFORMER_MODELS = {
    # BS Roformer
    'BS-Roformer-Viperx-1053': 'model_bs_roformer_ep_937_sdr_10.5309.ckpt',
    'BS-Roformer-Viperx-1296': 'model_bs_roformer_ep_368_sdr_12.9628.ckpt',
    'BS-Roformer-Viperx-1297': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
    'BS-Roformer-De-Reverb': 'deverb_bs_roformer_8_384dim_10depth.ckpt',
    'BS Roformer | Chorus Male-Female by Sucial': 'model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt',

    # MelBand Roformer
    'Mel-Roformer-Crowd-Aufr33-Viperx': 'mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt',
    'Mel-Roformer-Karaoke-Aufr33-Viperx': 'mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt',
    'Mel-Roformer-Viperx-1143': 'model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt',
    'MelBand Roformer | De-Reverb-Echo by Sucial': 'dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt',
    'MelBand Roformer | De-Reverb-Echo V2 by Sucial': 'dereverb-echo_mel_band_roformer_sdr_13.4843_v2.ckpt',
    'MelBand Roformer | Aspiration Less Aggressive by Sucial': 'aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt',
    'MelBand Roformer | Aspiration by Sucial': 'aspiration_mel_band_roformer_sdr_18.9845.ckpt',
    'MelBand Roformer | De-Reverb Less Aggressive by anvuew': 'dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt',
    'MelBand Roformer | De-Reverb by anvuew': 'dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt',
    'MelBand Roformer | Vocals by Kimberley Jensen': 'vocals_mel_band_roformer.ckpt',
    'Mel-Roformer-Denoise-Aufr33-Aggr': 'denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt',
    'Mel-Roformer-Denoise-Aufr33': 'denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt',
    'MelBand Roformer | Bleed Suppressor V1 by unwa-97chris': 'mel_band_roformer_bleed_suppressor_v1.ckpt',

    # MelBand Roformer Kim
    'MelBand Roformer Kim | FT by unwa': 'mel_band_roformer_kim_ft_unwa.ckpt',
    'MelBand Roformer Kim | Big Beta 4 FT by unwa': 'melband_roformer_big_beta4.ckpt',
    'MelBand Roformer Kim | Big Beta 5e FT by unwa': 'melband_roformer_big_beta5e.ckpt',
    'MelBand Roformer Kim | Inst V1 by Unwa': 'melband_roformer_inst_v1.ckpt',
    'MelBand Roformer Kim | Inst V1 (E) by Unwa': 'melband_roformer_inst_v1e.ckpt',
    'MelBand Roformer Kim | Inst V2 by Unwa': 'melband_roformer_inst_v2.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V1 by Unwa': 'melband_roformer_instvoc_duality_v1.ckpt',
    'MelBand Roformer Kim | InstVoc Duality V2 by Unwa': 'melband_roformer_instvox_duality_v2.ckpt',
    'MelBand Roformer Kim | SYHFT by SYH99999': 'MelBandRoformerSYHFT.ckpt',
    'MelBand Roformer Kim | SYHFT V2 by SYH99999': 'MelBandRoformerSYHFTV2.ckpt',
    'MelBand Roformer Kim | SYHFT V2.5 by SYH99999': 'MelBandRoformerSYHFTV2.5.ckpt',
    'MelBand Roformer Kim | SYHFT V3 by SYH99999': 'MelBandRoformerSYHFTV3Epsilon.ckpt',
    'MelBand Roformer Kim | Big SYHFT V1 by SYH99999': 'MelBandRoformerBigSYHFTV1.ckpt',
}
#=========================#
#      MDX23C Models      #
#=========================#
MDX23C_MODELS = {
    'MDX23C DrumSep by aufr33-jarredou': 'MDX23C-DrumSep-aufr33-jarredou.ckpt',
    'MDX23C De-Reverb by aufr33-jarredou': 'MDX23C-De-Reverb-aufr33-jarredou.ckpt',
    'MDX23C-InstVoc HQ': 'MDX23C-8KFFT-InstVoc_HQ.ckpt',
    'VIP | MDX23C-InstVoc HQ 2': 'MDX23C-8KFFT-InstVoc_HQ_2.ckpt',
    'VIP | MDX23C_D1581': 'MDX23C_D1581.ckpt',
}
#=========================#
#     MDXN-NET Models     #
#=========================#
MDXNET_MODELS = {
    'UVR-MDX-NET 1': 'UVR_MDXNET_1_9703.onnx',
    'UVR-MDX-NET 2': 'UVR_MDXNET_2_9682.onnx',
    'UVR-MDX-NET 3': 'UVR_MDXNET_3_9662.onnx',
    'UVR-MDXNET_9482': 'UVR_MDXNET_9482.onnx',
    'UVR-MDX-NET Inst 1': 'UVR-MDX-NET-Inst_1.onnx',
    'UVR-MDX-NET Inst 2': 'UVR-MDX-NET-Inst_2.onnx',
    'UVR-MDX-NET Inst 3': 'UVR-MDX-NET-Inst_3.onnx',
    'UVR-MDX-NET Inst HQ 1': 'UVR-MDX-NET-Inst_HQ_1.onnx',
    'UVR-MDX-NET Inst HQ 2': 'UVR-MDX-NET-Inst_HQ_2.onnx',
    'UVR-MDX-NET Inst HQ 3': 'UVR-MDX-NET-Inst_HQ_3.onnx',
    'UVR-MDX-NET Inst HQ 4': 'UVR-MDX-NET-Inst_HQ_4.onnx',
    'UVR-MDX-NET Inst HQ 5': 'UVR-MDX-NET-Inst_HQ_5.onnx',
    'UVR-MDX-NET Inst Main': 'UVR-MDX-NET-Inst_Main.onnx',
    'UVR-MDX-NET Karaoke': 'UVR_MDXNET_KARA.onnx',
    'UVR-MDX-NET Karaoke 2': 'UVR_MDXNET_KARA_2.onnx',
    'UVR-MDX-NET Main': 'UVR_MDXNET_Main.onnx',
    'UVR-MDX-NET Voc FT': 'UVR-MDX-NET-Voc_FT.onnx',
    'Kim Inst': 'Kim_Inst.onnx',
    'Kim Vocal 1': 'Kim_Vocal_1.onnx',
    'Kim Vocal 2': 'Kim_Vocal_2.onnx',
    'kuielab_a_bass': 'kuielab_a_bass.onnx',
    'kuielab_a_drums': 'kuielab_a_drums.onnx',
    'kuielab_a_other': 'kuielab_a_other.onnx',
    'kuielab_a_vocals': 'kuielab_a_vocals.onnx',
    'kuielab_b_bass': 'kuielab_b_bass.onnx',
    'kuielab_b_drums': 'kuielab_b_drums.onnx',
    'kuielab_b_other': 'kuielab_b_other.onnx',
    'kuielab_b_vocals': 'kuielab_b_vocals.onnx',
    'Reverb HQ By FoxJoy': 'Reverb_HQ_By_FoxJoy.onnx',
    'VIP | UVR-MDX-NET_Inst_82_beta': 'UVR-MDX-NET_Inst_82_beta.onnx',
    'VIP | UVR-MDX-NET_Inst_90_beta': 'UVR-MDX-NET_Inst_90_beta.onnx',
    'VIP | UVR-MDX-NET_Inst_187_beta': 'UVR-MDX-NET_Inst_187_beta.onnx',
    'VIP | UVR-MDX-NET-Inst_full_292': 'UVR-MDX-NET-Inst_full_292.onnx',
    'VIP | UVR-MDX-NET_Main_340': 'UVR-MDX-NET_Main_340.onnx',
    'VIP | UVR-MDX-NET_Main_390': 'UVR-MDX-NET_Main_390.onnx',
    'VIP | UVR-MDX-NET_Main_406': 'UVR-MDX-NET_Main_406.onnx',
    'VIP | UVR-MDX-NET_Main_427': 'UVR-MDX-NET_Main_427.onnx',
    'VIP | UVR-MDX-NET_Main_438': 'UVR-MDX-NET_Main_438.onnx',
}
#========================#
#     VR-ARCH Models     #
#========================#
VR_ARCH_MODELS = {
    '1_HP-UVR': '1_HP-UVR.pth',
    '2_HP-UVR': '2_HP-UVR.pth',
    '3_HP-Vocal-UVR': '3_HP-Vocal-UVR.pth',
    '4_HP-Vocal-UVR': '4_HP-Vocal-UVR.pth',
    '5_HP-Karaoke-UVR': '5_HP-Karaoke-UVR.pth',
    '6_HP-Karaoke-UVR': '6_HP-Karaoke-UVR.pth',
    '7_HP2-UVR': '7_HP2-UVR.pth',
    '8_HP2-UVR': '8_HP2-UVR.pth',
    '9_HP2-UVR': '9_HP2-UVR.pth',
    '10_SP-UVR-2B-32000-1': '10_SP-UVR-2B-32000-1.pth',
    '11_SP-UVR-2B-32000-2': '11_SP-UVR-2B-32000-2.pth',
    '12_SP-UVR-3B-44100': '12_SP-UVR-3B-44100.pth',
    '13_SP-UVR-4B-44100-1': '13_SP-UVR-4B-44100-1.pth',
    '14_SP-UVR-4B-44100-2': '14_SP-UVR-4B-44100-2.pth',
    '15_SP-UVR-MID-44100-1': '15_SP-UVR-MID-44100-1.pth',
    '16_SP-UVR-MID-44100-2': '16_SP-UVR-MID-44100-2.pth',
    '17_HP-Wind_Inst-UVR': '17_HP-Wind_Inst-UVR.pth',
    'MGM_HIGHEND_v4': 'MGM_HIGHEND_v4.pth',
    'MGM_LOWEND_A_v4': 'MGM_LOWEND_A_v4.pth',
    'MGM_LOWEND_B_v4': 'MGM_LOWEND_B_v4.pth',
    'MGM_MAIN_v4': 'MGM_MAIN_v4.pth',
    'UVR-BVE-4B_SN-44100-1': 'UVR-BVE-4B_SN-44100-1.pth',
    'UVR-De-Reverb by aufr33-jarredou': 'UVR-De-Reverb-aufr33-jarredou.pth',
    'UVR-De-Echo-Aggressive by FoxJoy': 'UVR-De-Echo-Aggressive.pth',
    'UVR-De-Echo-Normal by FoxJoy': 'UVR-De-Echo-Normal.pth',
    'UVR-DeEcho-DeReverb by FoxJoy': 'UVR-DeEcho-DeReverb.pth',
    'UVR-DeNoise-Lite by FoxJoy': 'UVR-DeNoise-Lite.pth',
    'UVR-DeNoise by FoxJoy': 'UVR-DeNoise.pth',
}
#=======================#
#     DEMUCS Models     #
#=======================#
DEMUCS_MODELS = {
    'htdemucs': 'htdemucs.yaml',
    'htdemucs_6s': 'htdemucs_6s.yaml',
    'htdemucs_ft': 'htdemucs_ft.yaml',
    'hdemucs_mmi': 'hdemucs_mmi.yaml',
}

INPUT_ENDSWITH = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aiff", ".ac3")
OUTPUT_FORMAT = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"]

def print_message(input_file, model_name):
    """Prints information about the audio separation process."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    print("\n")
    print("🎵 PolUVR 🎵")
    print("Input audio:", base_name)
    print("Separation Model:", model_name)
    print("Audio Separation Process...")

def prepare_output_dir(input_file, output_dir):
    """Create a directory for the output files and clean it if it already exists."""
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    out_dir = os.path.join(output_dir, base_name)
    try:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
    except Exception as e:
        raise RuntimeError(f"Failed to prepare output directory {out_dir}: {e}")
    return out_dir

def rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model):
    base_name = os.path.splitext(os.path.basename(audio))[0]
    stems = {
        "Vocals": vocals_stem.replace("NAME", base_name).replace("STEM", "Vocals").replace("MODEL", model),
        "Instrumental": instrumental_stem.replace("NAME", base_name).replace("STEM", "Instrumental").replace("MODEL", model),
        "Drums": drums_stem.replace("NAME", base_name).replace("STEM", "Drums").replace("MODEL", model),
        "Bass": bass_stem.replace("NAME", base_name).replace("STEM", "Bass").replace("MODEL", model),
        "Other": other_stem.replace("NAME", base_name).replace("STEM", "Other").replace("MODEL", model),
        "Guitar": guitar_stem.replace("NAME", base_name).replace("STEM", "Guitar").replace("MODEL", model),
        "Piano": piano_stem.replace("NAME", base_name).replace("STEM", "Piano").replace("MODEL", model),
    }
    return stems

def leaderboard(list_filter, list_limit):
    try:
        result = subprocess.run(
            ["PolUVR", "-l", f"--list_filter={list_filter}", f"--list_limit={list_limit}"],
            capture_output=True,
            text=True,
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

def download_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '32',
        }],
        'postprocessor_args': [
            '-acodec', 'pcm_f32le'
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info_dict).replace('.webm', '.wav').replace('.mp4', '.wav')

def roformer_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using Roformer model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
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
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"Roformer separation failed: {e}") from e

def mdx23c_separator(audio, model_key, seg_size, override_seg_size, overlap, pitch_shift, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using MDX23C model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
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
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"MDX23C separation failed: {e}") from e

def mdx_separator(audio, model_key, hop_length, seg_size, overlap, denoise, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using MDX-NET model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
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
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"MDX-NET separation failed: {e}") from e

def vr_separator(audio, model_key, window_size, aggression, tta, post_process, post_process_threshold, high_end_process, model_dir, out_dir, out_format, norm_thresh, amp_thresh, batch_size, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using VR ARCH model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
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
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]
        return stems[0], stems[1]
    except Exception as e:
        raise RuntimeError(f"VR ARCH separation failed: {e}") from e

def demucs_separator(audio, model_key, seg_size, shifts, overlap, segments_enabled, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    """Separate audio using Demucs model."""
    stemname = rename_stems(audio, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, model_key)
    print_message(audio, model_key)
    model = DEMUCS_MODELS[model_key]
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
            }
        )

        progress(0.2, desc="Model loaded...")
        separator.load_model(model_filename=model)

        progress(0.7, desc="Audio separated...")
        separation = separator.separate(audio, stemname)
        print(f"Separation complete!\nResults: {', '.join(separation)}")

        stems = [os.path.join(out_dir, file_name) for file_name in separation]

        if model_key == "htdemucs_6s":
            return stems[0], stems[1], stems[2], stems[3], stems[4], stems[5]
        else:
            return stems[0], stems[1], stems[2], stems[3], None, None
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}") from e

def update_stems(model):
    """Update the visibility of stem outputs based on the selected Demucs model."""
    if model == "htdemucs_6s":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def show_hide_params(param):
    """Update the visibility of a parameter based on the checkbox state."""
    return gr.update(visible=param)

def process_single_audio(audio, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    if model_key in ROFORMER_MODELS:
        return roformer_separator(audio, model_key, 256, False, 8, 0, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif model_key in MDX23C_MODELS:
        return mdx23c_separator(audio, model_key, 256, False, 8, 0, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif model_key in MDXNET_MODELS:
        return mdx_separator(audio, model_key, 1024, 256, 0.25, False, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif model_key in VR_ARCH_MODELS:
        return vr_separator(audio, model_key, 512, 5, False, False, 0.2, False, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif model_key in DEMUCS_MODELS:
        return demucs_separator(audio, model_key, 40, 2, 0.25, True, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)

def process_batch_audio(files, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    results = []
    for file in files:
        if model_key in ROFORMER_MODELS:
            result = roformer_separator(file, model_key, 256, False, 8, 0, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
        elif model_key in MDX23C_MODELS:
            result = mdx23c_separator(file, model_key, 256, False, 8, 0, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
        elif model_key in MDXNET_MODELS:
            result = mdx_separator(file, model_key, 1024, 256, 0.25, False, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
        elif model_key in VR_ARCH_MODELS:
            result = vr_separator(file, model_key, 512, 5, False, False, 0.2, False, model_dir, out_dir, out_format, norm_thresh, amp_thresh, 1, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
        elif model_key in DEMUCS_MODELS:
            result = demucs_separator(file, model_key, 40, 2, 0.25, True, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
        results.extend(result)
    return results

def process_path_link(path, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress=gr.Progress(track_tqdm=True)):
    if os.path.isfile(path):
        return process_single_audio(path, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(INPUT_ENDSWITH)]
        return process_batch_audio(files, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    elif path.startswith('http'):
        downloaded_file = download_audio(path)
        return process_single_audio(downloaded_file, model_key, model_dir, out_dir, out_format, norm_thresh, amp_thresh, vocals_stem, instrumental_stem, other_stem, drums_stem, bass_stem, guitar_stem, piano_stem, progress)
    else:
        raise ValueError("Invalid path or URL")

with gr.Blocks(
    title="🎵 PolUVR 🎵",
    css="footer{display:none !important}",
    theme=gr.themes.Default(
        spacing_size="sm",
        radius_size="lg",
    )
) as app:
    gr.HTML("<h1><center> 🎵 PolUVR 🎵 </center></h1>")

    with gr.Tab("Roformer"):
        with gr.Group():
            with gr.Row():
                roformer_model = gr.Dropdown(value="MelBand Roformer Kim | Big Beta 5e FT by unwa", label="Select the Model", choices=list(ROFORMER_MODELS.keys()), scale=3)
                roformer_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output Format", info="The format of the output audio file.", scale=1)
            with gr.Accordion("Advanced settings", open=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        roformer_override_seg_size = gr.Checkbox(value=False, label="Override segment size", info="Override model default segment size instead of using the model default value.")
                        with gr.Row():
                            roformer_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.", visible=False)
                            roformer_overlap = gr.Slider(minimum=2, maximum=10, step=1, value=8, label="Overlap", info="Amount of overlap between prediction windows. Lower is better but slower.")
                            roformer_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch shift", info="Shift audio pitch by a number of semitones while processing. may improve output for deep/high vocals.")
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            roformer_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Larger consumes more RAM but may process slightly faster.")
                            roformer_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                            roformer_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
        
        with gr.Tabs():
            with gr.Tab("Single processing"):
                with gr.Row():
                    roformer_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Row():
                    roformer_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    roformer_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    roformer_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
            
            with gr.Tab("Batch processing"):
                with gr.Row():
                    roformer_batch_audio = gr.Files(label="Input Audio Files", file_types=["audio"], height=200)
                with gr.Row():
                    roformer_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    roformer_batch_output = gr.Files(label="Output Files", interactive=False)
            
            with gr.Tab("Path / Link"):
                with gr.Row():
                    roformer_path_link = gr.Textbox(label="Input Path or Link", placeholder="Enter the path to the audio file, folder, or a link to a video/audio")
                with gr.Row():
                    roformer_path_link_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    roformer_path_link_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    roformer_path_link_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX23C"):
        with gr.Group():
            with gr.Row():
                mdx23c_model = gr.Dropdown(value="MDX23C-InstVoc HQ", label="Select the Model", choices=list(MDX23C_MODELS.keys()), scale=3)
                mdx23c_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output Format", info="The format of the output audio file.", scale=1)
            with gr.Accordion("Advanced settings", open=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        mdx23c_override_seg_size = gr.Checkbox(value=False, label="Override segment size", info="Override model default segment size instead of using the model default value.")
                        with gr.Row():
                            mdx23c_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.", visible=False)
                            mdx23c_overlap = gr.Slider(minimum=2, maximum=50, step=1, value=8, label="Overlap", info="Amount of overlap between prediction windows. Higher is better but slower.")
                            mdx23c_pitch_shift = gr.Slider(minimum=-24, maximum=24, step=1, value=0, label="Pitch shift", info="Shift audio pitch by a number of semitones while processing. may improve output for deep/high vocals.")
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            mdx23c_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Larger consumes more RAM but may process slightly faster.")
                            mdx23c_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                            mdx23c_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
        
        with gr.Tabs():
            with gr.Tab("Single processing"):
                with gr.Row():
                    mdx23c_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Row():
                    mdx23c_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx23c_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    mdx23c_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
            
            with gr.Tab("Batch processing"):
                with gr.Row():
                    mdx23c_batch_audio = gr.Files(label="Input Audio Files", file_types=["audio"], height=200)
                with gr.Row():
                    mdx23c_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx23c_batch_output = gr.Files(label="Output Files", interactive=False)
            
            with gr.Tab("Path / Link"):
                with gr.Row():
                    mdx23c_path_link = gr.Textbox(label="Input Path or Link", placeholder="Enter the path to the audio file, folder, or a link to a video/audio")
                with gr.Row():
                    mdx23c_path_link_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx23c_path_link_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    mdx23c_path_link_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("MDX-NET"):
        with gr.Group():
            with gr.Row():
                mdx_model = gr.Dropdown(value="UVR-MDX-NET Inst HQ 5", label="Select the Model", choices=list(MDXNET_MODELS.keys()), scale=3)
                mdx_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output Format", info="The format of the output audio file.", scale=1)
            with gr.Accordion("Advanced settings", open=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        mdx_denoise = gr.Checkbox(value=False, label="Denoise", info="Enable denoising after separation.")
                        with gr.Row():
                            mdx_hop_length = gr.Slider(minimum=32, maximum=2048, step=32, value=1024, label="Hop Length", info="Usually called stride in neural networks; only change if you know what you're doing.")
                            mdx_seg_size = gr.Slider(minimum=32, maximum=4000, step=32, value=256, label="Segment Size", info="Larger consumes more resources, but may give better results.")
                            mdx_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Amount of overlap between prediction windows. Higher is better but slower.")
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            mdx_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Larger consumes more RAM but may process slightly faster.")
                            mdx_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                            mdx_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
        
        with gr.Tabs():
            with gr.Tab("Single processing"):
                with gr.Row():
                    mdx_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Row():
                    mdx_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    mdx_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
            
            with gr.Tab("Batch processing"):
                with gr.Row():
                    mdx_batch_audio = gr.Files(label="Input Audio Files", file_types=["audio"], height=200)
                with gr.Row():
                    mdx_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx_batch_output = gr.Files(label="Output Files", interactive=False)
            
            with gr.Tab("Path / Link"):
                with gr.Row():
                    mdx_path_link = gr.Textbox(label="Input Path or Link", placeholder="Enter the path to the audio file, folder, or a link to a video/audio")
                with gr.Row():
                    mdx_path_link_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    mdx_path_link_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    mdx_path_link_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("VR ARCH"):
        with gr.Group():
            with gr.Row():
                vr_model = gr.Dropdown(value="1_HP-UVR", label="Select the Model", choices=list(VR_ARCH_MODELS.keys()), scale=3)
                vr_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output Format", info="The format of the output audio file.", scale=1)
            with gr.Accordion("Advanced settings", open=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            vr_post_process = gr.Checkbox(value=False, label="Post Process", info="Identify leftover artifacts within vocal output; may improve separation for some songs.")
                            vr_tta = gr.Checkbox(value=False, label="TTA", info="Enable Test-Time-Augmentation; slow but improves quality.")
                            vr_high_end_process = gr.Checkbox(value=False, label="High End Process", info="Mirror the missing frequency range of the output.")
                        with gr.Row():
                            vr_post_process_threshold = gr.Slider(minimum=0.1, maximum=0.3, step=0.1, value=0.2, label="Post Process Threshold", info="Threshold for post-processing.", visible=False)
                            vr_window_size = gr.Slider(minimum=320, maximum=1024, step=32, value=512, label="Window Size", info="Balance quality and speed. 1024 = fast but lower, 320 = slower but better quality.")
                            vr_aggression = gr.Slider(minimum=1, maximum=100, step=1, value=5, label="Agression", info="Intensity of primary stem extraction.")
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            vr_batch_size = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Batch Size", info="Larger consumes more RAM but may process slightly faster.")
                            vr_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                            vr_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
        
        with gr.Tabs():
            with gr.Tab("Single processing"):
                with gr.Row():
                    vr_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Row():
                    vr_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    vr_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    vr_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
            
            with gr.Tab("Batch processing"):
                with gr.Row():
                    vr_batch_audio = gr.Files(label="Input Audio Files", file_types=["audio"], height=200)
                with gr.Row():
                    vr_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    vr_batch_output = gr.Files(label="Output Files", interactive=False)
            
            with gr.Tab("Path / Link"):
                with gr.Row():
                    vr_path_link = gr.Textbox(label="Input Path or Link", placeholder="Enter the path to the audio file, folder, or a link to a video/audio")
                with gr.Row():
                    vr_path_link_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    vr_path_link_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    vr_path_link_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)

    with gr.Tab("Demucs"):
        with gr.Group():
            with gr.Row():
                demucs_model = gr.Dropdown(value="htdemucs_ft", label="Select the Model", choices=list(DEMUCS_MODELS.keys()), scale=3)
                demucs_output_format = gr.Dropdown(value="wav", choices=OUTPUT_FORMAT, label="Output Format", info="The format of the output audio file.", scale=1)
            with gr.Accordion("Advanced settings", open=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        demucs_segments_enabled = gr.Checkbox(value=True, label="Segment-wise processing", info="Enable segment-wise processing.")
                        with gr.Row():
                            demucs_seg_size = gr.Slider(minimum=1, maximum=100, step=1, value=40, label="Segment Size", info="Size of segments into which the audio is split. Higher = slower but better quality.")
                            demucs_overlap = gr.Slider(minimum=0.001, maximum=0.999, step=0.001, value=0.25, label="Overlap", info="Overlap between prediction windows. Higher = slower but better quality.")
                            demucs_shifts = gr.Slider(minimum=0, maximum=20, step=1, value=2, label="Shifts", info="Number of predictions with random shifts, higher = slower but better quality.")
                with gr.Column(variant='panel'):
                    with gr.Group():
                        with gr.Row():
                            demucs_norm_threshold = gr.Slider(minimum=0.1, maximum=1, step=0.1, value=0.9, label="Normalization threshold", info="The threshold for audio normalization.")
                            demucs_amp_threshold = gr.Slider(minimum=0.0, maximum=1, step=0.1, value=0.0, label="Amplification threshold", info="The threshold for audio amplification.")
        
        with gr.Tabs():
            with gr.Tab("Single processing"):
                with gr.Row():
                    demucs_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Row():
                    demucs_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    demucs_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    demucs_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
                with gr.Row():
                    demucs_stem3 = gr.Audio(label="Stem 3", type="filepath", interactive=False)
                    demucs_stem4 = gr.Audio(label="Stem 4", type="filepath", interactive=False)
                with gr.Row(visible=False) as stem6:
                    demucs_stem5 = gr.Audio(label="Stem 5", type="filepath", interactive=False)
                    demucs_stem6 = gr.Audio(label="Stem 6", type="filepath", interactive=False)
            
            with gr.Tab("Batch processing"):
                with gr.Row():
                    demucs_batch_audio = gr.Files(label="Input Audio Files", file_types=["audio"], height=200)
                with gr.Row():
                    demucs_batch_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    demucs_batch_output = gr.Files(label="Output Files", interactive=False)
            
            with gr.Tab("Path / Link"):
                with gr.Row():
                    demucs_path_link = gr.Textbox(label="Input Path or Link", placeholder="Enter the path to the audio file, folder, or a link to a video/audio")
                with gr.Row():
                    demucs_path_link_button = gr.Button("Separate!", variant="primary")
                with gr.Row():
                    demucs_path_link_stem1 = gr.Audio(label="Stem 1", type="filepath", interactive=False)
                    demucs_path_link_stem2 = gr.Audio(label="Stem 2", type="filepath", interactive=False)
                with gr.Row():
                    demucs_path_link_stem3 = gr.Audio(label="Stem 3", type="filepath", interactive=False)
                    demucs_path_link_stem4 = gr.Audio(label="Stem 4", type="filepath", interactive=False)
                with gr.Row(visible=False) as stem6:
                    demucs_path_link_stem5 = gr.Audio(label="Stem 5", type="filepath", interactive=False)
                    demucs_path_link_stem6 = gr.Audio(label="Stem 6", type="filepath", interactive=False)

    with gr.Tab("Settings"):
        with gr.Group():
            with gr.Row():
                model_file_dir = gr.Textbox(value="/tmp/PolUVR-models/", label="Directory to cache model files", info="The directory where model files are stored.", placeholder="/tmp/PolUVR-models/")
                output_dir = gr.Textbox(value="output", label="File output directory", info="The directory where output files will be saved.", placeholder="output")

        with gr.Accordion("Rename Stems", open=False):
            gr.Markdown(
                """
                Keys for automatic determination of input file names, stems, and models to simplify the construction of output file names.

                Keys:
                * **NAME** - Input File Name
                * **STEM** - Stem Name (e.g., Vocals, Instrumental)
                * **MODEL** - Model Name (e.g., BS-Roformer-Viperx-1297)

                > Example:
                > * **Usage:** NAME_(STEM)_MODEL
                > * **Output File Name:** Music_(Vocals)_BS-Roformer-Viperx-1297
                """
            )
            with gr.Row():
                vocals_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Vocals Stem", info="Output example: Music_(Vocals)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                instrumental_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Instrumental Stem", info="Output example: Music_(Instrumental)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                other_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Other Stem", info="Output example: Music_(Other)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                drums_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Drums Stem", info="Output example: Music_(Drums)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                bass_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Bass Stem", info="Output example: Music_(Bass)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
            with gr.Row():
                guitar_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Guitar Stem", info="Output example: Music_(Guitar)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")
                piano_stem = gr.Textbox(value="NAME_(STEM)_MODEL", label="Piano Stem", info="Output example: Music_(Piano)_BS-Roformer-Viperx-1297", placeholder="NAME_(STEM)_MODEL")

    with gr.Tab("Leaderboard"):
        with gr.Group():
            with gr.Row(equal_height=True):
                list_filter = gr.Dropdown(value="vocals", choices=["vocals", "instrumental", "drums", "bass", "guitar", "piano", "other"], label="List filter", info="Filter and sort the model list by 'stem'")
                list_limit = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="List limit", info="Limit the number of models shown.")
                list_button = gr.Button("Show list", variant="primary")

        output_list = gr.HTML(label="Leaderboard")

    roformer_override_seg_size.change(show_hide_params, inputs=[roformer_override_seg_size], outputs=[roformer_seg_size])
    mdx23c_override_seg_size.change(show_hide_params, inputs=[mdx23c_override_seg_size], outputs=[mdx23c_seg_size])
    vr_post_process.change(show_hide_params, inputs=[vr_post_process], outputs=[vr_post_process_threshold])

    demucs_model.change(update_stems, inputs=[demucs_model], outputs=stem6)

    list_button.click(leaderboard, inputs=[list_filter, list_limit], outputs=output_list)

    roformer_button.click(
        process_single_audio,
        inputs=[
            roformer_audio,
            roformer_model,
            model_file_dir,
            output_dir,
            roformer_output_format,
            roformer_norm_threshold,
            roformer_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[roformer_stem1, roformer_stem2],
    )
    roformer_batch_button.click(
        process_batch_audio,
        inputs=[
            roformer_batch_audio,
            roformer_model,
            model_file_dir,
            output_dir,
            roformer_output_format,
            roformer_norm_threshold,
            roformer_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[roformer_batch_output],
    )
    roformer_path_link_button.click(
        process_path_link,
        inputs=[
            roformer_path_link,
            roformer_model,
            model_file_dir,
            output_dir,
            roformer_output_format,
            roformer_norm_threshold,
            roformer_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[roformer_path_link_stem1, roformer_path_link_stem2],
    )

    mdx23c_button.click(
        process_single_audio,
        inputs=[
            mdx23c_audio,
            mdx23c_model,
            model_file_dir,
            output_dir,
            mdx23c_output_format,
            mdx23c_norm_threshold,
            mdx23c_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx23c_stem1, mdx23c_stem2],
    )
    mdx23c_batch_button.click(
        process_batch_audio,
        inputs=[
            mdx23c_batch_audio,
            mdx23c_model,
            model_file_dir,
            output_dir,
            mdx23c_output_format,
            mdx23c_norm_threshold,
            mdx23c_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx23c_batch_output],
    )
    mdx23c_path_link_button.click(
        process_path_link,
        inputs=[
            mdx23c_path_link,
            mdx23c_model,
            model_file_dir,
            output_dir,
            mdx23c_output_format,
            mdx23c_norm_threshold,
            mdx23c_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx23c_path_link_stem1, mdx23c_path_link_stem2],
    )

    mdx_button.click(
        process_single_audio,
        inputs=[
            mdx_audio,
            mdx_model,
            model_file_dir,
            output_dir,
            mdx_output_format,
            mdx_norm_threshold,
            mdx_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx_stem1, mdx_stem2],
    )
    mdx_batch_button.click(
        process_batch_audio,
        inputs=[
            mdx_batch_audio,
            mdx_model,
            model_file_dir,
            output_dir,
            mdx_output_format,
            mdx_norm_threshold,
            mdx_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx_batch_output],
    )
    mdx_path_link_button.click(
        process_path_link,
        inputs=[
            mdx_path_link,
            mdx_model,
            model_file_dir,
            output_dir,
            mdx_output_format,
            mdx_norm_threshold,
            mdx_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[mdx_path_link_stem1, mdx_path_link_stem2],
    )

    vr_button.click(
        process_single_audio,
        inputs=[
            vr_audio,
            vr_model,
            model_file_dir,
            output_dir,
            vr_output_format,
            vr_norm_threshold,
            vr_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[vr_stem1, vr_stem2],
    )
    vr_batch_button.click(
        process_batch_audio,
        inputs=[
            vr_batch_audio,
            vr_model,
            model_file_dir,
            output_dir,
            vr_output_format,
            vr_norm_threshold,
            vr_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[vr_batch_output],
    )
    vr_path_link_button.click(
        process_path_link,
        inputs=[
            vr_path_link,
            vr_model,
            model_file_dir,
            output_dir,
            vr_output_format,
            vr_norm_threshold,
            vr_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[vr_path_link_stem1, vr_path_link_stem2],
    )

    demucs_button.click(
        process_single_audio,
        inputs=[
            demucs_audio,
            demucs_model,
            model_file_dir,
            output_dir,
            demucs_output_format,
            demucs_norm_threshold,
            demucs_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[demucs_stem1, demucs_stem2, demucs_stem3, demucs_stem4, demucs_stem5, demucs_stem6],
    )
    demucs_batch_button.click(
        process_batch_audio,
        inputs=[
            demucs_batch_audio,
            demucs_model,
            model_file_dir,
            output_dir,
            demucs_output_format,
            demucs_norm_threshold,
            demucs_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[demucs_batch_output],
    )
    demucs_path_link_button.click(
        process_path_link,
        inputs=[
            demucs_path_link,
            demucs_model,
            model_file_dir,
            output_dir,
            demucs_output_format,
            demucs_norm_threshold,
            demucs_amp_threshold,
            vocals_stem,
            instrumental_stem,
            other_stem,
            drums_stem,
            bass_stem,
            guitar_stem,
            piano_stem,
        ],
        outputs=[demucs_path_link_stem1, demucs_path_link_stem2, demucs_path_link_stem3, demucs_path_link_stem4, demucs_path_link_stem5, demucs_path_link_stem6],
    )

def main():
    app.queue().launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        debug=True,
        show_error=True,
    )

if __name__ == "__main__":
    main()
