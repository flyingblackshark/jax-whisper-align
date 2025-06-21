
import jax
import jax.numpy as jnp
from transformers import WhisperProcessor
import numpy as np
from whisper_jax import FlaxWhisperForConditionalGeneration
import os
import librosa
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vad import (
    VadOptions,
    get_speech_timestamps,
    merge_segments,
)
import re
from align import load_align_model, align, SingleSegment
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
    "yue": "cantonese",
}



logical_axis_rules_dp = [
    ("batch", "data"),
    ("mlp", None),
    ("heads", None),
    ("vocab", None),
    ("embed", None),
    ("embed", None),
    ("joined_kv", None),
    ("kv", None),
    ("length", None),
    ("num_mel", None),
    ("channels", None),
]
global_align_model_cache = {}
whisper_model_cache = None
whisper_model_params_cache = None
whisper_model_processor_cache = None
def get_align_model_with_cache(language_code):
    if language_code in global_align_model_cache:
        print(f"Using cached align model for language: {language_code}")
        return global_align_model_cache[language_code]
    
    print(f"Loading align model for language: {language_code}")
    model_a, metadata = load_align_model(language_code=language_code)
    global_align_model_cache[language_code] = (model_a, metadata)
    return model_a, metadata

def remove_symbols(text):
    return re.sub(r"<\|([^|]+)\|>", r"\1", text)

def process_audio(file_path):
    BATCH_SIZE = 16
    LANGUAGE_DETECT_BATCH_SIZE = 8
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=("data",))
    
    global whisper_model_processor_cache
    global whisper_model_cache
    global whisper_model_params_cache
    if whisper_model_processor_cache is None:
        processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        whisper_model_processor_cache = processor
    else:
        processor = whisper_model_processor_cache
    if whisper_model_cache is None or whisper_model_params_cache is None:
        model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3",
            _do_init=False,
            dtype=jnp.bfloat16,
        )
        whisper_model_cache = model
        whisper_model_params_cache = params
    else:
        model = whisper_model_cache
        params = whisper_model_params_cache

    def all_language_tokens():
        result = []
        for token, token_id in zip(processor.tokenizer.all_special_tokens, processor.tokenizer.all_special_ids):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)
    
    replicate_sharding = NamedSharding(mesh, PartitionSpec(None))
    x_sharding = NamedSharding(mesh, PartitionSpec("data"))

    def generate(params, input_features, language):
        output_ids = model.generate(input_features, params=params, language=language).sequences
        return output_ids
    
    jitted_generate = jax.jit(generate, in_shardings=(replicate_sharding, x_sharding), out_shardings=x_sharding, static_argnums=(2,))
    params = model.to_bf16(params)


    audio_data, sample_rate = librosa.load(file_path, sr=16000)
    print(f"Successfully loaded {file_path}")
    
    vad_parameters = VadOptions(
        max_speech_duration_s=30,
        min_silence_duration_ms=160,
    )
    active_segments = get_speech_timestamps(audio_data, vad_parameters)
    clip_timestamps = merge_segments(active_segments, vad_parameters)
    
    audio_segments = []
    segments_info = []
    logits = None
    i = 0
    
    for timestamp in clip_timestamps:
        segment = audio_data[timestamp["start"]:timestamp["end"]]
        processed_segment = processor(segment, sampling_rate=16000, return_tensors="np")
        audio_segments.append(processed_segment.input_features[0])
        segments_info.append((os.path.basename(file_path), timestamp["start"], timestamp["end"]))

    def language_detect_wrap(params, input_features):
        encoder_outputs = model.encode(input_features=input_features, params=params)
        decoder_start_token_id = model.config.decoder_start_token_id
        decoder_input_ids = jnp.ones((input_features.shape[0], 1), dtype="i4") * decoder_start_token_id
        outputs = model.decode(decoder_input_ids, encoder_outputs, params=params)
        return outputs.logits
    
    
    params = jax.device_put(params, replicate_sharding)
    jitted_language_detect_func = jax.jit(language_detect_wrap, in_shardings=(replicate_sharding, x_sharding), out_shardings=x_sharding)
    language_detect_segments = jnp.stack(audio_segments[:LANGUAGE_DETECT_BATCH_SIZE], axis=0)
    LD_B_padding = LANGUAGE_DETECT_BATCH_SIZE - language_detect_segments.shape[0]
    padded_language_detect_segments = jnp.pad(language_detect_segments, ((0, LD_B_padding), (0, 0), (0, 0)))
    if logits is None:
        logits = jitted_language_detect_func(params, padded_language_detect_segments)
    else:
        logits += jitted_language_detect_func(params, padded_language_detect_segments)
    def language_mask_wrap(logits):
        logits = jnp.sum(logits, axis=0, keepdims=True)
        mask = jnp.ones(logits.shape[-1], dtype=jnp.bool)
        mask = mask.at[jnp.array(all_language_tokens())].set(False)
        logits = jnp.where(mask, -jnp.inf, logits)
        language_tokens = jnp.argmax(logits, axis=-1)
        return language_tokens
    
    language_tokens = jax.jit(language_mask_wrap)(logits)
    detected_language = processor.decode(language_tokens[0, 0])

    
    rounds = (len(audio_segments) - 1) // BATCH_SIZE + 1
    pred_ids_result = None
    for i in range(rounds):
        stacked_audio = audio_segments[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        stacked_audio = np.stack(stacked_audio)
        padding_size = BATCH_SIZE - stacked_audio.shape[0]
        padded_stacked_audio = np.pad(stacked_audio, ((0, padding_size), (0, 0), (0, 0)))
        padded_stacked_audio = jnp.asarray(padded_stacked_audio)
        pred_ids = jitted_generate(params, padded_stacked_audio, detected_language)
        pred_ids = pred_ids[:BATCH_SIZE - padding_size]
        pred_ids = np.asarray(pred_ids)
        if pred_ids_result is None:
            pred_ids_result = pred_ids
        else:
            pred_ids_result = np.concatenate([pred_ids_result, pred_ids], axis=0)
    transcriptions = processor.batch_decode(pred_ids_result, skip_special_tokens=True)


    model_a, metadata = get_align_model_with_cache(language_code=remove_symbols(detected_language))
    segs = []
    for (_, start_time, end_time), transcription in zip(segments_info, transcriptions):
        segs.append(SingleSegment(start=start_time, end=end_time, text=transcription))
    result = align(segs, model_a, metadata, audio_data, mesh, return_char_alignments=False)

    return result["segments"], detected_language