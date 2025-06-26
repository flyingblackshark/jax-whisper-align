
"""Audio inference module for JAX Whisper alignment."""

import os
import re
from typing import Dict, List, Tuple, Optional, Any

import jax
import jax.numpy as jnp
import librosa
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import WhisperProcessor

from jaxwhisperalign.align import align, load_align_model, SingleSegment
from jaxwhisperalign.modeling_flax_whisper import FlaxWhisperForConditionalGeneration
from jaxwhisperalign.vad import (
    VadOptions,
    get_speech_timestamps,
    merge_segments,
)
# Constants
SAMPLE_RATE = 16000
MODEL_NAME = "openai/whisper-large-v3"
DEFAULT_MAX_SPEECH_DURATION = 30
DEFAULT_MIN_SILENCE_DURATION = 160

def format_time(seconds: float) -> str:
    """Convert seconds to SRT time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
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



# Logical axis rules for data parallelism
LOGICAL_AXIS_RULES_DP = [
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

# Global caches
global_align_model_cache: Dict[str, Tuple[Any, Any]] = {}
whisper_model_cache: Optional[Any] = None
whisper_model_params_cache: Optional[Any] = None
whisper_model_processor_cache: Optional[WhisperProcessor] = None
def get_align_model_with_cache(language_code: str) -> Tuple[Any, Any]:
    """Get alignment model with caching.
    
    Args:
        language_code: Language code for the model
        
    Returns:
        Tuple of (model, metadata)
    """
    if language_code in global_align_model_cache:
        print(f"Using cached align model for language: {language_code}")
        return global_align_model_cache[language_code]
    
    print(f"Loading align model for language: {language_code}")
    model_a, metadata = load_align_model(language_code=language_code)
    global_align_model_cache[language_code] = (model_a, metadata)
    return model_a, metadata

def remove_symbols(text: str) -> str:
    """Remove special symbols from text."""
    return re.sub(r"<\|([^|]+)\|>", r"\1", text)

def process_audio(file_path: str, language: Optional[str] = None) -> Tuple[List[Dict[str, Any]], str]:
    """Process audio file and return aligned segments.
    
    Args:
        file_path: Path to audio file
        language: Language code for transcription. If None, auto-detect language.
        
    Returns:
        Tuple of (segments, detected_language)
    """
    batch_size = jax.device_count()
    language_detect_batch_size = jax.device_count()
    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(device_mesh, axis_names=("data",))
    
    # Load models with caching
    global whisper_model_processor_cache, whisper_model_cache, whisper_model_params_cache
    
    if whisper_model_processor_cache is None:
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        whisper_model_processor_cache = processor
    else:
        processor = whisper_model_processor_cache
        
    if whisper_model_cache is None or whisper_model_params_cache is None:
        model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            _do_init=False,
            dtype=jnp.bfloat16,
        )
        whisper_model_cache = model
        whisper_model_params_cache = params
    else:
        model = whisper_model_cache
        params = whisper_model_params_cache

    def all_language_tokens() -> List[int]:
        """Get all language token IDs."""
        return [
            token_id for token, token_id in zip(
                processor.tokenizer.all_special_tokens,
                processor.tokenizer.all_special_ids
            )
            if token.strip("<|>") in LANGUAGES
        ]
    
    replicate_sharding = NamedSharding(mesh, PartitionSpec(None))
    x_sharding = NamedSharding(mesh, PartitionSpec("data"))

    def generate(params, input_features, language):
        output_ids = model.generate(input_features, params=params, language=language).sequences
        return output_ids
    
    jitted_generate = jax.jit(generate, in_shardings=(replicate_sharding, x_sharding), out_shardings=x_sharding, static_argnums=(2,))
    params = model.to_bf16(params)


    # Load and process audio
    audio_data, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    print(f"Successfully loaded {file_path}")
    
    # Voice activity detection
    vad_parameters = VadOptions(
        max_speech_duration_s=DEFAULT_MAX_SPEECH_DURATION,
        min_silence_duration_ms=DEFAULT_MIN_SILENCE_DURATION,
    )
    active_segments = get_speech_timestamps(audio_data, vad_parameters)
    clip_timestamps = merge_segments(active_segments, vad_parameters)
    
    audio_segments = []
    segments_info = []
    logits = None
    i = 0
    
    # Process audio segments
    for timestamp in clip_timestamps:
        segment = audio_data[timestamp["start"]:timestamp["end"]]
        processed_segment = processor(segment, sampling_rate=SAMPLE_RATE, return_tensors="np")
        audio_segments.append(processed_segment.input_features[0])
        segments_info.append((os.path.basename(file_path), timestamp["start"], timestamp["end"]))

    def language_detect_wrap(params, input_features):
        encoder_outputs = model.encode(input_features=input_features, params=params)
        decoder_start_token_id = model.config.decoder_start_token_id
        decoder_input_ids = jnp.ones((input_features.shape[0], 1), dtype="i4") * decoder_start_token_id
        outputs = model.decode(decoder_input_ids, encoder_outputs, params=params)
        return outputs.logits
    
    
    params = jax.device_put(params, replicate_sharding)
    # Language detection or use provided language
    if language is None or language == "auto":
        # Auto-detect language
        jitted_language_detect_func = jax.jit(
            language_detect_wrap, 
            in_shardings=(replicate_sharding, x_sharding), 
            out_shardings=x_sharding
        )
        language_detect_segments = np.stack(audio_segments[:language_detect_batch_size], axis=0)
        ld_padding = language_detect_batch_size - language_detect_segments.shape[0]
        padded_language_detect_segments = np.pad(
            language_detect_segments, ((0, ld_padding), (0, 0), (0, 0))
        )
        padded_language_detect_segments = jnp.asarray(padded_language_detect_segments)
        if logits is None:
            logits = jitted_language_detect_func(params, padded_language_detect_segments)
        else:
            logits += jitted_language_detect_func(params, padded_language_detect_segments)
        
        def language_mask_wrap(logits):
            logits = np.sum(logits, axis=0, keepdims=True)
            mask = np.ones(logits.shape[-1], dtype=np.bool)
            mask[all_language_tokens()] = False
            logits = np.where(mask, -np.inf, logits)
            language_tokens = np.argmax(logits, axis=-1)
            return language_tokens
        
        language_tokens = language_mask_wrap(np.asarray(logits))
        detected_language = processor.decode(language_tokens[0, 0])
    else:
        # Use provided language
        detected_language = language
        print(f"Using specified language: {language}")

    
    # Batch transcription
    rounds = (len(audio_segments) - 1) // batch_size + 1
    pred_ids_result = None
    
    for i in range(rounds):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        stacked_audio = np.stack(audio_segments[start_idx:end_idx])
        
        padding_size = batch_size - stacked_audio.shape[0]
        padded_stacked_audio = np.pad(stacked_audio, ((0, padding_size), (0, 0), (0, 0)))
        padded_stacked_audio = jnp.asarray(padded_stacked_audio)
        
        pred_ids = jitted_generate(params, padded_stacked_audio, detected_language)
        pred_ids = pred_ids[:batch_size - padding_size]
        pred_ids = np.asarray(pred_ids)
        
        if pred_ids_result is None:
            pred_ids_result = pred_ids
        else:
            pred_ids_result = np.concatenate([pred_ids_result, pred_ids], axis=0)
    # Decode transcriptions
    transcriptions = processor.batch_decode(pred_ids_result, skip_special_tokens=True)

    # Alignment
    model_a, metadata = get_align_model_with_cache(
        language_code=remove_symbols(detected_language)
    )
    
    segs = [
        SingleSegment(start=start_time, end=end_time, text=transcription)
        for (_, start_time, end_time), transcription in zip(segments_info, transcriptions)
    ]
    
    result = align(
        segs, model_a, metadata, audio_data, mesh, return_char_alignments=False
    )

    return result["segments"], detected_language