import argparse
import time

import datasets
import jax
import jax.numpy as jnp
from datasets import concatenate_datasets, load_dataset
from flax.core.frozen_dict import freeze
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.sharding import PartitionSpec as P
from transformers import WhisperConfig, WhisperProcessor
import numpy as np
from whisper_jax import FlaxWhisperForConditionalGeneration, InferenceState, PjitPartitioner
import os
import librosa
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
datasets.logging.set_verbosity(datasets.logging.CRITICAL)

cc.set_cache_dir("./jax_cache")
#jax.config.update("jax_array", True)

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


def main():

    # processors/tokenizers are the same for all models, so just load from tiny and preprocess once
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model, params = FlaxWhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3",
        _do_init=False,
        dtype=jnp.bfloat16,
    )
    def all_language_tokens():
        result = []
        for token, token_id in zip(processor.tokenizer.all_special_tokens,processor.tokenizer.all_special_ids):
            if token.strip("<|>") in LANGUAGES:
                result.append(token_id)
        return tuple(result)#[: self.num_languages]
    #print(all_language_tokens())
    #params = np.asarray(params)
    #params = model.init_weights(model.key, model.input_shape,params=params)
    def init_fn():
        input_shape = (1, 128, 3000)

        input_features = jnp.zeros(input_shape, dtype="f4")
        input_features = input_features.at[(..., -1)].set(model.config.eos_token_id)

        decoder_input_ids = jnp.zeros((input_shape[0], 1), dtype="i4")
        decoder_attention_mask = jnp.ones_like(decoder_input_ids)

        batch_size, sequence_length = decoder_input_ids.shape
        decoder_position_ids = jnp.broadcast_to(jnp.arange(sequence_length)[None, :], (batch_size, sequence_length))

        rng = jax.random.PRNGKey(0)
        init_params = model.module.init(
            rng,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            return_dict=False,
        )
        return init_params

    # Axis names metadata
    param_axes = jax.eval_shape(init_fn)["params_axes"]

    # Create InferenceState, since the partitioner expects it
    state = InferenceState(
        step=jnp.array(0),
        params=freeze(model.params_shape_tree),
        params_axes=freeze(param_axes),
        flax_mutables=None,
        flax_mutables_axes=param_axes,
    )

    partitioner = PjitPartitioner(
        num_partitions=1,
        #model_parallel_submesh=(2,2,1,1),
        logical_axis_rules=logical_axis_rules_dp,
    )

    mesh_axes = partitioner.get_mesh_axes(state)
    params_spec = mesh_axes.params

    p_shard_params = partitioner.partition(model.to_bf16, (params_spec,), params_spec)

    def generate(params, input_features):
        output_ids = model.generate(input_features, params=params,language="en").sequences
        return output_ids

    p_generate = partitioner.partition(
        generate,
        in_axis_resources=(params_spec, P("data")),
        out_axis_resources=P("data"),
    )
    params = jax.device_put(params,jax.devices()[0])
    # This will auto-magically run in mesh context
    params = p_shard_params(freeze(params))
    silereo_model = load_silero_vad()

    start = time.time()
    supported_formats = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    # 遍历文件夹
    for root, _, files in os.walk("/bucket/aurora_filtered"):
        for file in files:
            # 检查文件扩展名是否是支持的格式
            if file.lower().endswith(supported_formats):
                file_path = os.path.join(root, file)

                # 使用 librosa 加载音频文件
                audio_data, sample_rate = librosa.load(file_path, sr=16000)  # sr=None 保持原始采样率
                print(f"Successfully loaded {file_path}")
                print(f"Sample Rate: {sample_rate}, Audio Data Length: {len(audio_data)}")
                speech_timestamps = get_speech_timestamps(
                    audio_data,
                    silereo_model,
                    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
                )
                stacked_audio = []
                logits = None
                i = 0
                for timestamp in speech_timestamps:
                    start, end = int(timestamp["start"] * sample_rate), int(timestamp["end"] * sample_rate)
                    segment = audio_data[start:end]
                    
                    # 对片段进行预处理
                    processed_segment = processor(segment, sampling_rate=16000, return_tensors="np")
                    stacked_audio.append(processed_segment.input_features[0])
                    if i < 4:
                        encoder_outputs = model.encode(input_features=processed_segment.input_features,params=params)
                        decoder_start_token_id = model.config.decoder_start_token_id
                        decoder_input_ids = jnp.ones((processed_segment.input_features.shape[0], 1), dtype="i4") * decoder_start_token_id
                        outputs = model.decode(decoder_input_ids, encoder_outputs,params=params)
                        if logits is None:
                            logits = outputs.logits
                        else:
                            logits_add = outputs.logits
                            logits += logits_add
                        i += 1
                mask = jnp.ones(logits.shape[-1], dtype=jnp.bool)
                mask = mask.at[jnp.array(all_language_tokens())].set(False)
                logits = logits.at[:,:, mask].set(-jnp.inf)
                language_tokens = jnp.argmax(logits,axis=-1)
                print(processor.decode(language_tokens[0,0]))
                breakpoint()

               
                stacked_audio = np.stack(stacked_audio)
                stacked_audio = jnp.asarray(stacked_audio)


                pred_ids = p_generate(params, stacked_audio)
                # post-process: convert tokens ids to text string
                transcription = processor.batch_decode(pred_ids, skip_special_tokens=True)
                breakpoint()
    runtime = time.time() - start
    print(f"{runtime:.06}")


if __name__ == "__main__":
    main()
