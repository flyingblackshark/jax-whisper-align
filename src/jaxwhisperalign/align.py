from dataclasses import dataclass
import time
from dataclasses import dataclass
from typing import Iterable, Union, List, TypedDict, Optional

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import FlaxWav2Vec2ForCTC, Wav2Vec2Processor
import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
import concurrent.futures
import threading
from functools import partial

class SingleWordSegment(TypedDict):
    word: str
    start: float
    end: float
    score: float

class SingleCharSegment(TypedDict):
    char: str
    start: float
    end: float
    score: float


class SingleSegment(TypedDict):
    start: float
    end: float
    text: str


class SingleAlignedSegment(TypedDict):
    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    segments: List[SingleSegment]
    language: str


class AlignedTranscriptionResult(TypedDict):
    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]

# Constants
PUNKT_ABBREVIATIONS = ['dr', 'vs', 'mr', 'mrs', 'prof']
LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]
BATCH_SIZE = 16
SAMPLE_RATE = 16000
MAX_LENGTH_SECONDS = 32
FRAME_SHIFT = 320
FRAME_OFFSET = 80

DEFAULT_ALIGN_MODELS_HF = {
    "en": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal",
    "nn": "NbAiLab/nb-wav2vec2-300m-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "it": "jonatasgrosman/wav2vec2-large-xlsr-53-italian",
    "fr": "jonatasgrosman/wav2vec2-large-xlsr-53-french",
    "de": "jonatasgrosman/wav2vec2-large-xlsr-53-german",
    "es": "jonatasgrosman/wav2vec2-large-xlsr-53-spanish",
}

def interpolate_nans(x, method='nearest'):
    if x.notnull().sum() > 1:
        return x.interpolate(method=method).ffill().bfill()
    else:
        return x.ffill().bfill()

def load_align_model(mesh,language_code, model_name=None, model_dir=None):
    if model_name is None:
        if language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            print(f"There is no default alignment model set for this language ({language_code}).\                Please find a wav2vec2.0 model finetuned on this language in https://huggingface.co/models, then pass the model name in --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    align_model = FlaxWav2Vec2ForCTC.from_pretrained(model_name)
    align_model.params = jax.device_put(align_model.params, NamedSharding(mesh, PartitionSpec()))
    align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary}

    return align_model, align_metadata

def process_ctc_emissions(
    transcript: Iterable[SingleSegment],
    model,
    audio: np.ndarray,
    mesh
) -> List[np.ndarray]:
    """
    处理CTC模型，生成emissions
    
    Args:
        transcript: 转录段列表
        model: CTC模型
        model_type: 模型类型
        audio: 音频数据
        mesh: JAX mesh
        
    Returns:
        List[np.ndarray]: 每个段的emissions列表
    """
    x_sharding = NamedSharding(mesh, PartitionSpec("data"))
    pre_emissions = []
    pre_waveform_segments = []
    pre_segment_lengths = []
    
    def pad_and_stack_waveforms(waveforms, max_length):
        """Pad waveforms to the same length and stack them."""
        padded_waveforms = [
            np.pad(w, ((0, 0), (0, max_length - w.shape[-1]))) 
            for w in waveforms
        ]
        return np.concatenate(padded_waveforms, axis=0)
    
    def slice_emissions(emissions, lengths):
        """Slice emissions based on actual audio lengths."""
        return [np.asarray(emissions[i, :(l - FRAME_OFFSET)//FRAME_SHIFT,:]) for i, l in enumerate(lengths)]
    
    def model_wrap(waveform_seg):
        """Wrapper function for the CTC model."""
        return jnp.log(jax.nn.softmax(model(waveform_seg).logits, axis=-1))
    
    jitted_model_wrap = jax.jit(model_wrap, in_shardings=x_sharding, out_shardings=x_sharding)
    
    ctc_start_time = time.time()
    
    for sdx, segment in enumerate(transcript):
        t1 = segment["start"]
        t2 = segment["end"]
        
        f1 = t1
        f2 = t2
        
        waveform_segment = audio[:, f1:f2]
        MAX_LENGTH = MAX_LENGTH_SECONDS * SAMPLE_RATE
        pre_waveform_segments.append(waveform_segment)
        pre_segment_lengths.append(waveform_segment.shape[-1])
        
        if len(pre_waveform_segments) == BATCH_SIZE or sdx == len(transcript) - 1:
            waveform_segments_padded = pad_and_stack_waveforms(pre_waveform_segments, MAX_LENGTH)
            lengths = np.asarray(pre_segment_lengths)
            batch_padding = BATCH_SIZE - waveform_segments_padded.shape[0]
            waveform_segments_padded = np.pad(waveform_segments_padded, ((0, batch_padding), (0, 0)))
            
            emissions_batch = jitted_model_wrap(waveform_segments_padded)

            emissions_batch = emissions_batch[:BATCH_SIZE - batch_padding]
            pre_emissions.extend(slice_emissions(emissions_batch, lengths))
            pre_waveform_segments = []
            pre_segment_lengths = []
    
    ctc_end_time = time.time()
    print(f"CTC processing time: {ctc_end_time - ctc_start_time:.2f}s")
    
    return pre_emissions



def process_segment_alignment(
    segment_data,
    model_dictionary: dict,
    model_lang: str,
    return_char_alignments: bool,
    interpolate_method: str
):
    """
    处理单个segment的对齐
    
    Args:
        segment_data: 包含segment信息和emission的元组
        model_dictionary: 模型字典
        model_lang: 模型语言
        return_char_alignments: 是否返回字符对齐
        interpolate_method: 插值方法
        
    Returns:
        对齐后的segment或None（如果失败）
    """
    sdx, segment, emission = segment_data
    
    t1 = segment["start"]
    t2 = segment["end"]
    text = segment["text"]
    
    aligned_seg: SingleAlignedSegment = {
        "start": t1,
        "end": t2,
        "text": text,
        "words": [],
    }
    
    if return_char_alignments:
        aligned_seg["chars"] = []
    
    if len(segment["clean_char"]) == 0:
        print(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original...')
        return aligned_seg
    
    text_clean = "".join(segment["clean_char"])
    tokens = [model_dictionary[c] for c in text_clean]
    
    # Find blank token ID
    blank_id = 0
    for char, code in model_dictionary.items():
        if char in ['[pad]', '<pad>']:
            blank_id = code
            break
    
    # 这里是多线程处理的核心部分
    trellis = get_trellis(emission, tokens, blank_id)
    path = backtrack(trellis, emission, tokens, blank_id)
    
    if path is None:
        print(f'Failed to align segment ("{segment["text"]}"): backtrack failed, resorting to original...')
        return aligned_seg
    
    char_segments = merge_repeats(path, text_clean)
    
    duration = t2 - t1
    ratio = duration / (trellis.shape[0] - 1)
    
    char_segments_arr = []
    word_idx = 0
    for cdx, char in enumerate(text):
        start, end, score = None, None, None
        if cdx in segment["clean_cdx"]:
            char_seg = char_segments[segment["clean_cdx"].index(cdx)]
            start = round(char_seg.start * ratio + t1, 3)
            end = round(char_seg.end * ratio + t1, 3)
            score = round(char_seg.score, 3)
        
        char_segments_arr.append({
            "char": char,
            "start": start,
            "end": end,
            "score": score,
            "word-idx": word_idx,
        })
        
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            word_idx += 1
        elif cdx == len(text) - 1 or text[cdx+1] == " ":
            word_idx += 1
    
    char_segments_arr = pd.DataFrame(char_segments_arr)
    
    aligned_subsegments = []
    
    char_segments_arr["sentence-idx"] = None
    for sdx_inner, (sstart, send) in enumerate(segment["sentence_spans"]):
        curr_chars = char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send)]
        char_segments_arr.loc[(char_segments_arr.index >= sstart) & (char_segments_arr.index <= send), "sentence-idx"] = sdx_inner
        
        sentence_text = text[sstart:send]
        sentence_start = curr_chars["start"].min()
        end_chars = curr_chars[curr_chars["char"] != ' ']
        sentence_end = end_chars["end"].max()
        sentence_words = []
        
        for word_idx in curr_chars["word-idx"].unique():
            word_chars = curr_chars.loc[curr_chars["word-idx"] == word_idx]
            word_text = "".join(word_chars["char"].tolist()).strip()
            if len(word_text) == 0:
                continue
            
            word_chars = word_chars[word_chars["char"] != " "]
            
            word_start = word_chars["start"].min()
            word_end = word_chars["end"].max()
            word_score = round(word_chars["score"].mean(), 3)
            
            # Create word segment with available data
            word_segment = {"word": word_text}
            
            if not np.isnan(word_start):
                word_segment["start"] = round(word_start, 3)
            if not np.isnan(word_end):
                word_segment["end"] = round(word_end, 3)
            if not np.isnan(word_score):
                word_segment["score"] = word_score
            
            sentence_words.append(word_segment)
        
        aligned_subsegments.append({
            "text": sentence_text,
            "start": round(sentence_start, 3) if not np.isnan(sentence_start) else None,
            "end": round(sentence_end, 3) if not np.isnan(sentence_end) else None,
            "words": sentence_words,
        })
        
        if return_char_alignments:
            curr_chars = curr_chars[["char", "start", "end", "score"]]
            curr_chars.fillna(-1, inplace=True)
            curr_chars = curr_chars.to_dict("records")
            curr_chars = [{key: val for key, val in char.items() if val != -1} for char in curr_chars]
            aligned_subsegments[-1]["chars"] = curr_chars
    
    # Process aligned subsegments
    if aligned_subsegments:
        aligned_subsegments_df = pd.DataFrame(aligned_subsegments)
        aligned_subsegments_df["start"] = interpolate_nans(
            aligned_subsegments_df["start"], method=interpolate_method
        )
        aligned_subsegments_df["end"] = interpolate_nans(
            aligned_subsegments_df["end"], method=interpolate_method
        )
        
        # Aggregate segments
        agg_dict = {"text": " ".join, "words": "sum"}
        if model_lang in LANGUAGES_WITHOUT_SPACES:
            agg_dict["text"] = "".join
        if return_char_alignments:
            agg_dict["chars"] = "sum"
        
        aligned_subsegments_df = aligned_subsegments_df.groupby(
            ["start", "end"], as_index=False
        ).agg(agg_dict)
        
        return aligned_subsegments_df.to_dict('records')
    
    return [aligned_seg]

def align(
    transcript: Iterable[SingleSegment],
    model,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, np.ndarray],
    mesh,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    max_workers: int = None,  # 新增参数控制线程数
) -> AlignedTranscriptionResult:

    if len(audio.shape) == 1:
        audio = audio[np.newaxis,:]
    
    MAX_DURATION = audio.shape[1]

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]

    total_segments = len(transcript)
    for sdx, segment in enumerate(transcript):

        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")
            
        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        # Split text into words based on language
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = list(text)

        # Clean characters and track their indices
        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()

            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")
            
            # Skip leading/trailing whitespace and check if char is in dictionary
            if (num_leading <= cdx <= len(text) - num_trailing - 1 and 
                char_ in model_dictionary):
                clean_char.append(char_)
                clean_cdx.append(cdx)

        # Find valid word indices
        clean_wdx = [
            wdx for wdx, wrd in enumerate(per_word)
            if any(c.lower() in model_dictionary for c in wrd)
        ]

        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(PUNKT_ABBREVIATIONS)
        sentence_splitter = PunktSentenceTokenizer(punkt_param)
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment["clean_char"] = clean_char
        segment["clean_cdx"] = clean_cdx
        segment["clean_wdx"] = clean_wdx
        segment["sentence_spans"] = sentence_spans
    
    # 使用封装的CTC处理函数
    pre_emissions = process_ctc_emissions(transcript, model, model_type, audio, mesh)
    
    # 准备多线程处理的数据
    segment_data_list = []
    for sdx, segment in enumerate(transcript):
        if sdx < len(pre_emissions):  # 确保有对应的emission
            segment_data_list.append((sdx, segment, pre_emissions[sdx]))
    
    # 多线程处理对齐
    aligned_segments = []
    
    # 如果没有指定max_workers，使用CPU核心数
    if max_workers is None:
        max_workers = min(len(segment_data_list), 128)  # 限制最大线程数为8
    
    process_func = partial(
        process_segment_alignment,
        model_dictionary=model_dictionary,
        model_lang=model_lang,
        return_char_alignments=return_char_alignments,
        interpolate_method=interpolate_method
    )
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_segment = {executor.submit(process_func, segment_data): segment_data 
                           for segment_data in segment_data_list}
        
        # 收集结果（保持原始顺序）
        results = [None] * len(segment_data_list)
        for future in concurrent.futures.as_completed(future_to_segment):
            segment_data = future_to_segment[future]
            sdx = segment_data[0]
            try:
                result = future.result()
                results[sdx] = result
            except Exception as exc:
                print(f'Segment {sdx} generated an exception: {exc}')
                # 创建默认的aligned_seg作为fallback
                segment = segment_data[1]
                results[sdx] = [{
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "words": [],
                }]
    
    # 展平结果
    for result in results:
        if result:
            if isinstance(result, list):
                aligned_segments.extend(result)
            else:
                aligned_segments.append(result)

    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        if "words" in segment:
            word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}



def get_trellis(emission, tokens, blank_id=0):
    """Compute the trellis matrix for CTC alignment."""
    num_frame = emission.shape[0]
    num_tokens = len(tokens)
    
    # Initialize trellis
    trellis = np.empty((num_frame + 1, num_tokens + 1))
    trellis[0, 0] = 0
    trellis[1:, 0] = np.cumsum(emission[:, blank_id], 0)
    trellis[0, -num_tokens:] = -float("inf")
    trellis[-num_tokens:, 0] = float("inf")

    # Fill trellis using dynamic programming
    for t in range(num_frame):
        trellis[t + 1, 1:] = np.maximum(
            trellis[t, 1:] + emission[t, blank_id],  # Stay in same state
            trellis[t, :-1] + emission[t, tokens],   # Advance to next token
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

def backtrack(trellis, emission, tokens, blank_id=0):
    """Backtrack through the trellis to find the optimal alignment path."""
    j = trellis.shape[1] - 1
    t_start = np.argmax(trellis[:, j])

    path = []
    for t in range(t_start, 0, -1):
        # Calculate scores for staying vs advancing
        stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
        changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

        # Choose the better path and calculate probability
        token_idx = tokens[j - 1] if changed > stayed else blank_id
        prob = np.exp(emission[t - 1, token_idx])
        path.append(Point(j - 1, t - 1, prob))

        # Advance token index if we took the "changed" path
        if changed > stayed:
            j -= 1
            if j == 0:
                break
    else:
        # Failed to reach the beginning
        return None
    
    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start

def merge_repeats(path, transcript):
    """Merge consecutive path points with the same token index."""
    if not path:
        return []
    
    segments = []
    i1 = 0
    
    while i1 < len(path):
        i2 = i1
        # Find all consecutive points with the same token index
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        
        # Calculate average score for this segment
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    
    return segments

def merge_words(segments, separator="|"):
    """Merge character segments into word segments."""
    if not segments:
        return []
    
    words = []
    i1 = 0
    
    while i1 < len(segments):
        i2 = i1
        # Find segments until separator or end
        while i2 < len(segments) and segments[i2].label != separator:
            i2 += 1
        
        # Create word from character segments
        if i1 != i2:
            segs = segments[i1:i2]
            word = "".join(seg.label for seg in segs)
            
            # Calculate weighted average score
            total_length = sum(seg.length for seg in segs)
            if total_length > 0:
                score = sum(seg.score * seg.length for seg in segs) / total_length
            else:
                score = 0.0
            
            words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
        
        # Skip separator
        i1 = i2 + 1
    
    return words
