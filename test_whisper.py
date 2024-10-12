import os
import logging
import sys
import torch
import torchaudio
import whisper
import pandas as pd
import numpy as np
from tqdm import tqdm
import urllib.request
import tarfile
import io
import jiwer
import re
import matplotlib.pyplot as plt
from scipy.io import wavfile
import matplotlib.font_manager as fm
import unicodedata
import string
import time
from dtw import dtw
from scipy.ndimage import median_filter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from whisper.model import disable_sdpa
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation

# 定义项目根目录和数据目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"Base directory: {BASE_DIR}")
DATA_DIR = os.path.join(BASE_DIR, "../datasets")
FLEURS_DIR = os.path.join(DATA_DIR, "Fleurs")
LIBRISPEECH_DIR = os.path.join(DATA_DIR, "LibriSpeech")
# AUDIO_DIR = os.path.join(BASE_DIR, "../whisper/data/audio")

# 确保数据目录存在
for dir_path in [FLEURS_DIR, LIBRISPEECH_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 配置日志
log_file = os.path.join(BASE_DIR, "tests", "whisper_test_log.txt")
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# 定义要测试的模型列表
TEST_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "turbo"]

# Fleurs 到 Whisper 语言代码映射
FLEURS_TO_WHISPER_LANG = {
    "cmn_hans_cn": "Chinese",
    "en_us": "English",
    # ... 其他语言映射 ...
}


def preprocess_text(text, case_sensitive=False, keep_punctuation=False):
    if not case_sensitive:
        text = text.lower()
    if not keep_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    return text


def calculate_bleu(
    references, hypotheses, case_sensitive=False, keep_punctuation=False
):
    transform = []
    if not case_sensitive:
        transform.append(ToLowerCase())
    if not keep_punctuation:
        transform.append(RemovePunctuation())
    transform = Compose(transform) if transform else lambda x: x

    processed_refs = [
        [transform(ref).split()]
        for ref in references
    ]
    processed_hyps = [
        transform(hyp).split()
        for hyp in hypotheses
    ]
    return corpus_bleu(
        processed_refs, processed_hyps, smoothing_function=SmoothingFunction().method1
    )


def download(url: str, target_path: str):
    """下载文件并显示进度条"""
    try:
        with urllib.request.urlopen(url) as source, open(target_path, "wb") as output:
            total_size = int(source.info().get("Content-Length", 0))
            with tqdm(
                total=total_size,
                unit="iB",
                unit_scale=True,
                desc=os.path.basename(target_path),
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break
                    output.write(buffer)
                    loop.update(len(buffer))
    except Exception as e:
        logger.error(f"下载失败: {url} -> {target_path}， 错误: {e}")
        raise


def download_font(language):
    if language in {"Chinese", "Japanese", "Korean"}:
        font = "GoNotoCJKCore.ttf"
    else:
        font = "GoNotoCurrent.ttf"

    font_release = "https://github.com/satbyy/go-noto-universal/releases/download/v5.2"
    font_path = os.path.join(DATA_DIR, font)
    if not os.path.exists(font_path):
        download(f"{font_release}/{font}", font_path)

    return font_path


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, subsample_rate=1):
        self.subsample_rate = subsample_rate

    def __len__(self):
        return len(self.data[:: self.subsample_rate])

    def __getitem__(self, idx):
        return self.data[idx * self.subsample_rate]


class LibriSpeechDataset(BaseDataset):
    def __init__(self, split="test-clean", subsample_rate=1):
        super().__init__(subsample_rate)
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=LIBRISPEECH_DIR,
            url=split,
            download=True,
        )
        self.data = [
            (waveform.squeeze().numpy(), text)
            for waveform, _, text, _, _, _ in self.dataset
        ]


class FleursDataset(BaseDataset):
    def __init__(self, lang, split="test", subsample_rate=1):
        super().__init__(subsample_rate)
        self.lang = lang
        url = f"https://storage.googleapis.com/xtreme_translations/FLEURS102/{lang}.tar.gz"
        tar_path = os.path.join(FLEURS_DIR, f"{lang}.tgz")
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)

        if not os.path.exists(tar_path):
            logger.info(f"开始下载 Fleurs 数据集: {lang}")
            download(url, tar_path)
            logger.info(f"Fleurs 数据集下载完成: {tar_path}")

        all_audio = {}
        labels = None
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in tar.getmembers():
                name = member.name
                if name.endswith(f"{split}.tsv"):
                    labels = pd.read_table(
                        tar.extractfile(member),
                        names=(
                            "id",
                            "file_name",
                            "raw_transcription",
                            "transcription",
                            "_",
                            "num_samples",
                            "gender",
                        ),
                    )
                if f"/{split}/" in name and name.endswith(".wav"):
                    audio_bytes = tar.extractfile(member).read()
                    all_audio[os.path.basename(name)] = audio_bytes

        assert (
            labels is not None
        ), f"No labels found for split '{split}' in lang '{lang}'"
        self.labels = labels.to_dict("records")[::subsample_rate]
        self.all_audio = all_audio
        self.data = [
            (self.load_audio(record["file_name"]), record["transcription"])
            for record in self.labels
        ]

    def load_audio(self, file_name):
        audio_bytes = self.all_audio[file_name]
        audio, sr = torchaudio.load(io.BytesIO(audio_bytes))
        return audio.squeeze().numpy()


class CustomDataset(BaseDataset):
    def __init__(self, data_dir, subsample_rate=1):
        super().__init__(subsample_rate)
        self.audio_files = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.endswith(".wav") or f.endswith(".mp3")
            ]
        )
        self.transcripts = [
            "What is your name?",
            "what are your hobbies?",
            "Do you like beijing",
            "How are you feeling today?",
            "what is the weather like today?",
        ]
        assert len(self.audio_files) == len(
            self.transcripts
        ), "音频文件数量与转录文本数量不匹配"
        self.data = [
            (self.load_audio(os.path.join(data_dir, audio_file)), transcript)
            for audio_file, transcript in zip(self.audio_files, self.transcripts)
        ]

    def load_audio(self, file_path):
        return whisper.load_audio(file_path)


def device():
    """确定使用 GPU 还是 CPU"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {dev}")
    return dev


def load_model(model_name, device):
    """加载 Whisper 模型"""
    logger.info(f"开始加载 Whisper 模型: {model_name}...")
    try:
        mdl = whisper.load_model(model_name, device=device)
        logger.info(f"Whisper 模型 {model_name} 加载成功")
        return mdl
    except Exception as e:
        logger.error(f"加载 Whisper 模型 {model_name} 失败: {e}")
        raise


# 保留原有的时间戳相关函数
def split_tokens_on_unicode(tokens: torch.Tensor, tokenizer):
    words = []
    word_tokens = []
    current_tokens = []

    for token in tokens.tolist():
        current_tokens.append(token)
        decoded = tokenizer.decode_with_timestamps(current_tokens)
        if "\ufffd" not in decoded:
            words.append(decoded)
            word_tokens.append(current_tokens)
            current_tokens = []

    return words, word_tokens


def split_tokens_on_spaces(tokens: torch.Tensor, tokenizer, merge_punctuation=True):
    subwords, subword_tokens_list = split_tokens_on_unicode(tokens, tokenizer)
    words = []
    word_tokens = []

    for subword, subword_tokens in zip(subwords, subword_tokens_list):
        special = subword_tokens[0] >= tokenizer.eot
        with_space = subword.startswith(" ")
        punctuation = subword.strip() in string.punctuation
        if special or with_space:
            words.append(subword)
            word_tokens.append(subword_tokens)
        elif punctuation and len(words) > 0 and merge_punctuation:
            words[-1] = words[-1] + subword
            word_tokens[-1].extend(subword_tokens)
        else:
            if len(words) > 0 and merge_punctuation:
                words[-1] = words[-1] + subword
                word_tokens[-1].extend(subword_tokens)
            else:
                words.append(subword)
                word_tokens.append(subword_tokens)

    return words, word_tokens


def get_split_tokens_function(language, merge_punctuation=True):
    if language in {"Chinese", "Japanese", "Thai", "Lao", "Myanmar"}:
        return split_tokens_on_unicode
    else:
        return lambda tokens, tokenizer: split_tokens_on_spaces(
            tokens, tokenizer, merge_punctuation
        )


def extract_word_timestamps(model, mel, tokens, duration, tokenizer):
    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE
    medfilt_width = 7
    qk_scale = 1.0

    # 初始化QKs列表
    QKs = [None] * model.dims.n_text_layer

    # 创建一个闭包函数来替代lambda
    def create_hook(index):
        # 定义Hook函数
        def hook_func(module, input, output):
            # print(f"index: {index}")
            # print(f"input[0] shape: {input[0].shape} input[1] shape: {input[1].shape}")
            # print(f"output[0] shape: {output[0].shape} output[1] shape: {output[1].shape}")
            QKs[index] = (
                output[-1].detach() if output[-1] is not None else torch.zeros(1)
            )

        return hook_func

    # 注册Hook，并确保每个Hook都正确捕获当前层的索引
    hooks = []
    for i, block in enumerate(model.decoder.blocks):
        hook = block.cross_attn.register_forward_hook(create_hook(i))
        hooks.append(hook)

    # 禁用SDPA
    with disable_sdpa():
        # 进行前向传播
        with torch.no_grad():
            logits = model(mel.unsqueeze(0), tokens.unsqueeze(0))

    # 解除注册的Hook
    for hook in hooks:
        hook.remove()

    # 检查QKs中每个元素的状态
    for idx, qk in enumerate(QKs):
        if qk is not None:
            # logger.info(f"QKs[{idx}] shape: {qk.shape}")
            continue
        else:
            logger.warning(f"QKs[{idx}] is None")

    # 过滤掉None值
    valid_QKs = [qk for qk in QKs if qk is not None]
    if not valid_QKs:
        raise ValueError("所有的QKs都是None，无法拼接。")

    # 拼接QKs
    weights = torch.cat(valid_QKs)  # layers * heads * tokens * frames
    # print(f"weights shape: {weights.shape}")

    # 确保weights具有预期的维度
    if weights.dim() != 4:
        raise ValueError(f"权重维度为{weights.dim()}，预期为4维。")

    # 截取权重
    weights = weights[:, :, :, : duration // AUDIO_SAMPLES_PER_TOKEN].cpu()

    # 后续处理
    weights = median_filter(weights, (1, 1, 1, medfilt_width))
    weights = torch.tensor(weights * qk_scale).softmax(dim=-1)

    w = weights / weights.norm(dim=-2, keepdim=True)
    matrix = w[-6:].mean(axis=(0, 1))

    alignment = dtw(-matrix.double().numpy())

    jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignment.index2s[jumps] * AUDIO_TIME_PER_TOKEN

    split_tokens_func = get_split_tokens_function(
        tokenizer.language, merge_punctuation=True
    )
    words, word_tokens = split_tokens_func(tokens, tokenizer)

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0))
    begin_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    word_timestamps = [
        (word, begin, end)
        for word, begin, end in zip(words[:-1], begin_times, end_times)
        # if not word.startswith("<|") and word.strip() not in ".,!?、。"
        if not word.startswith("<|")
    ]

    return word_timestamps, matrix, alignment


def print_timestamps(word_timestamps):
    logger.info("Word-level timestamps:")
    for word, start, end in word_timestamps[:10]:  # 只打印前10个作为示例
        logger.info(f"Word: {word}, Start: {start:.2f}s, End: {end:.2f}s")


def plot_timestamps(word_timestamps, model_name, task, sample_name, language):
    font_path = download_font(language)
    chinese_font = fm.FontProperties(fname=font_path)

    plt.figure(figsize=(12, 6))
    words = [w[0] for w in word_timestamps[:20]]
    starts = [w[1] for w in word_timestamps[:20]]
    ends = [w[2] for w in word_timestamps[:20]]

    plt.barh(
        range(len(words)),
        [end - start for start, end in zip(starts, ends)],
        left=starts,
        height=0.5,
    )
    plt.yticks(range(len(words)), words, fontproperties=chinese_font)
    plt.xlabel("Time (seconds)")
    plt.title(
        f"Word Timestamps for {model_name} - {task} - {sample_name}",
        fontproperties=chinese_font,
    )
    plt.tight_layout()

    plot_path = os.path.join(
        BASE_DIR, "tests", f"word_timestamps_{model_name}_{task}_{sample_name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Word timestamps plot saved to {plot_path}")


def plot_attention_matrix(
    matrix, alignment, word_timestamps, model_name, task, sample_name, language
):
    font_path = download_font(language)
    chinese_font = fm.FontProperties(fname=font_path)

    AUDIO_SAMPLES_PER_TOKEN = whisper.audio.HOP_LENGTH * 2
    AUDIO_TIME_PER_TOKEN = AUDIO_SAMPLES_PER_TOKEN / whisper.audio.SAMPLE_RATE

    plt.figure(figsize=(12, 8))
    plt.imshow(matrix, aspect="auto")
    plt.plot(alignment.index2s, alignment.index1s, color="red")

    xticks = np.arange(0, matrix.shape[1], 1 / AUDIO_TIME_PER_TOKEN)
    xticklabels = (xticks * AUDIO_TIME_PER_TOKEN).round().astype(np.int32)
    plt.xticks(xticks[::5], xticklabels[::5])
    plt.xlabel("Time (s)")

    words = [word for word, _, _ in word_timestamps]
    plt.yticks(range(len(words)), words, fontproperties=chinese_font)
    plt.ylabel("Words")

    plt.title(
        f"Attention Matrix - {model_name} - {task} - {sample_name}",
        fontproperties=chinese_font,
    )
    plt.tight_layout()

    plot_path = os.path.join(
        BASE_DIR, "tests", f"attention_matrix_{model_name}_{task}_{sample_name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Attention matrix plot saved to {plot_path}")


def process_timestamps(
    result,
    model,
    model_name,
    task,
    sample_name,
    timestamp_level,
    tokenizer,
    language,
    audio,
):
    if timestamp_level == "word":
        # 方法1: 使用 transcribe 方法自带的时间戳功能
        transcribe_timestamps = []
        for segment in result["segments"]:
            for word in segment["words"]:
                transcribe_timestamps.append((word["word"], word["start"], word["end"]))

        # 方法2: 使用自定义的时间戳计算方法
        mel = whisper.log_mel_spectrogram(
            audio=whisper.pad_or_trim(audio), n_mels=model.dims.n_mels
        ).to(device())
        tokens = torch.tensor(tokenizer.encode(result["text"])).to(device())
        duration = len(audio)

        custom_word_timestamps, attention_matrix, alignment = extract_word_timestamps(
            model, mel, tokens, duration, tokenizer
        )

        # 打印两种方法的结果进行对比
        logger.info("Comparing word-level timestamps:")
        logger.info("Transcribe method timestamps:")
        print_timestamps(transcribe_timestamps[:10])  # 只打印前10个作为示例

        logger.info("Custom method timestamps:")
        print_timestamps(custom_word_timestamps[:10])  # 只打印前10个作为示例

        # 计算两种方法的时间戳差异
        logger.info("Comparing timestamps difference...")
        compare_timestamps(transcribe_timestamps, custom_word_timestamps)

        # 绘制时间戳对比图
        plot_timestamp_comparison(
            transcribe_timestamps,
            custom_word_timestamps,
            model_name,
            task,
            sample_name,
            language,
        )

        # 绘制原有的图表
        plot_timestamps(custom_word_timestamps, model_name, task, sample_name, language)
        plot_attention_matrix(
            attention_matrix,
            alignment,
            custom_word_timestamps,
            model_name,
            task,
            sample_name,
            language,
        )


def compare_timestamps(transcribe_timestamps, custom_timestamps):
    trans_index = 0
    custom_index = 0

    while trans_index < len(transcribe_timestamps) and custom_index < len(
        custom_timestamps
    ):
        trans_word = transcribe_timestamps[trans_index][0].strip().lower()
        custom_word = custom_timestamps[custom_index][0].strip().lower()

        if trans_word == custom_word:
            # Compare timestamps
            trans_start, trans_end = (
                float(transcribe_timestamps[trans_index][1]),
                float(transcribe_timestamps[trans_index][2]),
            )
            custom_start, custom_end = (
                custom_timestamps[custom_index][1],
                custom_timestamps[custom_index][2],
            )

            logger.info(
                f"Word: {trans_word}, Transcribe: {trans_start:.3f}-{trans_end:.3f}, Custom: {custom_start:.3f}-{custom_end:.3f}"
            )

            trans_index += 1
            custom_index += 1
        elif len(trans_word) > len(custom_word):
            # Custom method might have split a word
            logger.info(f"Mismatch: Transcribe '{trans_word}', Custom '{custom_word}'")
            custom_index += 1
        else:
            # Transcribe method might have split a word
            logger.info(f"Mismatch: Transcribe '{trans_word}', Custom '{custom_word}'")
            trans_index += 1

    if trans_index < len(transcribe_timestamps):
        logger.info(
            f"Remaining transcribe words: {transcribe_timestamps[trans_index:]}"
        )
    if custom_index < len(custom_timestamps):
        logger.info(f"Remaining custom words: {custom_timestamps[custom_index:]}")


def plot_timestamp_comparison(
    transcribe_timestamps, custom_timestamps, model_name, task, sample_name, language
):
    font_path = download_font(language)
    chinese_font = fm.FontProperties(fname=font_path)

    plt.figure(figsize=(15, 8))

    common_words = []
    trans_index = 0
    custom_index = 0

    while (
        trans_index < len(transcribe_timestamps)
        and custom_index < len(custom_timestamps)
        and len(common_words) < 20
    ):
        if (
            transcribe_timestamps[trans_index][0].strip().lower()
            == custom_timestamps[custom_index][0].strip().lower()
        ):
            common_words.append(transcribe_timestamps[trans_index][0])

            trans_start, trans_end = (
                float(transcribe_timestamps[trans_index][1]),
                float(transcribe_timestamps[trans_index][2]),
            )
            custom_start, custom_end = (
                custom_timestamps[custom_index][1],
                custom_timestamps[custom_index][2],
            )

            plt.plot(
                [trans_start, trans_end],
                [len(common_words) - 1, len(common_words) - 1],
                "b-",
                linewidth=2,
                label="Transcribe" if len(common_words) == 1 else "",
            )
            plt.plot(
                [custom_start, custom_end],
                [len(common_words) - 1, len(common_words) - 1],
                "r-",
                linewidth=2,
                label="Custom" if len(common_words) == 1 else "",
            )

            trans_index += 1
            custom_index += 1
        elif len(transcribe_timestamps[trans_index][0]) > len(
            custom_timestamps[custom_index][0]
        ):
            custom_index += 1
        else:
            trans_index += 1

    plt.yticks(range(len(common_words)), common_words, fontproperties=chinese_font)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Words")
    plt.title(
        f"Timestamp Comparison - {model_name} - {task} - {sample_name}",
        fontproperties=chinese_font,
    )
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(
        BASE_DIR, "tests", f"timestamp_comparison_{model_name}_{task}_{sample_name}.png"
    )
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Timestamp comparison plot saved to {plot_path}")


def evaluate_model(
    model,
    dataset,
    transcribe_options,
    task,
    model_name,
    timestamp_level,
    tokenizer,
    language,
    case_sensitive=False,
    keep_punctuation=False,
):
    hypotheses, references = [], []
    start_time = time.time()
    total_samples = len(dataset)
    processed_samples = 0
    
    for i, (audio, text) in enumerate(
        tqdm(dataset, desc=f"{type(dataset).__name__} {task.capitalize()}")
    ):
        try:
            result = model.transcribe(audio, **transcribe_options)

            hypothesis = result["text"]
            reference = text

            hypotheses.append(hypothesis)
            references.append(reference)
            processed_samples += 1

            if i < 5:  # 只处理前5个样本的时间戳
                logger.info(f"Sample {i+1} - Reference: {reference}")
                logger.info(f"Sample {i+1} - Hypothesis: {hypothesis}")
                process_timestamps(
                    result,
                    model,
                    model_name,
                    task,
                    f"{type(dataset).__name__}_sample_{i+1}",
                    timestamp_level,
                    tokenizer,
                    language,
                    audio,
                )
        except Exception as e:
            logger.error(f"Sample {i+1} processing failed: {e}")
            continue

    end_time = time.time()
    total_time = end_time - start_time

    # 手动进行预处理
    transform = []
    if not case_sensitive:
        transform.append(ToLowerCase())
    if not keep_punctuation:
        transform.append(RemovePunctuation())
    transform = Compose(transform) if transform else lambda x: x

    transformed_references = [transform(ref) for ref in references]
    transformed_hypotheses = [transform(hyp) for hyp in hypotheses]

    # 计算指标
    wer_score = wer(transformed_references, transformed_hypotheses)
    cer_score = cer(transformed_references, transformed_hypotheses)
    bleu = calculate_bleu(references, hypotheses, case_sensitive, keep_punctuation)

    logger.info(f"{type(dataset).__name__} {task.capitalize()} Results:")
    logger.info(f"WER: {wer_score * 100:.2f}%")
    logger.info(f"CER: {cer_score * 100:.2f}%")
    logger.info(f"BLEU: {bleu:.4f}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Processed samples: {processed_samples}")
    logger.info(f"Processing speed: {processed_samples / total_time:.2f} samples/second")
    logger.info(f"Average reference length: {sum(len(r.split()) for r in references) / len(references):.2f} words")
    logger.info(f"Average hypothesis length: {sum(len(h.split()) for h in hypotheses) / len(hypotheses):.2f} words")
    logger.info("===============================================================")

    return wer_score, cer_score, bleu


def run_tests(
    models,
    datasets,
    tasks,
    timestamp_levels,
    case_sensitive=False,
    keep_punctuation=False,
):
    results = {}
    dev = device()

    for model_name in models:
        model = load_model(model_name, dev)
        tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual)

        for dataset_name, dataset in datasets.items():
            language = (
                FLEURS_TO_WHISPER_LANG.get(dataset.lang, "en")
                if hasattr(dataset, "lang")
                else "en"
            )

            for task in tasks:
                for timestamp_level in timestamp_levels:
                    logger.info(
                        f"Testing {model_name} on {dataset_name} - Task: {task}, Timestamp Level: {timestamp_level}"
                    )

                    transcribe_options = {
                        "task": task,
                        "language": language,
                        "word_timestamps": timestamp_level == "word",
                    }

                    wer, cer, bleu = evaluate_model(
                        model,
                        dataset,
                        transcribe_options,
                        task,
                        model_name,
                        timestamp_level,
                        tokenizer,
                        language,
                        case_sensitive=case_sensitive,
                        keep_punctuation=keep_punctuation,
                    )

                    results[f"{model_name}_{dataset_name}_{task}_{timestamp_level}"] = {
                        "WER": wer,
                        "CER": cer,
                        "BLEU": bleu,
                    }

    return results


def summarize_results(results):
    logger.info("=============== WHISPER TEST SUMMARY ===============")
    total_wer, total_cer, total_bleu = 0, 0, 0
    total_tests = len(results)
    
    for test_name, metrics in results.items():
        logger.info(f"\nResults for {test_name}:")
        wer = metrics['WER']
        cer = metrics['CER']
        bleu = metrics['BLEU']
        total_wer += wer
        total_cer += cer
        total_bleu += bleu
        
        logger.info(f"WER: {wer * 100:.2f}%")
        logger.info(f"CER: {cer * 100:.2f}%")
        logger.info(f"BLEU: {bleu:.4f}")
    
    logger.info("\nOverall Average Results:")
    logger.info(f"Average WER: {(total_wer / total_tests) * 100:.2f}%")
    logger.info(f"Average CER: {(total_cer / total_tests) * 100:.2f}%")
    logger.info(f"Average BLEU: {total_bleu / total_tests:.4f}")
    logger.info("====================================================")


if __name__ == "__main__":
    datasets = {
        "LibriSpeech": LibriSpeechDataset(split="test-clean", subsample_rate=50),
        "Fleurs": FleursDataset(lang="en_us", split="test", subsample_rate=10),
        "Custom": CustomDataset("./data/samples"),
    }
    whisper_test_configs = {
        "models": ["turbo"],
        "tasks": ["transcribe"],  # ["transcribe", "translate"]
        "timestamp_levels": ["word"],  # ["segment", "word"]
    }
    whisper_results = run_tests(
        models=whisper_test_configs["models"],
        datasets=datasets,
        tasks=whisper_test_configs["tasks"],
        timestamp_levels=whisper_test_configs["timestamp_levels"],
        case_sensitive=False,
        keep_punctuation=False,
    )
    
    # 添加总结
    summarize_results(whisper_results)