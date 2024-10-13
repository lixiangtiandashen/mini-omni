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
from jiwer import wer, cer, Compose, ToLowerCase, RemovePunctuation

# 定义项目根目录和数据目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../datasets")
FLEURS_DIR = os.path.join(DATA_DIR, "Fleurs")
LIBRISPEECH_DIR = os.path.join(DATA_DIR, "LibriSpeech")

# 确保数据目录存在
for dir_path in [FLEURS_DIR, LIBRISPEECH_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Fleurs 到 Whisper 语言代码映射
FLEURS_TO_WHISPER_LANG = {
    "cmn_hans_cn": "Chinese",
    "en_us": "English",
    # ... 其他语言映射 ...
}


def configure_logger(name, log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


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

    processed_refs = [[transform(ref).split()] for ref in references]
    processed_hyps = [transform(hyp).split() for hyp in hypotheses]
    return corpus_bleu(
        processed_refs, processed_hyps, smoothing_function=SmoothingFunction().method1
    )


def download(url: str, target_path: str):
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
        print(f"下载失败: {url} -> {target_path}， 错误: {e}")
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
            print(f"开始下载 Fleurs 数据集: {lang}")
            download(url, tar_path)
            print(f"Fleurs 数据集下载完成: {tar_path}")

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
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {dev}")
    return dev


def load_model(model_name, device):
    print(f"开始加载 Whisper 模型: {model_name}...")
    try:
        mdl = whisper.load_model(model_name, device=device)
        print(f"Whisper 模型 {model_name} 加载成功")
        return mdl
    except Exception as e:
        print(f"加载 Whisper 模型 {model_name} 失败: {e}")
        raise


def summarize_results(results, logger):
    logger.info("=============== TEST SUMMARY ===============")
    total_wer, total_cer, total_bleu = 0, 0, 0
    total_tests = len(results)

    for test_name, metrics in results.items():
        logger.info(f"\nResults for {test_name}:")
        wer = metrics["WER"]
        cer = metrics["CER"]
        bleu = metrics["BLEU"]
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
