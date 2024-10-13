from test_utils import *
from snac import SNAC
from litgpt import Tokenizer
from litgpt.model import GPT, Config
from lightning.fabric.utilities.load import _lazy_load as lazy_load
from utils.snac_utils import layershift, reconscruct_snac, reconstruct_tensors
from huggingface_hub import snapshot_download
import lightning as L

# 配置日志
logger = configure_logger(__name__, os.path.join(BASE_DIR, "tests", "mini_omni_test_log.txt"))

# 定义常量
_eot = 151936
_pad_t = 151937
_input_t = 151938
_answer_t = 151939
_asr = 151940

_eoa = 4096
_pad_a = 4097
_input_a = 4098
_answer_a = 4099
_split = 4100

def load_model(ckpt_dir, device):
    snacmodel = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to(device)
    whispermodel = whisper.load_model("small").to(device)
    text_tokenizer = Tokenizer(ckpt_dir)
    fabric = L.Fabric(devices=1, strategy="auto")
    config = Config.from_file(ckpt_dir + "/model_config.yaml")
    config.post_adapter = False

    with fabric.init_module(empty_init=False):
        model = GPT(config)

    model = fabric.setup(model)
    state_dict = lazy_load(ckpt_dir + "/lit_model.pth")
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    return fabric, model, text_tokenizer, snacmodel, whispermodel

def download_model(ckpt_dir):
    repo_id = "gpt-omni/mini-omni"
    snapshot_download(repo_id, local_dir=ckpt_dir, revision="main")

def preprocess_audio(audio, whispermodel, device):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)
    return mel, mel.shape[-1]

def get_input_ids_whisper(mel, leng, whispermodel, device, special_token_a=_answer_a, special_token_t=_answer_t):
    with torch.no_grad():
        mel = mel.unsqueeze(0).to(device)
        audio_feature = whispermodel.embed_audio(mel)[0][:leng]

    T = audio_feature.size(0)
    input_ids = []
    for i in range(7):
        input_ids_item = []
        input_ids_item.append(layershift(_input_a, i))
        input_ids_item += [layershift(_pad_a, i)] * T
        input_ids_item += [(layershift(_eoa, i)), layershift(special_token_a, i)]
        input_ids.append(torch.tensor(input_ids_item).unsqueeze(0))
    input_id_T = torch.tensor([_input_t] + [_pad_t] * T + [_eot, special_token_t])
    input_ids.append(input_id_T.unsqueeze(0))
    return audio_feature.unsqueeze(0), input_ids

def A1_T1(fabric, audio_feature, input_ids, leng, model, text_tokenizer, step):
    from inference import generate_ASR

    with fabric.init_tensor():
        model.set_kv_cache(batch_size=1)
    tokenlist = generate_ASR(
        model,
        audio_feature,
        input_ids,
        [leng],
        ["A1T1"],
        max_returned_tokens=2048,
        temperature=0.9,
        top_k=1,
        eos_id_a=_eoa,
        eos_id_t=_eot,
        pad_id_t=_pad_t,
        shift=151936 + 64,
        include_prompt=True,
        generate_text=True,
    )
    model.clear_kv_cache()
    return text_tokenizer.decode(torch.tensor(tokenlist)).strip()

def evaluate_omni_model(model, dataset, fabric, text_tokenizer, snacmodel, whispermodel, device, case_sensitive=False, keep_punctuation=False):
    hypotheses, references = [], []
    start_time = time.time()
    total_samples = len(dataset)
    processed_samples = 0

    for i, (audio, text) in enumerate(
        tqdm(dataset, desc=f"{type(dataset).__name__} Evaluation")
    ):
        try:
            mel, leng = preprocess_audio(audio, whispermodel, device)
            audio_feature, input_ids = get_input_ids_whisper(
                mel,
                leng,
                whispermodel,
                device,
                special_token_a=_pad_a,
                special_token_t=_asr,
            )

            output = A1_T1(
                fabric=fabric,
                audio_feature=audio_feature,
                input_ids=input_ids,
                leng=leng,
                model=model,
                text_tokenizer=text_tokenizer,
                step=i,
            )

            hypotheses.append(output)
            references.append(text)
            processed_samples += 1

            if i < 5:  # 只打印前5个样本
                logger.info(f"Sample {i+1}:")
                logger.info(f"Reference: {text}")
                logger.info(f"Hypothesis: {output}")
                logger.info("---")
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

    logger.info(f"{type(dataset).__name__} Results:")
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

def run_omni_tests(datasets, case_sensitive=False, keep_punctuation=False):
    results = {}

    ckpt_dir = "./checkpoint"
    if not os.path.exists(ckpt_dir):
        logger.info(
            f"Checkpoint directory {ckpt_dir} not found, downloading from huggingface"
        )
        download_model(ckpt_dir)

    fabric, model, text_tokenizer, snacmodel, whispermodel = load_model(
        ckpt_dir, device()
    )

    for dataset_name, dataset in datasets.items():
        logger.info(f"Testing Omni model on {dataset_name}")

        wer, cer, bleu = evaluate_omni_model(
            model,
            dataset,
            fabric,
            text_tokenizer,
            snacmodel,
            whispermodel,
            device(),
            case_sensitive=case_sensitive,
            keep_punctuation=keep_punctuation,
        )

        results[f"omni_{dataset_name}"] = {"WER": wer, "CER": cer, "BLEU": bleu}

    return results

if __name__ == "__main__":
    datasets = {
        "LibriSpeech": LibriSpeechDataset(split="test-clean", subsample_rate=50),
        "Fleurs": FleursDataset(lang="en_us", split="test", subsample_rate=10),
        "Custom": CustomDataset("./data/samples"),
    }

    omni_results = run_omni_tests(datasets, case_sensitive=False, keep_punctuation=False)
    
    # 添加总结
    summarize_results(omni_results, logger)