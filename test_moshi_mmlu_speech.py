import argparse
import asyncio
import time
import os
import json
from tqdm import tqdm
import websockets
import sphn
import numpy as np

# 配置日志
from test_utils import configure_logger, BASE_DIR

logger = configure_logger(__name__, os.path.join(BASE_DIR, "tests", "moshi_mmlu_speech_test_log.txt"))


def load_mmlu_audio_dataset(dataset_dir):
    samples = []
    for subject in os.listdir(dataset_dir):
        subject_dir = os.path.join(dataset_dir, subject)
        logger.info(f"加载数据集子主题: {subject_dir}")
        if os.path.isdir(subject_dir):
            for sample_dir in os.listdir(subject_dir):
                sample_path = os.path.join(subject_dir, sample_dir)
                if os.path.isdir(sample_path):
                    # 加载答案
                    answer_file = os.path.join(
                        sample_path, f"{subject}_{sample_dir.split('_')[-1]}_answer.txt"
                    )
                    if not os.path.exists(answer_file):
                        logger.info(f"答案文件不存在: {answer_file}")
                        continue
                    with open(answer_file, "r") as f:
                        answer = f.read().strip()

                    # 遍历所有音色目录
                    for voice_dir in os.listdir(sample_path):
                        voice_path = os.path.join(sample_path, voice_dir)
                        if os.path.isdir(voice_path):
                            audio_files = [
                                f for f in os.listdir(voice_path) if f.endswith(".wav")
                            ]
                            if not audio_files:
                                logger.info(f"没有音频文件: {voice_path}")
                                continue
                            audio_path = os.path.join(voice_path, audio_files[0])

                            samples.append(
                                {
                                    "subject": subject,
                                    "audio_path": audio_path,
                                    "answer": answer,
                                    "voice": voice_dir,
                                }
                            )
    return samples


def normalize_answer(answer):
    answer = answer.lower()
    if answer in ["a", "0"]:
        return "0"
    elif answer in ["b", "1"]:
        return "1"
    elif answer in ["c", "2"]:
        return "2"
    elif answer in ["d", "3"]:
        return "3"
    else:
        return answer


async def process_audio_question(uri, audio_path):
    async with websockets.connect(uri) as websocket:
        # 等待握手
        handshake = await websocket.recv()
        assert handshake == b"\x00", "Handshake failed"

        opus_writer = sphn.OpusStreamWriter(24000)
        opus_reader = sphn.OpusStreamReader(24000)

        main_text = []
        send_complete = asyncio.Event()

        # 读取音频文件
        audio_data, sample_rate = sphn.read(filename=audio_path, sample_rate=24000)
        # logger.info("audio_data.shape, sample_rate %s, %s", audio_data.shape, sample_rate)
        if audio_data.ndim > 1:
            audio_data = audio_data[0]  # 如果是多通道，只取第一个通道
        if sample_rate != 24000:
            audio_data = sphn.resample(audio_data, sample_rate, 24000)
            sample_rate = 24000

        assert sample_rate == 24000, f"Expected sample rate 24000, got {sample_rate}"

        async def send_audio():
            chunk_size = 24000 // 50  # 20ms的音频, 24000 / 50 = 480
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size] 
                # logger.info("chunk shape %s", chunk.shape)
                # 最后一个数据快的长度可能小于chunk_size，需要补全
                if chunk.shape[0] != chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - chunk.shape[0]))
                # pcm length has to match an allowed frame size [120, 240, 480, 960, 1920, 2880]
                opus_writer.append_pcm(chunk)
                opus_data = opus_writer.read_bytes()
                if opus_data:
                    await websocket.send(b"\x01" + opus_data)
                # await asyncio.sleep(0.02)  # 模拟实时音频流
            send_complete.set()  # 标记发送完成

        async def receive_response():
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    kind = response[0]
                    payload = response[1:]

                    if kind == 1:  # 音频
                        pass  # 我们不需要处理返回的音频
                    elif kind == 2:  # 文本
                        text = payload.decode("utf-8")
                        main_text.append(text)

                except asyncio.TimeoutError:
                    if send_complete.is_set():
                        break  # 如果发送已完成且接收超时,则退出
                except websockets.exceptions.ConnectionClosed:
                    break

        start_time = time.time()
        await asyncio.gather(send_audio(), receive_response())
        end_time = time.time()

        processing_time = end_time - start_time
        response = "".join(main_text).strip()

        return response, processing_time


async def test_mmlu_speech(uri, dataset_dir, output_dir):
    # 加载测试集
    samples = load_mmlu_audio_dataset(dataset_dir)

    # 按主题分组样本
    samples_by_subject = {}
    for sample in samples:
        subject = sample["subject"]
        if subject not in samples_by_subject:
            samples_by_subject[subject] = []
        samples_by_subject[subject].append(sample)

    results = {}
    log_file = os.path.join(output_dir, "mmlu_speech_results_log.jsonl")
    
    # 使用tqdm显示主题级别的进度
    for subject, subject_samples in tqdm(
        samples_by_subject.items(), desc="Processing subjects"
    ):
        logger.info(f"Processing subject: {subject}")
        results[subject] = {"correct": 0, "total": 0, "responses": {}}

        # 使用tqdm显示每个主题内的样本处理进度
        for sample in tqdm(
            subject_samples, desc=f"Processing {subject} samples", leave=False
        ):
            voice = sample["voice"]
            if voice not in results[subject]["responses"]:
                results[subject]["responses"][voice] = []

            # 处理音频问题
            response, processing_time = await process_audio_question(
                uri, sample["audio_path"]
            )

            # 规范化答案和响应
            normalized_response = normalize_answer(response)
            normalized_answer = normalize_answer(sample["answer"])

            # 检查答案是否正确
            is_correct = normalized_response == normalized_answer
            results[subject]["correct"] += int(is_correct)
            results[subject]["total"] += 1
            
            # 创建单个样本的结果字典
            sample_result = {
                "subject": subject,
                "voice": voice,
                "audio_path": sample["audio_path"],
                "response": response,
                "normalized_response": normalized_response,
                "correct_answer": sample["answer"],
                "normalized_answer": normalized_answer,
                "is_correct": is_correct,
                "processing_time": processing_time,
            }
            
            # 将结果添加到 results 字典
            results[subject]["responses"][voice].append(sample_result)
            
            # 立即将单个样本的结果写入日志文件
            with open(log_file, "a") as f:
                json.dump(sample_result, f)
                f.write("\n")

    # 计算并记录结果
    overall_correct = sum(
        subject_result["correct"] for subject_result in results.values()
    )
    overall_total = sum(subject_result["total"] for subject_result in results.values())
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0

    logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
    for subject, subject_result in results.items():
        subject_accuracy = (
            subject_result["correct"] / subject_result["total"]
            if subject_result["total"] > 0
            else 0
        )
        logger.info(f"{subject} Accuracy: {subject_accuracy:.2%}")

        # 计算每个音色的准确率
        for voice, voice_responses in subject_result["responses"].items():
            voice_correct = sum(int(r["is_correct"]) for r in voice_responses)
            voice_total = len(voice_responses)
            voice_accuracy = voice_correct / voice_total if voice_total > 0 else 0
            logger.info(f"  {voice} Accuracy: {voice_accuracy:.2%}")

    # 计算并记录平均处理时间
    total_time = sum(
        sum(r["processing_time"] for r in voice_responses)
        for subject_result in results.values()
        for voice_responses in subject_result["responses"].values()
    )
    total_questions = sum(
        sum(
            len(voice_responses)
            for voice_responses in subject_result["responses"].values()
        )
        for subject_result in results.values()
    )
    average_time = total_time / total_questions if total_questions > 0 else 0

    logger.info(f"Average processing time per question: {average_time:.4f} seconds")

    # 保存完整的结果
    with open(os.path.join(output_dir, "mmlu_speech_results_full.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        type=str,
        default="ws://localhost:8998/api/chat",
        help="WebSocket URI of the Moshi server",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="../datasets/mmlu_test_tts",
        help="Directory of the MMLU audio dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/moshi_mmlu_speech_test",
        help="Directory to save the test results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    asyncio.run(test_mmlu_speech(args.uri, args.dataset_dir, args.output_dir))


if __name__ == "__main__":
    main()
