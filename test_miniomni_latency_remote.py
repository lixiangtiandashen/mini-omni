import argparse
import asyncio
import time
import random

import numpy as np
import websockets
import soundfile as sf
import base64
import json
import requests


async def run_benchmark(uri, audio_file):
    # 读取完整的音频文件二进制内容
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # 准备请求数据
    data = {
        "audio": audio_base64,
        "stream_stride": 4,  # 每次生成4个token
        "max_tokens": 2048
    }

    start_time = time.time()

    # 发送POST请求
    response = requests.post(uri, json=data, stream=True)

    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        return

    total_audio_chunks = 0
    total_tokens = 0
    token_times = []
    last_token_time = start_time
    first_token_time = None

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            total_audio_chunks += 1
            current_time = time.time()

            # 每个音频chunk对应stream_stride个token
            total_tokens += data["stream_stride"]
            
            if first_token_time is None:
                first_token_time = current_time
                last_token_time = current_time
            else:
                token_time = current_time - last_token_time
                token_times.append(token_time)
                last_token_time = current_time

    end_time = time.time()
    total_time = end_time - start_time
    generation_time = end_time - first_token_time

    print(f"总运行时间: {total_time:.2f} 秒")
    print(f"首token延迟: {first_token_time - start_time:.2f} 秒")
    print(f"总音频块数: {total_audio_chunks}")
    print(f"总生成token数: {total_tokens}")

    if token_times:
        avg_token_time = sum(token_times) / len(token_times)
        print(f"中间每token延迟: {avg_token_time:.4f} 秒")
        print(f"最小token延迟: {min(token_times):.4f} 秒")
        print(f"最大token延迟: {max(token_times):.4f} 秒")
        print(f"吞吐量: {(total_tokens - data['stream_stride']) / generation_time:.2f} tokens/秒")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        type=str,
        default="http://localhost:60808/chat",
        help="URI of the Miniomni server",
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="./data/samples/output1.wav",
        help="Path to the test audio file",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.uri, args.audio_file))


if __name__ == "__main__":
    main()
