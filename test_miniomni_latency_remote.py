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


async def run_benchmark(uri, steps, audio_file):
    # 读取完整的音频文件二进制内容
    with open(audio_file, "rb") as f:
        audio_data = f.read()
    audio_base64 = base64.b64encode(audio_data).decode("utf-8")

    # 准备请求数据
    data = {
        "audio": audio_base64,
        "stream_stride": 4,
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

    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            total_audio_chunks += 1
            current_time = time.time()

            # 假设每个音频chunk对应一个token
            total_tokens += 1
            token_time = current_time - last_token_time
            token_times.append(token_time)
            last_token_time = current_time

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total audio chunks received: {total_audio_chunks}")
    print(f"Total tokens generated: {total_tokens}")

    if token_times:
        avg_token_time = sum(token_times) / len(token_times)
        print(f"Average token generation time: {avg_token_time:.4f} seconds")
        print(f"Min token generation time: {min(token_times):.4f} seconds")
        print(f"Max token generation time: {max(token_times):.4f} seconds")
        print(f"Tokens per second: {total_tokens / total_time:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        type=str,
        default="http://localhost:60808/chat",
        help="URI of the Miniomni server",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of steps to run the benchmark"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        default="./data/samples/output1.wav",
        help="Path to the test audio file",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.uri, args.steps, args.audio_file))


if __name__ == "__main__":
    main()
