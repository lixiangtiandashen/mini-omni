import argparse
import asyncio
import time
import random

import numpy as np
import websockets
import sphn


async def run_benchmark(uri, steps):
    async with websockets.connect(uri) as websocket:
        # 等待握手
        handshake = await websocket.recv()
        assert handshake == b"\x00", "Handshake failed"

        opus_writer = sphn.OpusStreamWriter(24000)  # 假设采样率为24000
        opus_reader = sphn.OpusStreamReader(24000)

        main_audio = []
        main_text = []
        token_times = []  # 新增:用于存储每个token的生成时间
        input_processing_time = 0  # 新增:用于记录输入处理时间

        send_complete = asyncio.Event()

        async def send_audio():
            for _ in range(steps):
                # 生成随机音频数据
                chunk = np.random.randn(24000 // 50).astype(np.float32)  # 20ms的音频
                opus_writer.append_pcm(chunk)
                opus_data = opus_writer.read_bytes()
                if opus_data:
                    await websocket.send(b"\x01" + opus_data)
                # await asyncio.sleep(0.02)  # 模拟实时音频流
            send_complete.set()  # 标记发送完成

        async def receive_response():
            nonlocal input_processing_time
            last_token_time = time.time()  # 新增:记录上一个token的时间
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    current_time = time.time()
                    kind = response[0]
                    payload = response[1:]

                    if kind == 1:  # 音频
                        opus_reader.append_bytes(payload)
                        pcm = opus_reader.read_pcm()
                        if pcm.size > 0:
                            main_audio.append(pcm)
                    elif kind == 2:  # 文本
                        text = payload.decode("utf-8")
                        main_text.append(text)
                        print(f"Received text: {text}")
                        
                        # 新增:计算并记录token生成时间
                        token_time = current_time - last_token_time
                        token_times.append(token_time)
                        last_token_time = current_time
                        
                        if len(main_text) == 1:
                            # 新增:记录第一个token的时间作为输入处理时间
                            input_processing_time = current_time - start_time

                except asyncio.TimeoutError:
                    if send_complete.is_set():
                        break  # 如果发送已完成且接收超时,则退出
                except websockets.exceptions.ConnectionClosed:
                    break

        start_time = time.time()
        await asyncio.gather(send_audio(), receive_response())
        end_time = time.time()

        total_time = end_time - start_time
        total_tokens = len(main_text)
        
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Input processing time: {input_processing_time:.2f} seconds")
        print(f"Total tokens generated: {total_tokens}")
        
        if token_times:
            # 计算token速率应当除去input_processing_time
            avg_token_time = sum(token_times) / len(token_times)
            print(f"Average token generation time: {avg_token_time:.4f} seconds")
            print(f"Min token generation time: {min(token_times):.4f} seconds")
            print(f"Max token generation time: {max(token_times):.4f} seconds")
            print(f"Tokens per second: {1 / avg_token_time:.2f}")
        
        print("Generated text:")
        print("".join(main_text))

        # 保存生成的音频
        if main_audio:
            main_audio_np = np.concatenate(main_audio)
            sphn.write_wav("gen_main_remote.wav", main_audio_np, 24000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uri",
        type=str,
        default="ws://localhost:8998/api/chat",
        help="WebSocket URI of the Moshi server",
    )
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of audio chunks to send"
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.uri, args.steps))


if __name__ == "__main__":
    main()
