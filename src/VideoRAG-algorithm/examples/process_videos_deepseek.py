import asyncio
import logging
import multiprocessing
import os
import warnings

from ..videorag import VideoRAG
from ..videorag.llm import deepseek_bge_config

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# 设置DeepSeek和硅基流动的API密钥
os.environ["DEEPSEEK_API_KEY"] = "sk-*******"
os.environ["SILICONFLOW_API_KEY"] = "sk-******"


async def main():
    # 必须的设置
    multiprocessing.set_start_method("spawn")

    # 将你的视频文件路径放入这个列表中
    video_paths = [
        "/home/zy/VideoRAG/fc6207139422fa5f030113b7d79efe5e.mp4",
    ]

    # 初始化 VideoRAG，指定一个工作目录来存放索引文件
    videorag = VideoRAG(llm=deepseek_bge_config, working_dir="./videorag-workdir")

    # 开始处理视频
    await videorag.insert_video(video_path_list=video_paths)


if __name__ == "__main__":
    asyncio.run(main())
