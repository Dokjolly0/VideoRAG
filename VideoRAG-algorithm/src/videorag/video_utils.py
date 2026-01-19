import logging
import os
import shutil
import time

import numpy as np
import torch
from faster_whisper import WhisperModel
from imagebind import data
from imagebind.models.imagebind_model import ImageBindModel, ModalityType
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from tqdm import tqdm

from ..utils.file_locator import FileLocator
from .llm import load_model_with_fast_fallback, load_tokenizer_with_fast_fallback
from .utils import logger


class VideoUtils:
    def __init__(self):
        self.fl = FileLocator()

    def split_video(
        self,
        video_path,
        working_dir,
        segment_length,
        num_frames_per_segment,
        audio_output_format="mp3",
    ):
        unique_timestamp = str(int(time.time() * 1000))
        video_name = os.path.basename(video_path).split(".")[0]
        video_segment_cache_path = os.path.join(working_dir, "_cache", video_name)
        if os.path.exists(video_segment_cache_path):
            shutil.rmtree(video_segment_cache_path)
        os.makedirs(video_segment_cache_path, exist_ok=False)

        segment_index = 0
        segment_index2name, segment_times_info = {}, {}
        with VideoFileClip(video_path) as video:
            total_video_length = int(video.duration)
            start_times = list(range(0, total_video_length, segment_length))
            # if the last segment is shorter than 5 seconds, we merged it to the last segment
            if len(start_times) > 1 and (total_video_length - start_times[-1]) < 5:
                start_times = start_times[:-1]

            for start in tqdm(start_times, desc=f"Spliting Video {video_name}"):
                if start != start_times[-1]:
                    end = min(start + segment_length, total_video_length)
                else:
                    end = total_video_length

                subvideo = video.subclip(start, end)
                subvideo_length = subvideo.duration
                frame_times = np.linspace(
                    0, subvideo_length, num_frames_per_segment, endpoint=False
                )
                frame_times += start

                segment_index2name[f"{segment_index}"] = (
                    f"{unique_timestamp}-{segment_index}-{start}-{end}"
                )
                segment_times_info[f"{segment_index}"] = {
                    "frame_times": frame_times,
                    "timestamp": (start, end),
                }

                # save audio
                audio_file_base_name = segment_index2name[f"{segment_index}"]
                audio_file = f"{audio_file_base_name}.{audio_output_format}"
                try:
                    subaudio = subvideo.audio
                    subaudio.write_audiofile(
                        os.path.join(video_segment_cache_path, audio_file),
                        codec="mp3",
                        verbose=False,
                        logger=None,
                    )
                except Exception as e:
                    logger.warning(
                        f"Warning: Failed to extract audio for video {video_name} ({start}-{end}). Probably due to lack of audio track. {e}"
                    )

                segment_index += 1

        return segment_index2name, segment_times_info

    def saving_video_segments(
        self,
        video_name,
        video_path,
        working_dir,
        segment_index2name,
        segment_times_info,
        error_queue,
        video_output_format="mp4",
    ):
        try:
            with VideoFileClip(video_path) as video:
                video_segment_cache_path = os.path.join(
                    working_dir, "_cache", video_name
                )
                for index in tqdm(
                    segment_index2name, desc=f"Saving Video Segments {video_name}"
                ):
                    start, end = (
                        segment_times_info[index]["timestamp"][0],
                        segment_times_info[index]["timestamp"][1],
                    )
                    video_file = f"{segment_index2name[index]}.{video_output_format}"
                    subvideo = video.subclip(start, end)
                    subvideo.write_videofile(
                        os.path.join(video_segment_cache_path, video_file),
                        codec="libx264",
                        verbose=False,
                        logger=None,
                    )
        except Exception as e:
            error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
            raise RuntimeError

    def encode_video_segments(self, video_paths, embedder: ImageBindModel):
        device = next(embedder.parameters()).device
        inputs = {
            ModalityType.VISION: data.load_and_transform_video_data(
                video_paths, device
            ),
        }
        with torch.no_grad():
            embeddings = embedder(inputs)[ModalityType.VISION]
        embeddings = embeddings.cpu()
        return embeddings

    def encode_string_query(self, query: str, embedder: ImageBindModel):
        device = next(embedder.parameters()).device
        inputs = {
            ModalityType.TEXT: data.load_and_transform_text([query], device),
        }
        with torch.no_grad():
            embeddings = embedder(inputs)[ModalityType.TEXT]
        embeddings = embeddings.cpu()
        return embeddings

    def encode_video(self, video, frame_times):
        frames = []
        for t in frame_times:
            frames.append(video.get_frame(t))
        frames = np.stack(frames, axis=0)
        frames = [
            Image.fromarray(v.astype("uint8")).resize((1280, 720)) for v in frames
        ]
        return frames

    def segment_caption(
        self,
        video_name,
        video_path,
        segment_index2name,
        transcripts,
        segment_times_info,
        caption_result,
        error_queue,
    ):
        try:
            model_path = self.fl.MiniCPM
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")

            abs_model_path = os.path.abspath(model_path)
            model = load_model_with_fast_fallback(abs_model_path)
            tokenizer = load_tokenizer_with_fast_fallback(abs_model_path)
            model.eval()

            with VideoFileClip(video_path) as video:
                for index in tqdm(
                    segment_index2name, desc=f"Captioning Video {video_name}"
                ):
                    frame_times = segment_times_info[index]["frame_times"]
                    video_frames = self.encode_video(video, frame_times)
                    segment_transcript = transcripts[index]
                    query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                    msgs = [{"role": "user", "content": video_frames + [query]}]
                    params = {}
                    params["use_image_id"] = False
                    params["max_slice_nums"] = 2
                    segment_caption = model.chat(
                        image=None, msgs=msgs, tokenizer=tokenizer, **params
                    )
                    caption_result[index] = segment_caption.replace("\n", "").replace(
                        "<|endoftext|>", ""
                    )
                    torch.cuda.empty_cache()
        except Exception as e:
            error_queue.put(f"Error in segment_caption:\n {str(e)}")
            raise RuntimeError

    def merge_segment_information(
        self, segment_index2name, segment_times_info, transcripts, captions
    ):
        inserting_segments = {}
        for index in segment_index2name:
            inserting_segments[index] = {"content": None, "time": None}
            segment_name = segment_index2name[index]
            inserting_segments[index]["time"] = "-".join(segment_name.split("-")[-2:])
            inserting_segments[index]["content"] = (
                f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
            )
            inserting_segments[index]["transcript"] = transcripts[index]
            inserting_segments[index]["frame_times"] = segment_times_info[index][
                "frame_times"
            ].tolist()
        return inserting_segments

    def retrieved_segment_caption(
        self,
        caption_model,
        caption_tokenizer,
        refine_knowledge,
        retrieved_segments,
        video_path_db,
        video_segments,
        num_sampled_frames,
    ):
        # model = AutoModel.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained('./MiniCPM-V-2_6-int4', trust_remote_code=True, use_fast=True)
        # model.eval()

        caption_result = {}
        for this_segment in tqdm(
            retrieved_segments, desc="Captioning Segments for Given Query"
        ):
            video_name = "_".join(this_segment.split("_")[:-1])
            index = this_segment.split("_")[-1]
            video_path = video_path_db._data[video_name]
            timestamp = video_segments._data[video_name][index]["time"].split("-")
            start, end = eval(timestamp[0]), eval(timestamp[1])
            video = VideoFileClip(video_path)
            frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
            video_frames = self.encode_video(video, frame_times)
            segment_transcript = video_segments._data[video_name][index]["transcript"]
            # query = f"The transcript of the current video:\n{segment_transcript}.\nGiven a question: {query}, you have to extract relevant information from the video and transcript for answering the question."
            query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}'"
            msgs = [{"role": "user", "content": video_frames + [query]}]
            params = {}
            params["use_image_id"] = False
            params["max_slice_nums"] = 2
            segment_caption = caption_model.chat(
                image=None, msgs=msgs, tokenizer=caption_tokenizer, **params
            )
            this_caption = segment_caption.replace("\n", "").replace(
                "<|endoftext|>", ""
            )
            caption_result[this_segment] = (
                f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
            )
            torch.cuda.empty_cache()

        return caption_result

    def speech_to_text(
        self, video_name, working_dir, segment_index2name, audio_output_format
    ):
        model_path = self.fl.FasterDistil
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        model = WhisperModel(os.path.abspath(model_path))
        model.logger.setLevel(logging.WARNING)

        cache_path = os.path.join(working_dir, "_cache", video_name)

        transcripts = {}
        for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
            segment_name = segment_index2name[index]
            audio_file = os.path.join(
                cache_path, f"{segment_name}.{audio_output_format}"
            )

            # if the audio file does not exist, skip it
            if not os.path.exists(audio_file):
                transcripts[index] = ""
                continue

            segments, info = model.transcribe(audio_file)
            result = ""
            for segment in segments:
                result += "[%.2fs -> %.2fs] %s\n" % (
                    segment.start,
                    segment.end,
                    segment.text,
                )
            transcripts[index] = result

        return transcripts
