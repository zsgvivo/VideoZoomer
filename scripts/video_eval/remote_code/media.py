import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig

MEDIA_TOKENS = {
    "image": "<image>",
    "video": "<vila/video>",
}


class Media:
    pass


class File(Media):
    def __init__(self, path: str) -> None:
        self.path = path


class Image(File):
    pass


class Video(File):
    pass


def make_list(obj: Any) -> List:
    return obj if isinstance(obj, list) else [obj]


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            print(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]

def _load_video_with_fps(video_path: str, *, num_frames: int, fps: float) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, min(num_frames, len(frame_paths)))).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    orig_fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Estimate video duration in seconds
    duration_sec = frame_count / orig_fps if orig_fps > 0 else 0

    if duration_sec == 0:
        raise ValueError(f"Video '{video_path}' seems to be empty or corrupted.")

    # Compute total frames to sample based on desired fps
    sampled_frame_count = min(((int(duration_sec * fps) + 127) // 128) * 128, ((num_frames + 127) // 128) * 128)

    # Compute which frame indices to sample
    indices = np.linspace(0, frame_count - 1, sampled_frame_count).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            print(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)

    vidcap.release()
    return [frames[index] for index in indices if index in frames]


def _extract_video(video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    video_path = video.path if isinstance(video, Video) else video["path"]
    if getattr(config, "fps") > 0:
        frames = _load_video_with_fps(video_path, num_frames=num_frames, fps=config.fps)
    else:
        frames = _load_video(video_path, num_frames=num_frames)
    return frames


def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        print(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            elif isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            elif isinstance(part, dict) or isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            else:
                raise ValueError(f"Unsupported prompt part type: {type(part)}")
        message["value"] = text

    if MEDIA_TOKENS["video"] in messages[0]["value"]:
        messages[0]["value"] = "<vila/video>" + messages[0]["value"].replace("<vila/video>", "")
    return media
