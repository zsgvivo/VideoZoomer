from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import os
import time

def prepare_tool_call_inputs(json_objects: list):
    for obj in json_objects:
        action_type = obj['arguments']['action']
        assert action_type in ["resize"], f"Unknown Tool Type: {action_type}. Available function tools are: `resize`"
        assert len(json_objects) == 1, "You should only call function `resize` once per function call."
    return action_type

def prepare_tool_call_inputs_video(json_objects: list):
    assert len(json_objects) == 1
    try:
        obj = json_objects[0]
        start_time: float = float(obj['segment'][0])
        end_time: float = float(obj['segment'][1])
        fps: float = float(obj['fps'])
    except Exception as e:
        raise ValueError(f"Invalid input for function `extract_video_clip`. Error: {e}, input: {json_objects[0]}")
    return {
        "start_time": start_time,
        "end_time": end_time,
        "fps": fps
    }


def extract_video_clip(video_path, start_time, end_time, fps: float = 4.0, max_pixels: int = 100352, min_pixels: int = 25088, max_frames: int = 16, storage_system: str = 'local', logger=None):
    try:

        function_start_time = time.time()
        def _log_info(msg):
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

        def _log_warning(msg):
            if logger is not None:
                logger.warning(msg)
            else:
                print(f"WARNING: {msg}")
        high_fps = float(os.environ.get('HIGHFPS', -1))
        if high_fps > 0:
            _log_info(f"setting highfps to {high_fps}")
            fps = high_fps
        if storage_system == 'auto':
            storage_system = 'local'
        if storage_system != 'local':
            raise NotImplementedError("Only local storage is supported for video loading.")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path {video_path} does not exist.")
        extract_frame_num = int(fps * (end_time - start_time))
        if extract_frame_num > max_frames:
            error_msg = f"Number of frames extracted ({extract_frame_num}) exceeds the maximum allowed ({max_frames})."
            raise ValueError(error_msg)
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=8)
        
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()
        
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        
        start_frame = max(0, min(start_frame, total_frame_num - 1))
        end_frame = max(start_frame, min(end_frame, total_frame_num - 1))
        
        frame_idx = np.linspace(start_frame, end_frame, extract_frame_num).astype(int).tolist()
        frame_time = [i / video_fps for i in frame_idx]
        if not frame_idx:
            return {'frame_time': frame_time, 'frames': np.array([])}
        
        _log_info(f'Extracting frames from {start_time:.2f}s to {end_time:.2f}s, Frame range: {start_frame} to {end_frame}, Total frames to extract: {len(frame_idx)}')

        try:
            frames = vr.get_batch(frame_idx).asnumpy()
        except Exception:
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            frames = vr.get_batch(frame_idx).asnumpy()
        video = np.array(frames)
        n_frames, h, w, _ = video.shape
        
        pil_images = []
        for i in range(n_frames):
            img = Image.fromarray(video[i].astype('uint8'))
            
            current_pixels = h * w
            if current_pixels > max_pixels:
                scale_factor = np.sqrt(max_pixels / current_pixels)
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                
                if new_h * new_w < min_pixels:
                    min_scale_factor = np.sqrt(min_pixels / (new_h * new_w))
                    new_h = int(new_h * min_scale_factor)
                    new_w = int(new_w * min_scale_factor)
                    
                img = img.resize((new_w, new_h), Image.LANCZOS)
            
            pil_images.append(img)
        
        assert len(frame_time) == len(pil_images)
        del vr
        _log_info(f'Extracted {len(frame_idx)} frames in {time.time() - function_start_time:.2f} seconds')
        return {'frame_time': frame_time, 'frames': pil_images}
    except Exception as e:
        error_msg = f"Video Extraction error: {e}"
        _log_warning(error_msg)
        return f"<tool_response>\nError: {e}<tool_response>"
