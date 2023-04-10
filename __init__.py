import server  # Root ComfyUI
import inspect
import cv2
import einops
import os
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import imageio_ffmpeg
import ffmpeg

import json
import numpy as np

INPUT_VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mkv', '.wmv', '.mov', '.mpeg', '.mpg', '.webm', '.vob', '.3gp', '.ogg', '.flv', '.gif',
                          '.ts', '.mts', '.m2ts', '.dv', '.asf', '.amv', '.m4p', '.m4v', '.mod', '.mxf', '.nsv', '.tp', '.trp', '.tsv', '.wtv')
# Thanks ChatGPT

BASE_COMFYUI_PATH = os.path.dirname(inspect.getfile(server))


class Video_Frame_Extractor:
    video_input_dir = os.path.join(BASE_COMFYUI_PATH, "input/videos")

    @classmethod
    def INPUT_TYPES(s):
        os.makedirs(s.video_input_dir, exist_ok=True)
        return {"required": {"video": (sorted(list(filter(lambda filename: filename.endswith(INPUT_VIDEO_EXTENSIONS), os.listdir(s.video_input_dir)))),),
                             "frame_index": ("INT", {"default": 1, "min": 1, "max": 2 ** 53 - 1, "step": 1})}}
        # https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_frame"
    CATEGORY = "video"

    def extract_frame(self, video, frame_index):
        cap = cv2.VideoCapture(os.path.join(self.video_input_dir, video))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert frame_index <= frame_count, f"FrameIndexOutOfBound: For the video '{video}', expected frame_index <= {frame_count}, but got {frame_index}. \n Please note that frame_index should be within the range of available frames for it."
        # Actual frame_index starts from 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ret, frame = cap.read()
        cap.release()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return (einops.rearrange(torch.from_numpy(frame), "h w ch -> 1 h w ch"), )


class Save_Frame_To_Folder:
    parent_output_dir = os.path.join(BASE_COMFYUI_PATH, "output/frames")

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"folder_name": ("STRING", {"default": "Bocchi The Canny Edge", "multiline": False}),
                             "images": ("IMAGE",),
                             "format": (["png", "jpg", "webp"], {"default": "png"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
    RETURN_TYPES = ()
    FUNCTION = "save_frames"

    OUTPUT_NODE = True

    CATEGORY = "video"

    def save_frames(self, folder_name, images, format, prompt=None, extra_pnginfo=None):
        full_output_folder = os.path.join(self.parent_output_dir, folder_name)
        os.makedirs(full_output_folder, exist_ok=True)
        counter = len(os.listdir(full_output_folder)) + 1

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            file = f"{counter}.{format}"

            if format == "png":
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                img.save(os.path.join(full_output_folder, file),
                         pnginfo=metadata, compress_level=4)
            else:
                img.info = {"prompt": prompt, **extra_pnginfo}
                img.save(os.path.join(full_output_folder, file))

            results.append({
                "filename": file,
                "folder_name": folder_name,
                "format": format
            })
            counter += 1

        return {"ui": {"images": results}}


class Simple_Frame_Folder_To_Video:
    frame_folder_input_dir = os.path.join(BASE_COMFYUI_PATH, "output/frames")
    video_output_dir = os.path.join(BASE_COMFYUI_PATH, "output/videos")

    @classmethod
    def INPUT_TYPES(s):
        os.makedirs(s.frame_folder_input_dir, exist_ok=True)
        os.makedirs(s.video_output_dir, exist_ok=True)
        return {"required": {"folder_name": ([folder for folder in os.listdir(s.frame_folder_input_dir) if os.path.isdir(os.path.join(s.frame_folder_input_dir, folder))],),
                             "video_file_name": ("STRING", {"default": "Bocchi The Canny Edge.mp4", "multiline": False}),
                             "fps": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 8056.0, "step": 1})}}
    RETURN_TYPES = ()
    FUNCTION = "save_video_from_folder"

    OUTPUT_NODE = True

    CATEGORY = "video"

    def save_video_from_folder(self, folder_name, video_file_name, fps):
        video_save_path = os.path.join(self.video_output_dir, video_file_name)
        if os.path.exists(video_save_path):
            return
        (
            ffmpeg
            .input(f'{os.path.join(self.frame_folder_input_dir, folder_name)}/*', pattern_type='glob', framerate=fps)
            .output(video_save_path)
            .run()
        )


NODE_CLASS_MAPPINGS = {
    "Video Frame Extractor": Video_Frame_Extractor,
    "Save Frame To Folder": Save_Frame_To_Folder,
    "Simple Frame Folder To Video": Simple_Frame_Folder_To_Video
}
