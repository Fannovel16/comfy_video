import server  # Root ComfyUI
import inspect
import cv2
import einops
import os
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from .video_writer_thread import video_writing_queue
import threading
 
import json
import numpy as np

INPUT_VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mkv', '.wmv', '.mov', '.mpeg', '.mpg', '.webm', '.vob', '.3gp', '.ogg', '.flv', '.gif',
                          '.ts', '.mts', '.m2ts', '.dv', '.asf', '.amv', '.m4p', '.m4v', '.mod', '.mxf', '.nsv', '.tp', '.trp', '.tsv', '.wtv')
# Thanks ChatGPT
DEFAULT_COMMON_FOURCC = "H264"
KNOWN_AUTO_EXT_FOURCC_MAP = {".webm": "VP90", ".ogg": "THEO"}

video_writers = {}

class Video_Frame_Extractor:
    video_input_dir = os.path.join(os.path.dirname(
        inspect.getfile(server)), "input/videos")

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

class Write_Frame_To_Video_Gif:
    video_output_dir = os.path.join(os.path.dirname(
        inspect.getfile(server)), "output/videos")

    @classmethod
    def INPUT_TYPES(s):
        os.makedirs(s.video_output_dir, exist_ok=True)
        return {"required": {"file_name": ("STRING", {"default": "Bocchi The Canny Edge.mp4", "multiline": False}),
                             "fourCC": ("STRING", {"default": "AUTO", "multiline": False}),
                             "fps": ("FLOAT", {"default": 24.0, "min": 1.0}),
                             "images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "write_frame"
    CATEGORY = "video"
    OUTPUT_NODE = True

    def write_frame(self, file_name, fourCC, fps, images):
        fourCC = fourCC.strip().upper()
        assert len(fourCC) == 4, f"InvalidFourCCFormat: Expected length of FourCC input = 4, got {len(fourCC)} ({fourCC})"
        assert fourCC.isalnum(), f"InvalidFourCCFormat: Expected FourCC input is alphanumeric, got {fourCC}"
        save_loc = os.path.join(self.video_output_dir, file_name)
        _, file_ext = os.path.splitext(file_name)

        if fourCC == "AUTO":
            fourCC = KNOWN_AUTO_EXT_FOURCC_MAP[file_ext] if fourCC in KNOWN_AUTO_EXT_FOURCC_MAP else DEFAULT_COMMON_FOURCC

        for image_tensor in images:
            if file_ext == ".gif":
                frames = []
                if os.path.exists(save_loc):
                    with Image.open(save_loc) as im:
                        for i in range(im.n_frames):
                            im.seek(i)
                            frames.append(im.copy())

                i = 255. * image_tensor.detach().cpu().numpy()
                frame = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                frames.append(frame)
                frames[0].save(save_loc, format='GIF', save_all=True,
                            append_images=frames[1:], duration=1 / fps, loop=0)
                continue
            i = 255. * image_tensor.detach().cpu().numpy()
            frame = cv2.cvtColor(np.clip(i, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            #if save_loc not in video_writers:
            #    video_writers[save_loc] = cv2.VideoWriter(save_loc, cv2.VideoWriter_fourcc(*fourCC), fps,
            #                    (frame.shape[1], frame.shape[0]))
            #video_writers[save_loc].write(frame)
            writing_completed_event = threading.Event()
            video_writing_queue.put(((save_loc, fourCC, fps, (frame.shape[1], frame.shape[0]), frame), writing_completed_event))
            writing_completed_event.wait()
            writing_completed_event.clear()
        return {"ui": {"video_name": file_name}}

class Save_Frame_To_Folder:
    parent_output_dir = video_output_dir = os.path.join(os.path.dirname(
        inspect.getfile(server)), "output")
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"folder_name": ("STRING", {"default": "Bocchi The Canny Edge", "multiline": False}),
                             "images": ("IMAGE",),
                             "format": (["png", "jpg", "webp"], {"default": "png"})},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}
    RETURN_TYPES = ()
    FUNCTION = "save_frames"

    OUTPUT_NODE = True

    CATEGORY = "image"

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
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            else:
                img.info = {"prompt": prompt, **extra_pnginfo}
                img.save(os.path.join(full_output_folder, file))
            
            results.append({
                "filename": file,
                "folder_name": folder_name,
                "format": format
            })
            counter += 1

        return { "ui": { "images": results } }

NODE_CLASS_MAPPINGS = {
    "Video Frame Extractor": Video_Frame_Extractor,
    "Write Frame To Video Or Gif": Write_Frame_To_Video_Gif,
    "Save Frame To Folder": Save_Frame_To_Folder
}
