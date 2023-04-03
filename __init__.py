import cv2
import torchvision.transforms as transforms
import einops
import os
video_captures = {}

VIDEO_EXTENSIONS = ('.avi', '.mp4', '.mkv', '.wmv', '.mov', '.mpeg', '.mpg', '.webm', '.vob', '.3gp', '.ogg', '.flv', '.gif',
                    '.ts', '.mts', '.m2ts', '.dv', '.asf', '.amv', '.m4p', '.m4v', '.mod', '.mxf', '.nsv', '.tp', '.trp', '.tsv', '.wtv')
# Thanks ChatGPT


class VideoFrameExtractor:
    video_input_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "../../input/videos")

    @classmethod
    def INPUT_TYPES(s):
        if not os.path.exists(s.video_input_dir):
            os.mkdir(s.video_input_dir)
        return {"required": {"video": (sorted(list(filter(lambda filename: filename.endwiths(VIDEO_EXTENSIONS), os.listdir(s.video_input_dir)))),),
                             "frame_index": ("INT", {"default": 1, "min": 1, "max": 2 ** 53 - 1, "step": 1})}}
        #https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract_frame"

    CATEGORY = "video"

    def extract_frame(self, video, frame_index):
        if not video in video_captures:
            video_captures[video] = cv2.VideoCapture(
                os.path.join(self.video_input_dir, video))
        cap = video_captures[video]
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert frame_index <= frame_count, f"FrameIndexOutOfBound: For the video '{video}', expected frame_index <= {frame_count}, but got {frame_index}. \n Please note that frame_index should be within the range of available frames for it."

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1) #Actual frame_index starts from 0
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return einops.rearrange(transforms.ToTensor()(frame), "ch w h -> 1 w h ch")
