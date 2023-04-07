#NOTE: In the Write_Frame_To_Video_Gif node, when VideoWriter writes some frames then closes itself, only the last frame is saved
#The reason? Idk.
import threading
from queue import Queue
from time import sleep
import timeit
import cv2

video_writing_queue = Queue()

def queue_func(video_writing_queue):
    video_writers = {}
    while True:
        sleep(0.1)
        video_writer_keys = list(video_writers.keys())
        for key in video_writer_keys:
            start_time, video_writer = video_writers[key]
            if timeit.default_timer() - start_time > 5:
                video_writer.release()
                del video_writers[key]
        
        if video_writing_queue.qsize() == 0:
            continue #Only a video writing task is made at the same time, and only this thread can get the data
        
        input_data, writing_completed_event = video_writing_queue.get()
        video_save_loc, fourCC, fps, frame_shape, frame = input_data
        
        if video_save_loc not in video_writers:
            video_writers[video_save_loc] = [timeit.default_timer(), cv2.VideoWriter(video_save_loc, cv2.VideoWriter_fourcc(*fourCC), fps, frame_shape)]
        
        start_time, video_writer = video_writers[video_save_loc]
        video_writer.write(frame)
        video_writers[video_save_loc][0] = timeit.default_timer()
        writing_completed_event.set()
        
video_writer_thread = threading.Thread(target=queue_func, args=(video_writing_queue,))
video_writer_thread.start()
