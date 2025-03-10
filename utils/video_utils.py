from decord import VideoReader, cpu
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def load_video(video_file, max_num_frames=None, target_fps=None):
    import av
    import numpy as np
    assert max_num_frames is None or target_fps is None, "max_num_frames and target_fps cannot both be specified"
    vr = VideoReader(str(video_file), ctx=cpu(0), num_threads=1)
    fps = vr.get_avg_fps()
    def record_video_length_stream(container):
        video = container.streams.video[0]
        video_length = float(video.duration * video.time_base)  # in seconds
        return video_length
    duration = record_video_length_stream(av.open(video_file))
    total_valid_frames = int(duration * fps)
    if max_num_frames is not None:
        num_frames = min(max_num_frames, total_valid_frames)
        frame_indices = [int(total_valid_frames / num_frames) * i for i in range(num_frames)]
    elif target_fps is not None:
        # Compute frame sampling interval
        frame_interval = fps / target_fps
        frame_indices = np.arange(0, total_valid_frames, frame_interval).astype(int)

    frames = vr.get_batch(frame_indices)
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    else:
        frames = frames.asnumpy()
    frame_timestamps = [frame_index / fps for frame_index in frame_indices]
    
    return [Image.fromarray(fr).convert("RGB") for fr in frames], frame_timestamps, total_valid_frames

def compute_frame_timestamps(duration, max_num_frames=16):
    if duration > max_num_frames:
        return [duration / max_num_frames * i for i in range(max_num_frames)]
    else:
        return [i for i in range(int(duration))]


def make_video(max_frame, video_frames, per_frame_subtitles, subtitle_by_frame):

    subtitle_images = []
    for frame_idx in range(max_frame):
        frame = video_frames[frame_idx]
        if subtitle_by_frame is not None:
            image_width, image_height = frame.size
            draw = ImageDraw.Draw(frame)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30) 
            
            text = "\n".join(per_frame_subtitles[frame_idx])
            bbox = draw.textbbox(xy=(0, 0), text=text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (image_width - text_width) / 2
            y = image_height - text_height - 10  # 10 pixels from the bottom
            draw.text((x, y), text, font=font, fill="white")
        subtitle_images.append(frame)
    video_path = f"examples/video_frames{max_frame}.mp4"
    save_video_from_pil(subtitle_images, video_path, fps=1)
    
def save_video_from_pil(subtitle_images, video_path="subtitle_video.mp4", fps=1):
    # Get frame dimensions correctly (PIL gives (width, height), swap needed)
    frame_width, frame_height = subtitle_images[0].size

    # Define video codec and writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'H264' if needed
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

    for frame in subtitle_images:
        # Convert PIL Image to NumPy array
        frame_np = np.array(frame)
        # Convert RGB (PIL) to BGR (OpenCV expects BGR)
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        # Write frame
        video_writer.write(frame_np)

    # Release video writer
    video_writer.release()
    print(f"Video saved to {video_path}")
