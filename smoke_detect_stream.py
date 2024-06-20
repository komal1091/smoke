import streamlit as st
from PIL import Image
import cv2
import os
from ultralytics import YOLO
import tempfile
from moviepy.editor import VideoFileClip
import time 

st.title('Smoke Detection App')

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    model = YOLO('/home/codezeros/Documents/fire&smoke detection/Test/best.pt')
    frames = []
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Write processed frames to output video
    fourcc = cv2.VideoWriter_fourcc(*'MP4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform inference on the frame
        results = model(frame_rgb)
        label = "smoke"

        for result in results:
            for box in result.boxes.xyxy:
                conf = result.boxes.conf[0]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 5)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # frame_placeholder.image(frame_rgb, channels='RGB')
        out.write(frame)
        frames.append(frame)

    # Close the video capture object
    cap.release()
    out.release()

    
    # for frame in frames:
    #     out.write(frame)
    # out.release()



def convert_to_mp4(input_file, output_file):
    video = VideoFileClip(input_file)
    video.write_videofile(output_file, codec='libx264', audio_codec='aac')
    video.close()


# st.title('Fire & Smoke Detection App')

uploads_dir = 'uploads'
os.makedirs(uploads_dir, exist_ok=True)

uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])



if uploaded_file is not None:
    video_path = os.path.join(uploads_dir, uploaded_file.name)
    # with open(video_path, "wb") as f:
    #     f.write(uploaded_file.getbuffer())

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if st.button("Process Video"):
        start_time = time.time()
        # frame_placeholder = st.empty()
        output_video_path = os.path.join(uploads_dir, "output_video.mp4")
        process_video(tfile.name, output_video_path)
        st.markdown("---")
        st.subheader("Output Video")

        output_video_path_conv = os.path.join(uploads_dir, "new_output_video.mp4")
        convert_to_mp4(output_video_path, output_video_path_conv)
        end_time = time.time()
        print("Total time to consumed :", end_time-start_time)

        # Display the processed video
        st.video(output_video_path_conv)
        st.session_state['video_bytes'] = open(output_video_path, 'rb').read()
        st.write(f"Output video saved at: {output_video_path}")
    if 'video_bytes' in st.session_state:
        st.markdown("---")
        st.download_button(label = "Download output video",
                            data = st.session_state['video_bytes'],
                            file_name = "output_video.mp4",
                            mime="video/mp4")
            


