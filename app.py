import cv2 as cv
import mediapipe as mp
import numpy as np
import gradio as gr


def blur_faces_image(image):
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    H, W = image.shape[:2]

    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0) as face_detection:
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)

        if out.detections is not None:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box

                x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)
                # Apply blur to the face region
                face_roi = image[y1:y1 + h, x1:x1 + w:]
                blurred_face = cv.blur(face_roi, (50, 50))
                image[y1:y1 + h, x1:x1 + w] = blurred_face

    return cv.cvtColor(image, cv.COLOR_BGR2RGB)  # Convert BGR back to RGB for Gradio

def blur_faces_video(video_path, output_path="blurred_output.mp4"):
    cap = cv.VideoCapture(video_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define video writer
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        blurred_frame = blur_faces_image(frame_rgb)
        frame_bgr = cv.cvtColor(blurred_frame, cv.COLOR_RGB2BGR)

        out.write(frame_bgr)

    cap.release()
    out.release()

    return output_path
def process_video_interface(video_file):
    video_path = video_file["name"] if isinstance(video_file, dict) else video_file
    return blur_faces_video(video_path)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕵️‍♂️ Face Blur App")

    with gr.Tab("Image Face Blur"):
        with gr.Row():
            image_input = gr.Image(label="Upload Image")
            image_output = gr.Image(label="Blurred Output")

        image_btn = gr.Button("Blur Faces")
        image_btn.click(fn=blur_faces_image, inputs=image_input, outputs=image_output)
        gr.Examples(
            examples=["examples/testImg.png"],
            inputs=image_input,
            label="Try Example Image"
        )
    with gr.Tab("Video Face Blur"):
        with gr.Row():
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Video(label="Blurred Video Output")

        video_btn = gr.Button("Blur Faces in Video")
        video_btn.click(fn=process_video_interface, inputs=video_input, outputs=video_output)
        gr.Examples(
            examples=["examples/testVideo.mp4"],
            inputs=video_input,
            label="Try Example Video"
        )
demo.launch()
