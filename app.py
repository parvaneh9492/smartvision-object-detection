import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from detect import detect_objects

st.set_page_config(page_title="SmartVision", layout="centered")
st.title("🔍 SmartVision: Real-Time Object Detection")

option = st.sidebar.radio("Choose Input Mode", ("Upload Image", "Upload Video", "Use Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            image_path = tmp.name

        annotated, counts, crops = detect_objects(image_path)

        st.subheader("📊 Detected Object Summary")
        for label, count in counts.items():
            st.write(f"**{label}**: {count}")

        st.image(annotated, channels="BGR", use_column_width=True)

        if crops:
            st.subheader("🖼️ Cropped Objects")
            for i, (label, crop_img) in enumerate(crops):
                st.image(crop_img, caption=label, width=150)
                # Convert to PIL and enable download
                pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                    pil_img.save(f.name)
                    with open(f.name, "rb") as file:
                        st.download_button(f"Download {label}_{i}", file, file_name=f"{label}_{i}.png")
                        
elif option == "Upload Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        st.video(uploaded_video)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getvalue())
            input_path = tmp.name
            output_path = "output_detected.mp4"

        with st.spinner("Processing video..."):
            from detect import detect_video
            result_path = detect_video(input_path, output_path)

        st.success("Detection complete!")
        st.video(result_path)

        with open(result_path, "rb") as file:
            st.download_button("Download Detected Video", file, file_name="detected_video.mp4")
                        

elif option == "Use Webcam":
    st.warning("Click below to start webcam and press `Q` to quit.", icon="🎥")
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imwrite("live_frame.jpg", frame)
            annotated, _, _ = detect_objects("live_frame.jpg")
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            stframe.image(annotated, channels="RGB")

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

