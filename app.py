import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import model_from_json, Model
from matplotlib import cm as c
import tempfile

# --- Part 3: Neema (Analysis & Logic) ---
ZONES = [
    {"name": "Zone 1", "rect": (0, 0, 320, 480), "thresholds": {"Caution": 10, "Alert": 20}},
    {"name": "Zone 2", "rect": (320, 0, 320, 480), "thresholds": {"Caution": 10, "Alert": 20}},
]

COLORS = {
    "Safe": (0, 255, 0),  # Green
    "Caution": (0, 255, 255),  # Yellow
    "Alert": (0, 0, 255),  # Red
}

def load_model():
    # Function to load and return neural network model 
    json_file = open('models/Model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'Model': Model})
    loaded_model.load_weights("weights/model_A_weights.h5")
    return loaded_model

def preprocess_frame(frame):
    """
    Preprocesses a single frame for the crowd counting model.
    """
    # Resize the frame for faster processing
    frame = cv2.resize(frame, (640, 480))
    # Convert frame to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    im = Image.fromarray(frame)
    
    # Convert to numpy array
    im = np.array(im)
    
    # Normalize and standardize
    im = im / 255.0
    im[:,:,0] = (im[:,:,0] - 0.485) / 0.229
    im[:,:,1] = (im[:,:,1] - 0.456) / 0.224
    im[:,:,2] = (im[:,:,2] - 0.406) / 0.225
    
    # Add batch dimension
    im = np.expand_dims(im, axis=0)
    
    return im

def analyze_density_map(density_map, zones):
    """
    Analyzes the density map to calculate crowd count in each zone.
    """
    total_count = np.sum(density_map)
    zone_results = []

    for zone in zones:
        x, y, w, h = zone["rect"]
        # Adjust zone coordinates for the density map size
        dm_h, dm_w = density_map.shape
        frame_h, frame_w = 480, 640
        dm_x = int(x * dm_w / frame_w)
        dm_y = int(y * dm_h / frame_h)
        dm_w = int(w * dm_w / frame_w)
        dm_h = int(h * dm_h / frame_h)

        zone_map = density_map[dm_y:dm_y+dm_h, dm_x:dm_x+dm_w]
        zone_count = np.sum(zone_map)

        status = "Safe"
        if zone_count > zone["thresholds"]["Alert"]:
            status = "Alert"
        elif zone_count > zone["thresholds"]["Caution"]:
            status = "Caution"

        zone_results.append({"name": zone["name"], "rect": zone["rect"], "count": zone_count, "status": status})

    return total_count, zone_results

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Crowd Counting Application")
    
    # Load the model
    model = load_model()

    app_mode = st.sidebar.selectbox('Select Mode', ['Image', 'Video'])

    if app_mode == 'Image':
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Convert the file to an opencv image.
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            st.image(frame, channels="BGR", caption="Uploaded Image")

            # Preprocess the frame
            processed_frame = preprocess_frame(frame.copy())
            
            # --- Part 2: Nireeksha (AI & Model) ---
            density_map = model.predict(processed_frame)
            
            # Reshape density map
            density_map = density_map.reshape(density_map.shape[1], density_map.shape[2])

            # --- Part 3: Neema (Analysis & Logic) ---
            total_count, zone_results = analyze_density_map(density_map, ZONES)
            st.header(f"Total Estimated People: {int(total_count)}")

            for zone in zone_results:
                st.subheader(f'{zone["name"]}: {int(zone["count"])} people')
                if zone["status"] == "Alert":
                    st.error(f"ALERT: Overcrowding in {zone['name']}")
                elif zone["status"] == "Caution":
                    st.warning(f"CAUTION: High density in {zone['name']}")
                else:
                    st.success(f"SAFE: Density is normal in {zone['name']}")

            # --- Part 4: Dhanush (UI & Alerting) ---
            display_frame = cv2.resize(frame, (640, 480))
            alert_message = ""
            for zone in zone_results:
                x, y, w, h = zone["rect"]
                color = COLORS[zone["status"]]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f'{zone["name"]}: {zone["count"]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if zone["status"] == "Alert":
                    alert_message = f"ALERT: Overcrowding in {zone['name']}"

            if alert_message:
                cv2.putText(display_frame, alert_message, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Display the resulting frame and density map
            density_map_vis = (density_map - np.min(density_map)) / (np.max(density_map) - np.min(density_map) + 1e-5)
            density_map_vis = (c.jet(density_map_vis)[:, :, :3] * 255).astype(np.uint8)
            density_map_vis = cv2.resize(density_map_vis, (display_frame.shape[1], display_frame.shape[0]))

            st.image(display_frame, channels="BGR", caption="Processed Image")
            st.image(density_map_vis, caption="Density Map")

    elif app_mode == 'Video':
        st.sidebar.markdown('---')
        frame_skip = st.sidebar.slider('Process 1 frame every N frames', 1, 10, 2)
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            frame_count = 0

            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    # Preprocess the frame
                    processed_frame = preprocess_frame(frame.copy())
                    
                    # --- Part 2: Nireeksha (AI & Model) ---
                    density_map = model.predict(processed_frame)
                    
                    # Reshape density map
                    density_map = density_map.reshape(density_map.shape[1], density_map.shape[2])

                    # --- Part 3: Neema (Analysis & Logic) ---
                    total_count, zone_results = analyze_density_map(density_map, ZONES)
                    
                    # --- Part 4: Dhanush (UI & Alerting) ---
                    display_frame = cv2.resize(frame, (640, 480))
                    alert_message = ""
                    for zone in zone_results:
                        x, y, w, h = zone["rect"]
                        color = COLORS[zone["status"]]
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(display_frame, f'{zone["name"]}: {zone["count"]:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        if zone["status"] == "Alert":
                            alert_message = f"ALERT: Overcrowding in {zone['name']}"

                    if alert_message:
                        cv2.putText(display_frame, alert_message, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                    stframe.image(display_frame, channels="BGR", caption=f"Total Estimated People: {int(total_count)}")
                
                frame_count += 1

            vf.release()

if __name__ == '__main__':
    main()