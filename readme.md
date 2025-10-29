# Real-Time Crowd Density Estimation and Zonal Analysis System

## 1. Project Abstract

This application provides a robust solution for real-time crowd counting and density analysis from image and video sources. It leverages a deep learning-based approach to generate high-fidelity density maps, enabling accurate estimations of crowd size. The system features a configurable zonal analysis module that monitors specific areas of interest and issues alerts based on predefined density thresholds, making it an effective tool for crowd management and safety monitoring.

The front-end is delivered through an interactive web interface built with Streamlit, ensuring ease of use and clear visualization of results.

## 2. Key Features

-   **Multi-Source Input:** Supports analysis of both static images (`JPG`, `PNG`) and video files (`MP4`, `AVI`, `MOV`).
-   **Deep Learning Model:** Employs a Keras-based Convolutional Neural Network (CNN) to generate precise density maps from input frames.
-   **Zonal Density Monitoring:** Allows for the definition of multiple, distinct zones within the frame for granular analysis.
-   **Threshold-Based Alerting System:** Triggers multi-level alerts (Safe, Caution, Alert) for each zone when crowd counts exceed user-defined thresholds.
-   **Interactive Visualization:** Renders a comprehensive output including the source video with annotated zones, real-time crowd counts, and a corresponding heat map for intuitive density assessment.

## 3. Technical Architecture

The system's processing pipeline follows a sequential workflow:

1.  **Frame Ingestion:** An image or video frame is captured from the user-uploaded source.
2.  **Preprocessing:** The frame is resized, normalized, and standardized to meet the input requirements of the neural network.
3.  **Model Inference:** The preprocessed frame is passed through the pre-trained Keras model, which outputs a crowd density map.
4.  **Post-processing & Analysis:** The total crowd count is estimated by summing the values in the density map. The map is then segmented according to the defined zones, and zone-specific counts are calculated.
5.  **Status Evaluation:** Each zone's count is compared against its configured thresholds to determine the current status (Safe, Caution, or Alert).
6.  **Output Visualization:** The results are rendered on the Streamlit front-end, displaying the annotated video frame and the generated density map.

## 4. Technology Stack

-   **Backend Framework:** Python 3.8+
-   **Web Interface:** Streamlit
-   **Machine Learning:** Keras (TensorFlow backend)
-   **Image/Video Processing:** OpenCV, Pillow
-   **Numerical Computation:** NumPy

## 5. Installation and Execution

### 5.1. Prerequisites

-   Python 3.8 or higher
-   Git
-   Git Large File Storage (LFS)

Ensure Git LFS is installed and initialized to handle the large model weights file.
```bash
# Install Git LFS on your system (e.g., using 'brew', 'apt', or from the official website)
git lfs install
```

### 5.2. Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/PBL.git](https://github.com/your-username/PBL.git)
    cd PBL
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a `requirements.txt` file with the following contents:
    ```txt
    streamlit
    opencv-python-headless
    numpy
    Pillow
    tensorflow
    matplotlib
    ```
    Install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Verify Project Structure:**
    Confirm that the model architecture and weights are located in the correct directories:
    ```
    /
    ├── models/
    │   └── Model.json
    ├── weights/
    │   └── model_A_weights.h5
    └── app.py
    ```

### 5.3. Running the Application

Execute the following command in the root directory of the project:
```bash
streamlit run app.py
```
The application will be accessible via a local URL displayed in the terminal.

## 6. System Configuration

The primary configuration for zonal analysis is managed through the `ZONES` list within the main script. Users can modify this structure to define custom monitoring areas and alerting criteria.

**Example Configuration:**
```python
ZONES = [
    {"name": "Zone 1", "rect": (0, 0, 320, 480), "thresholds": {"Caution": 10, "Alert": 20}},
    {"name": "Zone 2", "rect": (320, 0, 320, 480), "thresholds": {"Caution": 15, "Alert": 25}},
]
```
-   `name`: A unique identifier for the zone.
-   `rect`: A tuple `(x, y, width, height)` defining the zone's bounding box.
-   `thresholds`: A dictionary specifying the crowd count that triggers `Caution` and `Alert` statuses.

## 7. Contributors

This project was developed by the following team members:

-   **AI & Model Implementation**: Nireeksha
-   **Data Analysis & Core Logic**: Neema
-   **User Interface & Alerting System**: Dhanush

## 8. License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
