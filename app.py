import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import time
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import av
import tempfile
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns  # Import seaborn for plotting
import whisper
import subprocess
import os
import warnings
import openai
import json
from bs4 import BeautifulSoup


warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# =========================
# Global Variables
# =========================
FONTS = cv.FONT_HERSHEY_SIMPLEX
CENTER_THRESHOLD = 0.5
SIDE_THRESHOLD = 3
BLINK_THRESHOLD = 2
DISCOUNT_CENTER = 0.3
DISCOUNT_SIDE = 0.3
DISCOUNT_EYES = 0.5

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
            263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
             133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

mp_face_mesh = mp.solutions.face_mesh

# Initialize global variables for tracking
focus_score = 50  # Initialized start focus from 50
last_look_centered_time = None
not_looking_start_time = None
blink_start_time = None
total_blinks = 0
blink_detected = False
eyes_closed_start_time = None
# Variables to track the last time we increased or decreased the focus score
last_focus_increase_time = None
last_focus_decrease_time = None

# Initialize Whisper model and OpenAI API key placeholder
model = None  # We will load the model when needed
openai_api_key = None  # We will prompt the user to enter the API key

# =========================
# Helper Functions for Focus Detection
# =========================

def euclidean_distance(point1, point2):
    return math.hypot(point2[0] - point1[0], point2[1] - point1[1])

def blink_ratio(landmarks, right_indices, left_indices):
    rh_distance = euclidean_distance(landmarks[right_indices[0]], landmarks[right_indices[8]])
    rv_distance = euclidean_distance(landmarks[right_indices[12]], landmarks[right_indices[4]])
    lh_distance = euclidean_distance(landmarks[left_indices[0]], landmarks[left_indices[8]])
    lv_distance = euclidean_distance(landmarks[left_indices[12]], landmarks[left_indices[4]])

    if rv_distance == 0 or lv_distance == 0:
        return float('inf')

    re_ratio = rh_distance / rv_distance
    le_ratio = lh_distance / lv_distance
    return (re_ratio + le_ratio) / 2

def landmarks_detection(img, results):
    img_height, img_width = img.shape[:2]
    return [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]

def eye_direction(eye_points, iris_center, ratio):
    eye_left = np.min(eye_points[:, 0])
    eye_right = np.max(eye_points[:, 0])

    hor_range = eye_right - eye_left
    iris_x, _ = iris_center

    if ratio > 5.5:
        return "Blink"
    elif iris_x < eye_left + hor_range * 0.3:
        return "Left"
    elif iris_x > eye_right - hor_range * 0.3:
        return "Right"
    else:
        return "Center"

def process_frame(frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, last_focus_increase_time, last_focus_decrease_time):
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    eye_direction_text = "Not Detected"
    face_position = "Not Detected"

    current_time = time.time()

    if results.multi_face_landmarks:
        mesh_points = landmarks_detection(frame, results)
        
        # Face position monitoring
        face_3d = []
        face_2d = []
        for idx, lm in enumerate(results.multi_face_landmarks[0].landmark):
            if idx in [1, 33, 61, 199, 263, 291]:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                face_2d.append([x, y])
                face_3d.append([x, y, lm.z])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = 1 * frame.shape[1]
        cam_matrix = np.array([[focal_length, 0, frame.shape[1] / 2],
                               [0, focal_length, frame.shape[0] / 2],
                               [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

        x = angles[0] * 360
        y = angles[1] * 360

        if y < -10:
            face_position = "Looking Left"
        elif y > 10:
            face_position = "Looking Right"
        elif x < -10:
            face_position = "Looking Down"
        elif x > 20:
            face_position = "Looking Up"
        else:
            face_position = "Forward"

        # Eye direction and blink detection
        ratio = blink_ratio(mesh_points, RIGHT_EYE, LEFT_EYE)
        left_iris_points = np.array([mesh_points[i] for i in LEFT_IRIS], dtype=np.int32)
        right_iris_points = np.array([mesh_points[i] for i in RIGHT_IRIS], dtype=np.int32)
        (l_cx, l_cy), l_radius = cv.minEnclosingCircle(left_iris_points)
        (r_cx, r_cy), r_radius = cv.minEnclosingCircle(right_iris_points)
        center_left = np.array([l_cx, l_cy], dtype=np.int32)
        center_right = np.array([r_cx, r_cy], dtype=np.int32)
        left_eye_direction = eye_direction(np.array([mesh_points[p] for p in LEFT_EYE]), center_left, ratio)
        right_eye_direction = eye_direction(np.array([mesh_points[p] for p in RIGHT_EYE]), center_right, ratio)

        if left_eye_direction == right_eye_direction:
            eye_direction_text = left_eye_direction
        else:
            eye_direction_text = left_eye_direction if left_eye_direction in ["Left", "Right"] else right_eye_direction

        # Focus scoring algorithm
        if face_position == "Forward" and eye_direction_text == "Center":
            if last_look_centered_time is None:
                last_look_centered_time = current_time
            not_looking_start_time = None
            if current_time - last_look_centered_time >= CENTER_THRESHOLD:
                # Increase focus score by 5% every 1 second when increasing
                if last_focus_increase_time is None or current_time - last_focus_increase_time >= 0.1:
                    focus_score = min(100, focus_score + 0.3)
                    last_focus_increase_time = current_time
        else:
            last_look_centered_time = None
            if not not_looking_start_time:
                not_looking_start_time = current_time
            elif current_time - not_looking_start_time >= SIDE_THRESHOLD:
                # Decrease focus score by 5% every 1 second when decreasing
                if last_focus_decrease_time is None or current_time - last_focus_decrease_time >= 0.1:
                    focus_score = max(0, focus_score - DISCOUNT_SIDE)
                    last_focus_decrease_time = current_time

        if ratio > 5.5:
            if not blink_detected:
                blink_start_time = current_time
                blink_detected = True
            elif current_time - blink_start_time >= BLINK_THRESHOLD:
                # Decrease focus score by 20% if eyes are closed for 5 seconds
                if last_focus_decrease_time is None or current_time - last_focus_decrease_time >= 0.1:
                    focus_score = max(0, focus_score - DISCOUNT_EYES)
                    blink_start_time = current_time
        else:
            if blink_detected:
                blink_detected = False

        # Display information on frame
        cv.putText(frame, f"Face: {face_position}", (50, 50), FONTS, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Eyes: {eye_direction_text}", (50, 100), FONTS, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(frame, f"Focus Score: {int(focus_score)}%", (50, 150), FONTS, 1, (0, 0, 255), 2, cv.LINE_AA)

    else:
        # If no face is detected, decrease focus score by 1% every 1 second
        if last_focus_decrease_time is None or current_time - last_focus_decrease_time >= 0.1:
            focus_score = max(0, focus_score - 0.5)
            last_focus_decrease_time = current_time

    return (frame, focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected, eye_direction_text, face_position, last_focus_increase_time, last_focus_decrease_time)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    global focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected
    global last_focus_increase_time, last_focus_decrease_time

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        (img, focus_score, last_look_centered_time, not_looking_start_time, 
         blink_start_time, blink_detected, _, _, last_focus_increase_time, last_focus_decrease_time) = process_frame(
            img, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, 
            blink_start_time, blink_detected, last_focus_increase_time, last_focus_decrease_time
        )

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def process_uploaded_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(video_file.read())
    
    cap = cv.VideoCapture(tfile.name)
    
    global focus_score, last_look_centered_time, not_looking_start_time, blink_start_time, blink_detected
    # Initialize the new variables
    last_focus_increase_time = None
    last_focus_decrease_time = None

    focus_score = 50  # Initialized start focus from 50
    last_look_centered_time = None
    not_looking_start_time = None
    blink_start_time = None
    blink_detected = False
    
    data = []
    start_time = None
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            (frame, focus_score, last_look_centered_time, not_looking_start_time, 
             blink_start_time, blink_detected, eye_direction, face_position, last_focus_increase_time, last_focus_decrease_time) = process_frame(
                frame, face_mesh, focus_score, last_look_centered_time, not_looking_start_time, 
                blink_start_time, blink_detected, last_focus_increase_time, last_focus_decrease_time
            )
            
            timestamp = cap.get(cv.CAP_PROP_POS_MSEC) / 1000  # Convert to seconds
            
            if start_time is None:
                start_time = timestamp

            data.append({
                'timestamp': timestamp,
                'focus_score': focus_score,
                'eye_direction': eye_direction,
                'face_position': face_position,
                'is_front_camera': face_position != "Not Detected"
            })
    
    cap.release()
    df = pd.DataFrame(data)
    df['timestamp_min'] = (df['timestamp'] - start_time) / 60  # Convert to minutes

    # Calculate delta_time
    df['delta_time'] = df['timestamp'].diff().fillna(0)

    # Calculate Front Camera and Not Front Camera Time
    total_front_time = df[df['is_front_camera']]['delta_time'].sum()
    total_not_front_time = df[~df['is_front_camera']]['delta_time'].sum()
    total_time = total_front_time + total_not_front_time

    df.attrs['front_camera_percentage'] = (total_front_time / total_time) * 100 if total_time > 0 else 0
    df.attrs['not_front_camera_percentage'] = (total_not_front_time / total_time) * 100 if total_time > 0 else 0

    return df


import plotly.graph_objects as go

def create_dashboard(df, avg_focus_score_before_quiz, avg_focus_score_after_quiz):
    # Focus Score Trend
    fig_focus = go.Figure()
    fig_focus.add_trace(go.Scatter(x=df['timestamp_min'], y=df['focus_score'], mode='lines+markers', name='Score'))
    
    fig_focus.update_layout(
        xaxis_title='Time (minutes)',
        yaxis_title='Focus Score',
        yaxis_range=[0, 100],
        showlegend=True,
        xaxis=dict(
            tickmode='auto',  # Automatically decide where ticks are placed
            nticks=10,        # Set a reasonable number of ticks
            tickformat=".2f"  # Format the x-axis values to 4 decimal places
        )
    )
    
    st.plotly_chart(fig_focus)

    # Display Average Focus Scores
    st.markdown(
        f"""
        <div style="text-align: center; border: 1px solid #ddd; padding: 5px; border-radius: 5px; margin-bottom: 15px;">
            <h4 style="margin: 0; font-weight: bold;">Average Focus Score Before Quiz</h4>
            <h6 style="margin: 0; font-size: 16px;">{avg_focus_score_before_quiz:.2f}%</h6>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div style="text-align: center; border: 1px solid #ddd; padding: 5px; border-radius: 5px; margin-top: 15px;">
            <h4 style="margin: 0; font-weight: bold;">Average Focus Score After Quiz</h4>
            <h6 style="margin: 0; font-size: 16px;">{avg_focus_score_after_quiz:.2f}%</h6>
        </div>
        """, 
        unsafe_allow_html=True
    )

    return avg_focus_score_after_quiz  # Return the final average focus score


def export_to_pdf(df, avg_focus_score_before_quiz, avg_focus_score_after_quiz, quiz_score=None):
    buffer = BytesIO()
    sns.set_style("whitegrid")  # Set seaborn style to include grids
    with PdfPages(buffer) as pdf:
        # Focus Score Over Time
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='timestamp_min', y='focus_score', data=df)
        plt.title('Focus Score Over Time')
        plt.xlabel('Time (minutes)')
        plt.ylabel('Focus Score')
        plt.ylim(0, 100)
        pdf.savefig()
        plt.close()

        # Eye Direction Distribution as Pie Chart excluding 'Not Detected', 'Unknown', and 'Blink'
        eye_detected_df = df[~df['eye_direction'].isin(['Not Detected', 'Unknown', 'Blink'])]
        total_eye_detected_time = eye_detected_df['delta_time'].sum()
        eye_direction_times = eye_detected_df.groupby('eye_direction')['delta_time'].sum()
        eye_direction_percentages = (eye_direction_times / total_eye_detected_time) * 100

        plt.figure(figsize=(8, 6))
        plt.pie(
            eye_direction_times.values,
            labels=eye_direction_times.index,
            autopct=lambda pct: f'{pct:.1f}%'
        )
        plt.title('Eye Direction Distribution')
        pdf.savefig()
        plt.close()

        # Face Position Distribution as Pie Chart excluding 'Not Detected'
        face_detected_df = df[df['face_position'] != 'Not Detected']
        total_face_detected_time = face_detected_df['delta_time'].sum()
        face_position_times = face_detected_df.groupby('face_position')['delta_time'].sum()
        face_position_percentages = (face_position_times / total_face_detected_time) * 100

        plt.figure(figsize=(8, 6))
        plt.pie(
            face_position_times.values,
            labels=face_position_times.index,
            autopct=lambda pct: f'{pct:.1f}%'
        )
        plt.title('Face Position Distribution')
        pdf.savefig()
        plt.close()

        # Front Camera vs Not Front Camera Time as Pie Chart
        front_camera_time = df[df['is_front_camera']]['delta_time'].sum()
        not_front_camera_time = df[~df['is_front_camera']]['delta_time'].sum()
        total_time = front_camera_time + not_front_camera_time

        plt.figure(figsize=(8, 6))
        plt.pie(
            [front_camera_time, not_front_camera_time],
            labels=['Front Camera', 'Not Front Camera'],
            autopct=lambda pct: f'{pct:.1f}%'
        )
        plt.title('Front Camera vs Not Front Camera Time')
        pdf.savefig()
        plt.close()

        # Focus Statistics Table
        plt.figure(figsize=(8, 4))
        plt.axis('off')
        max_score = df['focus_score'].max()
        min_score = df['focus_score'].min()
        table_data = [
            ['Max Focus Score', f"{max_score:.2f}%"],
            ['Min Focus Score', f"{min_score:.2f}%"],
            ['Average Focus Score Before Quiz', f"{avg_focus_score_before_quiz:.2f}%"],
            ['Average Focus Score After Quiz', f"{avg_focus_score_after_quiz:.2f}%"],
            ['Quiz Score', f"{quiz_score:.2f}/10" if quiz_score is not None else "N/A"]
        ]
        table = plt.table(cellText=table_data, colLabels=None, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        plt.title('Focus Statistics')
        pdf.savefig()
        plt.close()

    buffer.seek(0)
    return buffer

# HTML Generation functions
def replace_placeholders(script_content: str, placeholder_mapping: dict) -> str:
    """Replaces the placeholders in the script content with actual values."""
    for placeholder, value in placeholder_mapping.items():
        script_content = script_content.replace(placeholder, str(value))

    return script_content

def generate_html_from_template(template_path: str, chart_data: dict, scores: dict, table_data: list) -> str:
    """Dynamically fills the HTML template with data and returns the populated HTML"""
    # Read the HTML template
    with open(template_path, "r", encoding="utf-8") as file:
        soup = BeautifulSoup(file, "html.parser")

    # Find the script tag containing the placeholders
    script_tag = soup.find("script", text=lambda x: x and "__LABELS_LINECHART__" in x)

    if script_tag and script_tag.string:
        # Create mapping of placeholders to actual values for each chart
        placeholder_mapping = {
            "__LABELS_LINECHART__": chart_data["line_chart"]["labels"],
            "__DATA_LINECHART__": chart_data["line_chart"]["data"],
            "__LABELS_PIECHART1__": chart_data["pie_chart1"]["labels"],
            "__DATA_PIECHART1__": chart_data["pie_chart1"]["data"],
            "__LABELS_PIECHART2__": chart_data["pie_chart2"]["labels"],
            "__DATA_PIECHART2__": chart_data["pie_chart2"]["data"],
            "__LABELS_PIECHART3__": chart_data["pie_chart3"]["labels"],
            "__DATA_PIECHART3__": chart_data["pie_chart3"]["data"],
        }

        # Replace placeholders in the script content
        updated_script_content = replace_placeholders(script_tag.string, placeholder_mapping)
        script_tag.string = updated_script_content

    # Update score cards dynamically
    score_cards = {
        "Final Score": scores["final"],
        "Highest Continuous Focus Score": scores["highest_continuous"],
        "Min. score": scores["min"],
        "Score after quiz": scores["after_quiz"],
    }
    for card_title, score_value in score_cards.items():
        card = soup.find("h5", text=card_title)
        if card:
            score_span = card.find_next("span", {"class": "h2 font-weight-bold mb-0"})
            if score_span:
                score_span.string = str(score_value)

    # Dynamically fill the table with quiz data
    table_body = soup.find("tbody")
    if table_body:
        table_body.clear()  # Remove existing rows
        for row in table_data:
            new_row = soup.new_tag("tr")

            question_cell = soup.new_tag("td")
            question_cell.string = row["question"]
            new_row.append(question_cell)

            status_cell = soup.new_tag("td")
            status_cell.string = row["status"]
            new_row.append(status_cell)

            score_cell = soup.new_tag("td")
            score_cell.string = row["score"]
            new_row.append(score_cell)

            table_body.append(new_row)

    return str(soup)

# =========================
# Helper Functions for Video Quiz Generator
# =========================

def generate_quiz_from_text(text):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Generate a multiple choice quiz as JSON with this exact structure: 
                 {"questions": [{"question": "Q1?", "options": ["A", "B", "C"], "correct_answer": "A"}]}.
                 Make questions short, general,simple, clear, easy, and focus on main points and key takeaways from the content.
                 Ensure questions are straightforward and test understanding rather than specific details.
                 Include some questions about the overall theme or main message."""},
                {"role": "user", "content": f"Create a 5-question quiz covering the main points and general understanding of this text:\n\n{text}"}
            ]
        )
        quiz_data = json.loads(response.choices[0].message['content'])
        if not isinstance(quiz_data, dict) or 'questions' not in quiz_data:
            raise ValueError("Invalid quiz format")
        return quiz_data
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def process_video_to_text(video_file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_video_path = os.path.join(temp_dir, video_file.name)
    audio_file_path = os.path.join(temp_dir, 'output_audio.mp3')
    
    try:
        with open(temp_video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        command = ["ffmpeg", "-i", temp_video_path, "-vn", "-acodec", "libmp3lame", audio_file_path]
        subprocess.run(command, check=True)
        
        result = model.transcribe(audio_file_path)
        return result['text']
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)

def display_quiz(quiz):
    st.subheader("Quiz")
    user_answers = []
    
    for i, question in enumerate(quiz['questions']):
        st.write(f"Question {i+1}: {question['question']}")  # Removed \n
        user_answer = st.radio(
            "##",  # Using "##" instead of empty string to minimize spacing
            question['options'],
            key=f"q{i}",
            label_visibility="collapsed"  # Hides the label completely
        )
        st.write("---")

        user_answers.append(user_answer)
    return user_answers

def calculate_score(quiz, user_answers):
    score = sum(1 for ua, q in zip(user_answers, quiz['questions'])
                if ua == q['correct_answer'])
    
    st.write("\n### Quiz Results")
    for i, (user_answer, question) in enumerate(zip(user_answers, quiz['questions']), 1):
        is_correct = user_answer == question['correct_answer']
        result_color = "green" if is_correct else "red"
        st.markdown(f"**Question {i}:** :{result_color}[{'✓' if is_correct else '✗'}]")
        st.write(f"Your answer: {user_answer}")
        st.write(f"Correct answer: {question['correct_answer']}")
        st.write("---")
    
    final_score = (score / len(quiz['questions'])) * 10
    st.success(f"Final score: {final_score:.2f}/10")
    return final_score, score

def adjust_focus_score_based_on_quiz(quiz, user_answers):
    global focus_score
    adjustment = 0
    for ua, q in zip(user_answers, quiz['questions']):
        if ua == q['correct_answer']:
            focus_score = min(100, focus_score + 5)
            adjustment += 5
        else:
            focus_score = max(0, focus_score - 5)
            adjustment -= 5
    return adjustment

# =========================
# Main Application
# =========================

def app():
    st.title("📊 Focus Detection with Integrated Quiz and Report")

    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    global CENTER_THRESHOLD, SIDE_THRESHOLD, BLINK_THRESHOLD, DISCOUNT_SIDE, DISCOUNT_EYES

    SIDE_THRESHOLD = st.sidebar.slider("Side Look Threshold (seconds)", 1, 50, 5, key="side_threshold")
    DISCOUNT_SIDE = st.sidebar.slider("Side Look Discount (%)", 1, 50, 5, key="discount_side")
    BLINK_THRESHOLD = st.sidebar.slider("Blink Threshold (seconds)", 1, 50, 5, key="blink_threshold")
    DISCOUNT_EYES = st.sidebar.slider("Closed Eyes Discount (%)", 5, 50, 20, key="discount_eyes")

    # Initialize session state variables
    if 'quiz_generated' not in st.session_state:
        st.session_state.quiz_generated = False
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'quiz' not in st.session_state:
        st.session_state.quiz = None
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = None
    if 'focus_score' not in st.session_state:
        st.session_state.focus_score = 50  # Initialized start focus from 50
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'avg_focus_score_before_quiz' not in st.session_state:
        st.session_state.avg_focus_score_before_quiz = None
    if 'avg_focus_score_after_quiz' not in st.session_state:
        st.session_state.avg_focus_score_after_quiz = None
    if 'quiz_score' not in st.session_state:
        st.session_state.quiz_score = None
    if 'uploaded_filename' not in st.session_state:
        st.session_state.uploaded_filename = None

    # Ask for OpenAI API key
    global openai_api_key
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    st.session_state.openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    openai_api_key = st.session_state.openai_api_key

    if openai_api_key:
        openai.api_key = openai_api_key
        # Load the Whisper model if not already loaded
        global model
        if model is None:
            model = whisper.load_model("base")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

    # Tabs for Live Video and Upload Video
    tab1, tab2 = st.tabs(["🎥 Live Video", "📤 Upload Video"])

    with tab1:
        st.header("🔴 Webcam Feed")
        st.write(f"Current Focus Score: {st.session_state.focus_score}%")
        webrtc_streamer(
            key="camera",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            video_frame_callback=video_frame_callback,
        )

    with tab2:
        st.header("📥 Upload Video for Analysis and Quiz")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

        if uploaded_file is not None:
            if st.session_state.uploaded_filename != uploaded_file.name:
                # New file uploaded, reset stored data
                st.session_state.results_df = None
                st.session_state.avg_focus_score_before_quiz = None
                st.session_state.avg_focus_score_after_quiz = None
                st.session_state.quiz = None
                st.session_state.quiz_generated = False
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = None
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.focus_score = 50  # Reset focus score to 50
            st.video(uploaded_file)

            if st.button("🔍 Analyze Video and Generate Quiz"):
             
                # Process video only if not already processed
                if st.session_state.results_df is None:
                    with st.spinner("Analyzing video..."):
                        results_df = process_uploaded_video(uploaded_file)
                        st.session_state.results_df = results_df  # Store in session state
                        st.success("✅ Analysis complete!")
                        st.session_state.avg_focus_score_before_quiz = results_df['focus_score'].mean()
                else:
                    st.success("Video already analyzed.")
                    results_df = st.session_state.results_df
                    st.session_state.avg_focus_score_before_quiz = results_df['focus_score'].mean()

                # Generate quiz only if not already generated
                if st.session_state.quiz is None:
                    with st.spinner("Processing video for quiz generation..."):
                        transcription = process_video_to_text(uploaded_file)
                        if transcription:
                            with st.spinner("Generating quiz..."):
                                st.session_state.quiz = generate_quiz_from_text(transcription)
                                if st.session_state.quiz:
                                    st.session_state.quiz_generated = True
                else:
                    st.success("Quiz already generated.")
                    st.session_state.quiz_generated = True

        if st.session_state.quiz_generated and not st.session_state.quiz_submitted:
            st.session_state.user_answers = display_quiz(st.session_state.quiz)

            if st.button("Submit Quiz"):
                final_score, correct_answers = calculate_score(st.session_state.quiz, st.session_state.user_answers)
                adjustment = adjust_focus_score_based_on_quiz(st.session_state.quiz, st.session_state.user_answers)
                st.session_state.quiz_submitted = True
                st.session_state.focus_score = focus_score  # Update session focus score
                st.session_state.quiz_score = final_score  # Store quiz score

                # Calculate average focus score after quiz
                avg_focus_score_before_quiz = st.session_state.avg_focus_score_before_quiz
                total_adjustment = adjustment  # Total adjustment based on quiz
                avg_focus_score_after_quiz = min(100, max(0, avg_focus_score_before_quiz + total_adjustment))
                st.session_state.avg_focus_score_after_quiz = avg_focus_score_after_quiz

                # Show detailed dashboard
                create_dashboard(st.session_state.results_df, avg_focus_score_before_quiz, avg_focus_score_after_quiz)

                # Export to PDF
                if st.session_state.results_df is not None:
                    pdf_file = export_to_pdf(
                        st.session_state.results_df,
                        avg_focus_score_before_quiz,
                        avg_focus_score_after_quiz,
                        final_score
                    )
                    
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_file,
                        file_name="focus_analysis_report.pdf",
                        mime="application/pdf"
                    )

        if st.button("Generate HTML Report"):
            if st.session_state.results_df is not None:
                chart_data = {
                    "line_chart": {
                        "labels": st.session_state.results_df['timestamp_min'].tolist(),
                        "data": st.session_state.results_df['focus_score'].tolist(),
                    },
                    "pie_chart1": {"labels": ["Left", "Right", "Up", "Down", "Forward"], "data": [10, 20, 30, 25, 15]},
                    "pie_chart2": {"labels": ["Forward", "Not Forward"], "data": [70, 30]},
                    "pie_chart3": {"labels": ["Person Detected", "Not Detected"], "data": [80, 20]},
                }

                scores = {
                    "final": int(st.session_state.avg_focus_score_after_quiz or 0),
                    "highest_continuous": int(st.session_state.results_df['focus_score'].max()),
                    "min": int(st.session_state.results_df['focus_score'].min()),
                    "after_quiz": int(st.session_state.avg_focus_score_after_quiz or 0),
                }

                table_data = [
                    {"question": "What is AI?", "status": "Correct", "score": "100%"},
                    {"question": "What is ML?", "status": "Correct", "score": "100%"},
                ]

                # Path to the HTML template file
                template_path = "index.html"

                # Generate populated HTML
                populated_html = generate_html_from_template(template_path, chart_data, scores, table_data)

                # Create a download button for the populated HTML
                with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                    tmp_file.write(populated_html.encode("utf-8"))
                    tmp_file_path = tmp_file.name

                st.download_button(
                    label="Download HTML Report",
                    data=open(tmp_file_path, "rb").read(),
                    file_name="populated_template.html",
                    mime="text/html",
                )
            else:
                st.error("No analysis data available. Please analyze the video first.")

    st.sidebar.write(f"Focus Score: {st.session_state.focus_score}%")


if __name__ == "__main__":
    app()