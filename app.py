import streamlit as st
import cv2
import numpy as np
import math
import json
import pyttsx3
import threading
import time

def speak(text):
    """Handles text-to-speech without freezing the video."""
    def run_tts():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_tts).start()

def load_questions():
    """Loads questions from the JSON file."""
    with open('questions.json', 'r') as file:
        return json.load(file)

def count_fingers_math(roi):
    """
    Pure OpenCV math to count fingers without MediaPipe.
    It looks for skin color, draws a boundary, and counts the 'valleys' between fingers.
    """
    # Convert image to HSV (Hue, Saturation, Value) for better color detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define broad bounds for human skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create a mask that isolates the skin
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=4) # Thicken the mask
    mask = cv2.GaussianBlur(mask, (5,5), 100) # Blur to smooth edges
    
    # Find the outlines (contours) of the skin
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0 # No hand detected
        
    # Assume the largest contour is the hand
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    
    if cv2.contourArea(cnt) < 3000: # If the object is too small, ignore it
        return 0
        
    # Draw a rough mathematical polygon around the hand
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Find the Convex Hull (an invisible rubber band stretched around the hand)
    hull = cv2.convexHull(cnt)
    
    # Find Convexity Defects (the 'valleys' or gaps between fingers)
    hull_indices = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull_indices)
    
    finger_count = 0
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0]) # The deepest point of the valley
            
            # Triangle math to find the angle of the valley
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # Cosine rule to calculate the angle
            angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            
            # If the valley angle is sharp (<= 90 degrees), it's a gap between two fingers
            if angle <= 90:
                finger_count += 1
                cv2.circle(roi, far, 5, [0, 0, 255], -1) # Draw a red dot at the valley
                
        # If we found gaps, the number of fingers is gaps + 1
        if finger_count > 0:
            finger_count += 1
        else:
            # If area is large but 0 gaps, it's likely 1 finger pointing up
            finger_count = 1 
            
    return finger_count

def put_multiline_text(img, text, x, y, font, scale, color, thickness, max_width):
    """Wraps text so it doesn't go off the edge of the camera screen."""
    words = text.split(' ')
    lines = []
    current_line = ""
    
    for word in words:
        test_line = current_line + word + " "
        size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        if size[0] > max_width and current_line != "":
            lines.append(current_line)
            current_line = word + " "
        else:
            current_line = test_line
    lines.append(current_line)
    
    for line in lines:
        cv2.putText(img, line.strip(), (x, y), font, scale, color, thickness)
        y += int(cv2.getTextSize(line, font, scale, thickness)[0][1] * 2.0) + 10 
    return y 

# --- STREAMLIT UI SETUP ---
st.set_page_config(layout="wide") 
st.title("AR Gestural Interview Prep (OpenCV Math Edition)")
st.write("Place your hand inside the GREEN BOX to answer. 1-4 Fingers to Answer. 5 Fingers for Next Question.")

data = load_questions()
roles = list(data.keys())
selected_role = st.selectbox("Select a Role to start:", ["None"] + roles)

# --- SESSION STATE VARIABLES ---
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'audio_played' not in st.session_state:
    st.session_state.audio_played = False
if 'answered' not in st.session_state:
    st.session_state.answered = False
if 'feedback_msg' not in st.session_state:
    st.session_state.feedback_msg = "" 
if 'feedback_color' not in st.session_state:
    st.session_state.feedback_color = (255, 255, 255) 

# --- MAIN LOGIC ---
if selected_role != "None":
    questions_list = data[selected_role]
    
    if st.session_state.q_index < len(questions_list):
        current_q = questions_list[st.session_state.q_index]
        
        # Audio logic
        if not st.session_state.audio_played:
            text_to_read = current_q['question'] + " The options are. " + ". ".join(current_q['options'])
            speak(text_to_read)
            st.session_state.audio_played = True
            
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        stframe = st.empty()
        last_action_time = time.time()
        
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
                
            img = cv2.flip(img, 1) 
            img_h, img_w, _ = img.shape
            
            # Define the green "Drop Hand Here" box on the right side of the screen
            box_x1, box_y1 = img_w - 350, 100
            box_x2, box_y2 = img_w - 50, 450
            
            # Extract the Region of Interest (ROI) - only process the area inside the green box
            roi = img[box_y1:box_y2, box_x1:box_x2]
            
            # Count fingers using purely math
            fingers_up = count_fingers_math(roi)
            
            # Draw dark background for text readability
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (img_w, img_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img) 

            # Draw the Hand Scanning Box
            cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            cv2.putText(img, "Place Hand Here", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            # --- DRAW TEXT ON THE CAMERA FEED ---
            current_y = 50
            # Wrap text to avoid going into the green box area
            text_max_width = img_w - 400 
            
            current_y = put_multiline_text(img, f"Q: {current_q['question']}", 30, current_y, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, text_max_width)
            
            current_y += 20
            for opt in current_q['options']:
                current_y = put_multiline_text(img, opt, 30, current_y, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, text_max_width)
                
            if st.session_state.answered and st.session_state.feedback_msg:
                current_y += 30
                put_multiline_text(img, st.session_state.feedback_msg, 30, current_y, cv2.FONT_HERSHEY_SIMPLEX, 0.7, st.session_state.feedback_color, 2, text_max_width)

            cv2.putText(img, f"Fingers Detected: {fingers_up}", (30, img_h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Score: {st.session_state.score} / {len(questions_list)}", (30, img_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            
            stframe.image(img, channels="BGR")
            
            # --- GESTURE LOGIC ---
            current_time = time.time()
            if current_time - last_action_time > 3: # Cooldown timer
                
                if fingers_up in [1, 2, 3, 4] and not st.session_state.answered:
                    user_choice = fingers_up
                    st.session_state.answered = True
                    
                    if user_choice == current_q['answer']:
                        st.session_state.feedback_msg = f"CORRECT! {current_q['explanation']}"
                        st.session_state.feedback_color = (0, 255, 0)
                        speak(st.session_state.feedback_msg)
                        st.session_state.score += 1
                    else:
                        st.session_state.feedback_msg = f"WRONG! Correct was {current_q['answer']}. {current_q['explanation']}"
                        st.session_state.feedback_color = (0, 0, 255) 
                        speak(st.session_state.feedback_msg)
                        
                    last_action_time = time.time()
                    
                elif fingers_up >= 5: # 5 or more fingers (just in case the math catches an extra defect)
                    st.session_state.q_index += 1
                    st.session_state.audio_played = False
                    st.session_state.answered = False
                    st.session_state.feedback_msg = "" 
                    time.sleep(1)
                    st.rerun() 
                    
        cap.release()
        st.rerun()

    else:
        st.success(f"Quiz Complete! Your final score is {st.session_state.score} out of {len(questions_list)}")
        if st.button("Restart"):
            st.session_state.q_index = 0
            st.session_state.score = 0
            st.session_state.audio_played = False
            st.session_state.answered = False
            st.session_state.feedback_msg = ""
            st.rerun()