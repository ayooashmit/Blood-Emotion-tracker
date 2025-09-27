import cv2
from deepface import DeepFace
import numpy as np
import os
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Google Sheets API setup
def setup_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", 
             "https://www.googleapis.com/auth/drive",
             ]
    
    #  download the credentials JSON file from Google Cloud Console
    creds = ServiceAccountCredentials.from_json_keyfile_name("blood-detection-logger-9275e2f40cb4.json", scope)
    client = gspread.authorize(creds)
    
    # Open the spreadsheet (replace with your spreadsheet name)
    sheet = client.open("Blood_Detection_Logs").sheet1
    return sheet

def log_blood_detection(sheet, timestamp, emotion):
    try:
        sheet.append_row([timestamp, "Blood Detected", emotion])
        print("Logged to Google Sheets")
    except Exception as e:
        print("Failed to log to Google Sheets:", e)

def detect_emotion(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if not result or not isinstance(result, list):
            return None
        dominant_emotion = result[0]['dominant_emotion']
        return dominant_emotion
    except Exception as e:
        print("Error:", e)
        return None

def detect_red_areas(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Strict red filter (blood-like)
    lower_deep_red1 = np.array([0, 180, 100])
    upper_deep_red1 = np.array([5, 255, 255])
    lower_deep_red2 = np.array([175, 180, 100])
    upper_deep_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_deep_red1, upper_deep_red1)
    mask2 = cv2.inRange(hsv, lower_deep_red2, upper_deep_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.GaussianBlur(red_mask, (7, 7), 0)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours if contours else []

def main():
    # Initialize Google Sheets connection
    try:
        sheet = setup_google_sheets()
        print("Connected to Google Sheets")
    except Exception as e:
        print("Google Sheets connection failed:", e)
        sheet = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Starting real-time facial emotion & red color detection with alert...")

    red_alert_triggered = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = detect_emotion(frame)

        red_contours = detect_red_areas(frame)
        red_detected = False

        for contour in red_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # filter out small red spots
                red_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, 'Blood detected', (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if red_detected and not red_alert_triggered:
            red_alert_triggered = True
            os.system("say Blood detected admitting to hospital")  # MacOS voice alert
            
            # Log to Google Sheets if available
            if sheet:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_blood_detection(sheet, timestamp, emotion if emotion else "Unknown")
                
        elif not red_detected:
            red_alert_triggered = False  # Reset if no red in frame

        if emotion:
            cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # Show "Face not visible" in center of the screen
            h, w, _ = frame.shape
            text = "Admitted to hospital"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = int((w - text_size[0]) / 2)
            text_y = int(h - 50)
            cv2.putText(frame, text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow('Facial Emotion & Red Color Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()