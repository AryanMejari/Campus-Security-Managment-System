import cv2
import numpy as np
import json
import time
from datetime import datetime
import os

# Load face database
FACE_DB_FILE = "face_database.json"
if os.path.exists(FACE_DB_FILE):
    with open(FACE_DB_FILE, 'r') as f:
        face_database = json.load(f)
else:
    print("⚠️  WARNING: No face database found. Please enroll students first.")
    face_database = {"labels": [], "faces": {}}

# Load multiple Haar Cascade models for better face detection
face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
]

def detect_faces_robust(image):
    """Enhanced face detection with multiple cascade classifiers"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    all_faces = []
    
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            all_faces.extend(faces)
    
    if len(all_faces) == 0:
        # Try with more lenient parameters
        faces = face_cascades[0].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces = faces
    
    # Return unique faces (largest area)
    if len(all_faces) > 0:
        # Sort by area
        all_faces = sorted(all_faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        return [all_faces[0]]  # Return largest face
    
    return []

def preprocess_face(face_img):
    """Preprocess face for comparison"""
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Resize
    gray = cv2.resize(gray, (100, 100))
    
    # Equalize histogram
    gray = cv2.equalizeHist(gray)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    return gray

def compare_faces(face1, face2):
    """Simple face comparison using histogram correlation"""
    # Preprocess both faces
    face1_proc = preprocess_face(face1)
    face2_proc = preprocess_face(face2)
    
    # Calculate histograms
    hist1 = cv2.calcHist([face1_proc], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([face2_proc], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare using correlation
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return correlation

def load_reference_face(roll):
    """Load reference face for a given roll number"""
    face_path = os.path.join("faces", f"{roll}_cropped.jpg")
    if os.path.exists(face_path):
        face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
        return face
    return None

# Open laptop camera
cap = cv2.VideoCapture(0)

# Set camera properties for better face detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 50)

# QR code detector
detector = cv2.QRCodeDetector()

print("=" * 60)
print("📷 CAMPUS SENTINEL - QR & FACE VERIFICATION SYSTEM")
print("=" * 60)
print("INSTRUCTIONS:")
print("1. Show student's digital ID QR code to camera")
print("2. Position your face clearly in the center")
print("3. Ensure good lighting on your face")
print("4. Press 'Q' to exit")
print("=" * 60)

# Variables
last_qr_scan_time = 0
current_student_info = None
face_verified = False
verification_confidence = 0
face_detected = False

# Face similarity threshold
SIMILARITY_THRESHOLD = 0.6  # 60% similarity

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️  Failed to capture frame")
        break

    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    
    # ---------- QR CODE DETECTION ----------
    current_time = time.time()
    if current_time - last_qr_scan_time > 2:  # Scan QR every 2 seconds
        data, bbox, _ = detector.detectAndDecode(frame)
        
        if data:
            print("\n" + "=" * 60)
            print("✅ QR CODE DETECTED!")
            print(f"📊 Raw Data: {data}")
            
            # Parse student information
            info = {}
            try:
                for item in data.split(";"):
                    if ":" in item:
                        key, value = item.split(":", 1)
                        info[key] = value
            except:
                print("⚠️  Could not parse QR data")
            
            print("\n🎓 STUDENT INFORMATION FROM QR CODE:")
            print(f"   Name       : {info.get('Name', 'N/A')}")
            print(f"   Roll No    : {info.get('Roll', 'N/A')}")
            print(f"   Department : {info.get('Dept', 'N/A')}")
            print(f"   Time       : {datetime.now().strftime('%H:%M:%S')}")
            
            current_student_info = info
            last_qr_scan_time = current_time
            face_verified = False
            face_detected = False
    
    # ---------- FACE DETECTION ----------
    faces = detect_faces_robust(frame)
    face_detected = len(faces) > 0
    
    # ---------- FACE VERIFICATION ----------
    if face_detected and current_student_info:
        roll_number = current_student_info.get('Roll')
        
        if roll_number and roll_number in face_database["faces"]:
            # Load reference face
            reference_face = load_reference_face(roll_number)
            
            if reference_face is not None:
                # Get the detected face
                (x, y, w, h) = faces[0]
                detected_face = frame[y:y+h, x:x+w]
                
                # Compare faces
                similarity = compare_faces(detected_face, reference_face)
                verification_confidence = similarity * 100
                
                if similarity >= SIMILARITY_THRESHOLD:
                    face_verified = True
                    print(f"\n👤 FACE VERIFICATION: ✅ MATCH")
                    print(f"   Similarity: {verification_confidence:.1f}%")
                    print(f"   Student: {face_database['faces'][roll_number]['name']}")
                else:
                    face_verified = False
                    print(f"\n👤 FACE VERIFICATION: ❌ NO MATCH")
                    print(f"   Similarity: {verification_confidence:.1f}% (threshold: {SIMILARITY_THRESHOLD*100:.0f}%)")
    
    # ---------- DISPLAY INFORMATION ----------
    # Draw green target box
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    box_size = 200
    
    cv2.rectangle(display_frame, 
                  (center_x - box_size//2, center_y - box_size//2),
                  (center_x + box_size//2, center_y + box_size//2),
                  (0, 255, 0), 2)
    
    # Draw crosshair
    cv2.line(display_frame, (center_x, center_y - 10), 
             (center_x, center_y + 10), (0, 255, 0), 2)
    cv2.line(display_frame, (center_x - 10, center_y), 
             (center_x + 10, center_y), (0, 255, 0), 2)
    
    # Draw face detection box
    for (x, y, w, h) in faces:
        if face_verified:
            color = (0, 255, 0)  # Green for verified
            thickness = 3
        else:
            color = (0, 255, 255)  # Yellow for detected
            thickness = 2
        
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, thickness)
        
        # Label
        if face_verified:
            label = f"VERIFIED ({verification_confidence:.1f}%)"
        else:
            label = "FACE DETECTED"
        
        cv2.putText(display_frame, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display header
    cv2.putText(display_frame, "CAMPUS SENTINEL - VERIFICATION SYSTEM", 
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, "CAMPUS SENTINEL - VERIFICATION SYSTEM", 
                (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 100, 255), 1)
    
    # Display QR code info if available
    if current_student_info:
        y_offset = 70
        
        # Background for QR info
        cv2.rectangle(display_frame, (10, 50), (500, 180), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 50), (500, 180), (0, 100, 255), 2)
        
        cv2.putText(display_frame, "QR CODE INFORMATION:", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        y_offset += 30
        
        cv2.putText(display_frame, f"Name: {current_student_info.get('Name', 'N/A')}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(display_frame, f"Roll No: {current_student_info.get('Roll', 'N/A')}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(display_frame, f"Department: {current_student_info.get('Dept', 'N/A')}", 
                    (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display verification status
    status_y = 200
    if current_student_info:
        if face_verified:
            status_text = "✅ FACE VERIFIED - ACCESS GRANTED"
            status_color = (0, 255, 0)
            confidence_text = f"Confidence: {verification_confidence:.1f}%"
            
            # Draw access granted animation
            cv2.putText(display_frame, "ACCESS GRANTED", 
                        (width//2 - 150, height//2 + 150), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 3)
            
        elif face_detected:
            status_text = "⚠️  FACE DETECTED - VERIFYING..."
            status_color = (0, 255, 255)
            confidence_text = "Checking against database..."
            
        else:
            status_text = "❌ NO FACE DETECTED"
            status_color = (0, 0, 255)
            confidence_text = "Position face in green box"
            
            # Draw instructions
            instructions = [
                "INSTRUCTIONS:",
                "1. Position face in green box",
                "2. Look directly at camera",
                "3. Ensure good lighting",
                "4. Remove glasses if needed"
            ]
            
            for i, line in enumerate(instructions):
                cv2.putText(display_frame, line, 
                           (width - 300, 100 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        status_text = "⏳ WAITING FOR QR CODE..."
        status_color = (255, 255, 0)
        confidence_text = "Show student ID QR code"
    
    # Draw status box
    cv2.rectangle(display_frame, (10, 190), (500, 240), (0, 0, 0), -1)
    cv2.rectangle(display_frame, (10, 190), (500, 240), status_color, 2)
    
    cv2.putText(display_frame, status_text, 
                (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    if 'confidence_text' in locals():
        cv2.putText(display_frame, confidence_text, 
                    (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display instructions
    cv2.putText(display_frame, "Press 'Q' to quit", 
                (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show camera feed
    cv2.imshow("Campus Sentinel - QR & Face Verification", display_frame)
    
    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n" + "=" * 60)
        print("👋 System shutdown")
        print("=" * 60)
        break
    elif key == ord('f'):
        # Force face detection test
        print("\n🔍 Forcing face detection test...")
        faces = detect_faces_robust(frame)
        print(f"Faces detected: {len(faces)}")

# Cleanup
cap.release()
cv2.destroyAllWindows()