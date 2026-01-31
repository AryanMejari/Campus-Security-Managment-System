import os
import cv2
import pytesseract
import qrcode
import re
import numpy as np
import json
import base64
import requests
from flask import Flask, request, send_file, render_template, jsonify
from datetime import datetime
from models import db, Student, Incident

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///campus_sec.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
FACES_FOLDER = "faces"
QR_FOLDER = "qrcodes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)
os.makedirs(QR_FOLDER, exist_ok=True)

GEMINI_API_KEY = "AIzaSyBunDAFzp4uj1Z_YszfEMl-BPQsRmIqpHI"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# --- Helper Functions (from original scripts) ---
face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
]

def detect_faces(image):
    if image is None: return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    all_faces = []
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0: all_faces.extend(faces)
    if len(all_faces) > 0:
        all_faces = sorted(all_faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        return [all_faces[0]]
    return []

def preprocess_face(face_img):
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    gray = cv2.resize(gray, (100, 100))
    gray = cv2.equalizeHist(gray)
    return gray

def compare_faces(face1, face2):
    f1 = preprocess_face(face1)
    f2 = preprocess_face(face2)
    hist1 = cv2.calcHist([f1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([f2], [0], None, [256], [0, 256])
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats")
def get_stats():
    total_students = Student.query.count()
    total_incidents = Incident.query.count()
    total_threats = Incident.query.filter(Incident.threat_level == 'HIGH').count()
    resolved_incidents = Incident.query.filter(Incident.status == 'Resolved').count()
    
    return jsonify({
        "total_students": total_students,
        "total_incidents": total_incidents,
        "total_threats": total_threats,
        "resolved_incidents": resolved_incidents
    })

# --- Student Registration Routes ---
@app.route("/api/register/upload_id", methods=["POST"])
def upload_id():
    if 'id_card' not in request.files: return jsonify({"success": False, "error": "No file"})
    file = request.files["id_card"]
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    try:
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        
        # Simple extraction logic
        name = "Unknown"
        roll = "Unknown"
        dept = "Unknown"
        
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                if re.search(r"name", line, re.IGNORECASE): name = line.split(":")[-1].strip()
                elif re.search(r"roll|id", line, re.IGNORECASE): roll = line.split(":")[-1].strip()
                elif re.search(r"dept", line, re.IGNORECASE): dept = line.split(":")[-1].strip()
        
        return jsonify({"success": True, "name": name, "roll": roll, "dept": dept})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/register/complete", methods=["POST"])
def register_student():
    data = request.json
    name = data.get("name")
    roll = data.get("roll")
    dept = data.get("dept")
    face_data = data.get("face_image")
    
    if not all([name, roll, dept, face_data]):
        return jsonify({"success": False, "error": "Missing data"})
    
    # Save face image
    face_bytes = base64.b64decode(face_data.split(",")[1])
    nparr = np.frombuffer(face_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    faces = detect_faces(img)
    if not faces: return jsonify({"success": False, "error": "No face detected"})
    
    (x, y, w, h) = faces[0]
    face_cropped = img[y:y+h, x:x+w]
    face_path = os.path.join(FACES_FOLDER, f"{roll}.jpg")
    cv2.imwrite(face_path, face_cropped)
    
    # Generate QR
    qr_content = f"Name:{name};Roll:{roll};Dept:{dept}"
    qr = qrcode.make(qr_content)
    qr_path = os.path.join(QR_FOLDER, f"{roll}.png")
    qr.save(qr_path)
    
    # Save to DB
    student = Student(name=name, roll_no=roll, department=dept, face_path=face_path, qr_path=qr_path)
    db.session.add(student)
    db.session.commit()
    
    return jsonify({"success": True, "qr_path": f"/get_qr/{roll}"})

@app.route("/get_qr/<roll>")
def get_qr(roll):
    return send_file(os.path.join(QR_FOLDER, f"{roll}.png"), mimetype='image/png')

@app.route("/api/verify/check_frame", methods=["POST"])
def verify_frame():
    data = request.get_json(silent=True) or {}
    image_data = data.get("image")
    if not image_data: return jsonify({"success": False, "error": "No image"})
    
    # Decode image
    try:
        img_bytes = base64.b64decode(image_data.split(",")[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"success": False, "error": f"Image decode failed: {str(e)}"})
    
    # Detect QR
    detector = cv2.QRCodeDetector()
    qr_data, _, _ = detector.detectAndDecode(img)
    
    if not qr_data:
        return jsonify({"success": False, "error": "No QR code detected in frame"})
    
    # Parse QR
    info = {}
    try:
        for item in qr_data.split(";"):
            if ":" in item:
                k, v = item.split(":", 1)
                info[k] = v
    except:
        return jsonify({"success": False, "error": "Invalid QR format"})
    
    roll = info.get("Roll")
    student = Student.query.filter_by(roll_no=roll).first()
    if not student: return jsonify({"success": False, "error": f"Student with Roll {roll} not found in database"})
    
    # Detect Face in same frame
    faces = detect_faces(img)
    if not faces: return jsonify({"success": False, "error": "No face detected in frame"})
    
    (x, y, w, h) = faces[0]
    captured_face_img = img[y:y+h, x:x+w]
    
    ref_face_img = cv2.imread(student.face_path)
    if ref_face_img is None: return jsonify({"success": False, "error": "Reference face not found"})
    
    similarity = compare_faces(captured_face_img, ref_face_img)
    
    if similarity > 0.5:
        return jsonify({"success": True, "student": student.to_dict(), "similarity": float(similarity)})
    else:
        return jsonify({"success": False, "error": "Face match failed", "similarity": float(similarity)})

# --- Threat Analysis Routes ---
@app.route("/api/threat/analyze", methods=["POST"])
def analyze_threat():
    data = request.get_json(silent=True) or {}
    
    if 'image' in request.files:
        file = request.files["image"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
    elif 'image' in data:
        # Assume base64
        face_data = data.get("image")
        face_bytes = base64.b64decode(face_data.split(",")[1])
        filename = f"incident_{int(datetime.now().timestamp())}.jpg"
        path = os.path.join(UPLOAD_FOLDER, filename)
        with open(path, "wb") as f: f.write(face_bytes)
    else:
        return jsonify({"success": False, "error": "No image source provided"})

    location = request.form.get("location") or data.get("location") or "Unknown"
    time_str = request.form.get("time") or data.get("time") or datetime.now().strftime("%H:%M")
    
    try:
        with open(path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        
        prompt = "Analyze the incident image. Return strictly in this format:\nThreat Confidence: <number>\nThreat Level: <LOW/MEDIUM/HIGH>\nIncident Summary: <text>"
        
        payload = {
            "contents": [{"parts": [{"text": f"Context: Location={location}, Time={time_str}\n\n{prompt}"}, {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}]}]
        }
        
        response = requests.post(GEMINI_URL, headers={"Content-Type": "application/json"}, json=payload)
        
        if response.status_code != 200:
            return jsonify({"success": False, "error": f"Gemini API Error: {response.text}"})
            
        result_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        
        # Parse result
        conf = re.search(r"Threat Confidence:\s*(\d+)", result_text)
        level = re.search(r"Threat Level:\s*(\w+)", result_text)
        summary = re.search(r"Incident Summary:\s*(.*)", result_text, re.DOTALL)
        
        threat_score = int(conf.group(1)) if conf else 50
        threat_level = level.group(1).upper() if level else "MEDIUM"
        summ = summary.group(1).strip() if summary else "Analysis complete."
        
        incident = Incident(image_path=path, location=location, time=time_str, threat_score=threat_score, threat_level=threat_level, summary=summ)
        db.session.add(incident)
        db.session.commit()
        
        return jsonify({"success": True, "incident": incident.to_dict()})
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"success": False, "error": f"Server Error: {str(e)}"})

# --- Alert Management & Database ---
@app.route("/api/incidents")
def get_incidents():
    incidents = Incident.query.order_by(Incident.created_at.desc()).all()
    return jsonify([i.to_dict() for i in incidents])

@app.route("/api/incidents/update", methods=["POST"])
def update_incident():
    data = request.json
    incident = Incident.query.get(data.get("id"))
    if incident:
        incident.status = data.get("status")
        db.session.commit()
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Not found"})

@app.route("/api/students")
def get_students():
    students = Student.query.all()
    return jsonify([s.to_dict() for s in students])

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)
