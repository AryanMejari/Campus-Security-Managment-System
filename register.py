import os
import cv2
import pytesseract
import qrcode
import re
import numpy as np
import json
import base64
from flask import Flask, request, send_file, render_template_string, jsonify
from io import BytesIO

# Uncomment if Tesseract is not in PATH (Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
FACES_FOLDER = "faces"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(FACES_FOLDER, exist_ok=True)

# Load multiple Haar Cascade models for better face detection
face_cascades = [
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt_tree.xml'),
    cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
]

# Face database
FACE_DB_FILE = "face_database.json"
if os.path.exists(FACE_DB_FILE):
    with open(FACE_DB_FILE, 'r') as f:
        face_database = json.load(f)
else:
    face_database = {"labels": [], "faces": {}}

def detect_faces_robust(image):
    """Detect faces using multiple cascade classifiers for better accuracy"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    all_faces = []
    
    for cascade in face_cascades:
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Reduced from 1.1 for better detection
            minNeighbors=3,    # Reduced for more sensitive detection
            minSize=(80, 80),  # Reduced minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            all_faces.extend(faces)
    
    if len(all_faces) == 0:
        # Try with more lenient parameters
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascades[0].detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,    # Even more sensitive
            minSize=(50, 50),  # Smaller minimum size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        all_faces = faces
    
    # Merge overlapping faces
    if len(all_faces) > 0:
        # Sort by area (largest first)
        all_faces = sorted(all_faces, key=lambda rect: rect[2] * rect[3], reverse=True)
        
        # Return the largest face
        return [all_faces[0]]
    
    return []

# ---------------- ENHANCED UI WITH BETTER FEEDBACK ----------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Campus Sentinel - Student Enrollment</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .step { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }
        .step h3 { margin-top: 0; color: #333; }
        #video { width: 100%; max-width: 640px; border: 2px solid #333; transform: scaleX(-1); }
        #capturedFace { width: 200px; height: 200px; border: 2px solid #333; }
        .button { 
            background: #4CAF50; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 10px 5px;
        }
        .button:hover { background: #45a049; }
        .button:disabled { background: #cccccc; cursor: not-allowed; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
        .hidden { display: none; }
        .face-frame { position: relative; width: 640px; height: 480px; margin: 10px 0; }
        .face-overlay { 
            position: absolute; 
            top: 50%; 
            left: 50%; 
            transform: translate(-50%, -50%);
            width: 250px; 
            height: 300px; 
            border: 3px solid #4CAF50;
            border-radius: 10px;
            background: rgba(76, 175, 80, 0.1);
        }
        .face-instruction { 
            position: absolute;
            bottom: 10px;
            left: 0;
            right: 0;
            text-align: center;
            color: white;
            background: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
        }
        .detection-box {
            position: absolute;
            border: 2px solid red;
            background: rgba(255, 0, 0, 0.1);
        }
    </style>
</head>
<body>
    <h1>📱 Campus Sentinel - Student Enrollment</h1>
    
    <div class="step">
        <h3>Step 1: Upload Student ID Card</h3>
        <form id="uploadForm">
            <input type="file" id="idCard" accept="image/*" required>
            <button type="button" onclick="uploadID()" class="button">Upload ID Card</button>
        </form>
        <div id="uploadStatus"></div>
    </div>
    
    <div class="step hidden" id="faceCaptureStep">
        <h3>Step 2: Capture Student's Face</h3>
        <p>Make sure:</p>
        <ul>
            <li>Face is well lit</li>
            <li>Look directly at camera</li>
            <li>Remove glasses if possible</li>
            <li>Keep face within green box</li>
        </ul>
        
        <div class="face-frame">
            <video id="video" autoplay></video>
            <div class="face-overlay"></div>
            <div class="face-instruction">Position your face in the green box</div>
            <canvas id="canvas" class="hidden"></canvas>
        </div>
        
        <div>
            <button onclick="startCamera()" class="button" id="startBtn">Start Camera</button>
            <button onclick="captureFace()" class="button" id="captureBtn" disabled>Capture Face</button>
            <button onclick="stopCamera()" class="button" id="stopBtn" disabled>Stop Camera</button>
            <button onclick="testFaceDetection()" class="button">Test Face Detection</button>
        </div>
        
        <div>
            <h4>Captured Face:</h4>
            <img id="capturedFace" src="" alt="No face captured">
            <div id="facePreview"></div>
        </div>
        <div id="faceStatus"></div>
    </div>
    
    <div class="step hidden" id="enrollmentStep">
        <h3>Step 3: Complete Enrollment</h3>
        <div id="extractedInfo"></div>
        <button onclick="completeEnrollment()" class="button">Generate Digital ID</button>
        <div id="enrollmentStatus"></div>
    </div>
    
    <script>
        let extractedData = {};
        let faceImageData = "";
        let videoStream = null;
        let faceDetected = false;
        
        function uploadID() {
            const fileInput = document.getElementById('idCard');
            if (!fileInput.files[0]) {
                document.getElementById('uploadStatus').innerHTML = 
                    `<div class="status error">Please select a file first!</div>`;
                return;
            }
            
            const formData = new FormData();
            formData.append('id_card', fileInput.files[0]);
            
            document.getElementById('uploadStatus').innerHTML = 
                `<div class="status info">Processing ID card...</div>`;
            
            fetch('/upload_id', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    extractedData = data;
                    document.getElementById('uploadStatus').innerHTML = 
                        `<div class="status success">✅ ID card processed successfully!</div>`;
                    document.getElementById('extractedInfo').innerHTML = 
                        `<p><strong>Name:</strong> ${data.name}</p>
                         <p><strong>Roll No:</strong> ${data.roll}</p>
                         <p><strong>Department:</strong> ${data.dept}</p>`;
                    document.getElementById('faceCaptureStep').classList.remove('hidden');
                    document.getElementById('enrollmentStep').classList.remove('hidden');
                } else {
                    document.getElementById('uploadStatus').innerHTML = 
                        `<div class="status error">❌ Error: ${data.error}</div>`;
                }
            })
            .catch(error => {
                document.getElementById('uploadStatus').innerHTML = 
                    `<div class="status error">❌ Error: ${error}</div>`;
            });
        }
        
        function startCamera() {
            document.getElementById('faceStatus').innerHTML = 
                `<div class="status info">Starting camera...</div>`;
            
            navigator.mediaDevices.getUserMedia({ 
                video: { 
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                } 
            })
            .then(stream => {
                videoStream = stream;
                const video = document.getElementById('video');
                video.srcObject = stream;
                
                // Enable buttons
                document.getElementById('captureBtn').disabled = false;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('startBtn').disabled = true;
                
                document.getElementById('faceStatus').innerHTML = 
                    `<div class="status success">✅ Camera started. Position your face in the green box.</div>`;
                
                // Start face detection preview
                startFaceDetectionPreview();
            })
            .catch(error => {
                document.getElementById('faceStatus').innerHTML = 
                    `<div class="status error">❌ Camera error: ${error.message}</div>`;
            });
        }
        
        function startFaceDetectionPreview() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            function detectFace() {
                if (!videoStream) return;
                
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert to blob and send for face detection
                canvas.toBlob(blob => {
                    const formData = new FormData();
                    formData.append('image', blob);
                    
                    fetch('/detect_face', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.face_detected) {
                            faceDetected = true;
                            document.getElementById('faceStatus').innerHTML = 
                                `<div class="status success">✅ Face detected! Ready to capture.</div>`;
                        } else {
                            faceDetected = false;
                            if (data.success) {
                                document.getElementById('faceStatus').innerHTML = 
                                    `<div class="status info">👤 No face detected. Please position your face in the green box.</div>`;
                            }
                        }
                    });
                }, 'image/jpeg');
                
                if (videoStream) {
                    setTimeout(detectFace, 1000); // Check every second
                }
            }
            
            detectFace();
        }
        
        function captureFace() {
            if (!faceDetected) {
                document.getElementById('faceStatus').innerHTML = 
                    `<div class="status error">❌ No face detected. Please position your face properly.</div>`;
                return;
            }
            
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            faceImageData = canvas.toDataURL('image/jpeg', 0.9);
            document.getElementById('capturedFace').src = faceImageData;
            
            // Send for face verification
            fetch('/verify_face_capture', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: faceImageData })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('faceStatus').innerHTML = 
                        `<div class="status success">✅ Face captured successfully! ${data.faces_detected} face(s) detected.</div>`;
                } else {
                    document.getElementById('faceStatus').innerHTML = 
                        `<div class="status error">❌ ${data.error}</div>`;
                    faceImageData = "";
                    document.getElementById('capturedFace').src = "";
                }
            });
        }
        
        function stopCamera() {
            if (videoStream) {
                videoStream.getTracks().forEach(track => track.stop());
                videoStream = null;
                document.getElementById('video').srcObject = null;
                
                // Disable buttons
                document.getElementById('captureBtn').disabled = true;
                document.getElementById('stopBtn').disabled = true;
                document.getElementById('startBtn').disabled = false;
                
                document.getElementById('faceStatus').innerHTML = 
                    `<div class="status info">Camera stopped.</div>`;
            }
        }
        
        function testFaceDetection() {
            const video = document.getElementById('video');
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            canvas.toBlob(blob => {
                const formData = new FormData();
                formData.append('image', blob);
                
                document.getElementById('faceStatus').innerHTML = 
                    `<div class="status info">Testing face detection...</div>`;
                
                fetch('/test_face_detection', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        let message = data.face_detected ? 
                            `✅ Face detected! ${data.faces_detected} face(s) found.` :
                            `❌ No face detected. Tips: ${data.tips}`;
                        
                        document.getElementById('faceStatus').innerHTML = 
                            `<div class="status ${data.face_detected ? 'success' : 'error'}">${message}</div>`;
                    }
                });
            }, 'image/jpeg');
        }
        
        function completeEnrollment() {
            if (!faceImageData) {
                document.getElementById('enrollmentStatus').innerHTML = 
                    `<div class="status error">❌ Please capture face first!</div>`;
                return;
            }
            
            if (!faceDetected) {
                document.getElementById('enrollmentStatus').innerHTML = 
                    `<div class="status error">❌ No valid face detected. Please recapture.</div>`;
                return;
            }
            
            const data = {
                ...extractedData,
                face_image: faceImageData
            };
            
            document.getElementById('enrollmentStatus').innerHTML = 
                `<div class="status info">Processing enrollment...</div>`;
            
            fetch('/complete_enrollment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('enrollmentStatus').innerHTML = 
                        `<div class="status success">
                            ✅ Enrollment completed! 
                            <a href="/download/${data.filename}" target="_blank">Download Digital ID</a>
                        </div>`;
                    // Stop camera after successful enrollment
                    stopCamera();
                } else {
                    document.getElementById('enrollmentStatus').innerHTML = 
                        `<div class="status error">❌ Error: ${data.error}</div>`;
                }
            });
        }
    </script>
</body>
</html>
"""

def detect_faces(image_data):
    """Enhanced face detection with multiple methods"""
    # Decode image
    if isinstance(image_data, str) and image_data.startswith('data:image'):
        # Base64 encoded
        face_bytes = base64.b64decode(image_data.split(",")[1])
        nparr = np.frombuffer(face_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Already an image array
        img = image_data
    
    if img is None:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple enhancements
    # 1. Histogram equalization
    gray_eq = cv2.equalizeHist(gray)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_clahe = clahe.apply(gray)
    
    # 3. Gaussian blur for noise reduction
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    all_faces = []
    
    # Try different image variations
    variations = [
        ("original", gray),
        ("equalized", gray_eq),
        ("clahe", gray_clahe),
        ("blurred", gray_blur)
    ]
    
    for cascade in face_cascades[:2]:  # Use first two cascades
        for var_name, gray_img in variations:
            faces = cascade.detectMultiScale(
                gray_img,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(60, 60),
                maxSize=(300, 300),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                all_faces.extend(faces)
    
    # If no faces found, try with more lenient parameters
    if len(all_faces) == 0:
        for cascade in face_cascades[:2]:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=2,
                minSize=(40, 40),
                maxSize=(400, 400),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                all_faces.extend(faces)
    
    # Remove duplicates and keep the largest face
    if len(all_faces) > 0:
        # Group overlapping faces
        filtered_faces = []
        for (x, y, w, h) in all_faces:
            # Check if this face overlaps significantly with any existing face
            overlapping = False
            for (fx, fy, fw, fh) in filtered_faces:
                # Calculate overlap
                x_overlap = max(0, min(x + w, fx + fw) - max(x, fx))
                y_overlap = max(0, min(y + h, fy + fh) - max(y, fy))
                overlap_area = x_overlap * y_overlap
                area1 = w * h
                area2 = fw * fh
                
                if overlap_area > 0.5 * min(area1, area2):
                    overlapping = True
                    # Keep the larger face
                    if area1 > area2:
                        filtered_faces.remove((fx, fy, fw, fh))
                        filtered_faces.append((x, y, w, h))
                    break
            
            if not overlapping:
                filtered_faces.append((x, y, w, h))
        
        return filtered_faces
    
    return []

def preprocess_face_image(img):
    """Preprocess face image for better recognition"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Resize to standard size
    gray = cv2.resize(gray, (100, 100))
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    return gray

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/upload_id", methods=["POST"])
def upload_id():
    if 'id_card' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    
    file = request.files["id_card"]
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"})
    
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # Extract text using OCR
    try:
        img = cv2.imread(image_path)
        if img is None:
            return jsonify({"success": False, "error": "Could not read image"})
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh)
    except Exception as e:
        return jsonify({"success": False, "error": f"OCR error: {str(e)}"})
    
    name = "NOT FOUND"
    roll = "NOT FOUND"
    dept = "NOT FOUND"

    for line in text.split("\n"):
        line = line.strip()
        
        # Look for patterns with colons
        if ":" in line:
            if re.search(r"name", line, re.IGNORECASE):
                name = line.split(":")[-1].strip()
            elif re.search(r"roll|id", line, re.IGNORECASE):
                roll = line.split(":")[-1].strip()
            elif re.search(r"dept|department", line, re.IGNORECASE):
                dept = line.split(":")[-1].strip()
        # Also try without colons
        elif re.search(r"name\s+[A-Za-z]", line, re.IGNORECASE):
            name = re.sub(r"name\s*", "", line, flags=re.IGNORECASE).strip()
        elif re.search(r"roll|id\s+[A-Za-z0-9]", line, re.IGNORECASE):
            roll = re.sub(r"roll|id\s*", "", line, flags=re.IGNORECASE).strip()

    return jsonify({
        "success": True,
        "name": name if name != "NOT FOUND" else "Unknown",
        "roll": roll if roll != "NOT FOUND" else "Unknown",
        "dept": dept if dept != "NOT FOUND" else "Unknown",
        "original_filename": file.filename
    })

@app.route("/detect_face", methods=["POST"])
def detect_face_route():
    """Endpoint for real-time face detection"""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image"})
    
    faces = detect_faces(img)
    
    return jsonify({
        "success": True,
        "face_detected": len(faces) > 0,
        "faces_detected": len(faces)
    })

@app.route("/test_face_detection", methods=["POST"])
def test_face_detection():
    """Test face detection with helpful tips"""
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image provided"})
    
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image"})
    
    faces = detect_faces(img)
    
    tips = []
    if len(faces) == 0:
        # Analyze image for issues
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50:
            tips.append("Image is too dark - improve lighting")
        elif brightness > 200:
            tips.append("Image is too bright - reduce glare")
        
        # Check if face might be at wrong angle
        height, width = img.shape[:2]
        if height < 200 or width < 200:
            tips.append("Move closer to camera")
        
        tips.append("Ensure face is clearly visible")
        tips.append("Look directly at camera")
        tips.append("Remove glasses if they cause glare")
        tips.append("Ensure good lighting from the front")
    
    return jsonify({
        "success": True,
        "face_detected": len(faces) > 0,
        "faces_detected": len(faces),
        "tips": tips
    })

@app.route("/verify_face_capture", methods=["POST"])
def verify_face_capture():
    """Verify that a captured face image contains a valid face"""
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"success": False, "error": "No image data"})
    
    image_data = data['image']
    
    # Decode base64 image
    face_bytes = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(face_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image"})
    
    faces = detect_faces(img)
    
    if len(faces) == 0:
        return jsonify({
            "success": False,
            "error": "No face detected in captured image. Please try again."
        })
    
    # Check if face is too small
    (x, y, w, h) = faces[0]
    if w < 100 or h < 100:
        return jsonify({
            "success": False,
            "error": "Face is too small. Move closer to camera."
        })
    
    return jsonify({
        "success": True,
        "faces_detected": len(faces),
        "face_size": f"{w}x{h}"
    })

@app.route("/complete_enrollment", methods=["POST"])
def complete_enrollment():
    data = request.json
    if not data:
        return jsonify({"success": False, "error": "No data provided"})
    
    name = data.get("name", "Unknown")
    roll = data.get("roll", "Unknown")
    dept = data.get("dept", "Unknown")
    face_image_data = data.get("face_image")
    
    if not face_image_data:
        return jsonify({"success": False, "error": "No face image provided"})
    
    # Decode base64 image
    face_bytes = base64.b64decode(face_image_data.split(",")[1])
    nparr = np.frombuffer(face_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"success": False, "error": "Could not decode face image"})
    
    # Detect and save face
    faces = detect_faces(img)
    
    if len(faces) == 0:
        return jsonify({"success": False, "error": "No face detected in image"})
    
    # Get the largest face
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    
    # Crop face
    face_cropped = img[y:y+h, x:x+w]
    
    # Preprocess face for better recognition
    face_processed = preprocess_face_image(face_cropped)
    
    # Save face images
    face_path = os.path.join(FACES_FOLDER, f"{roll}.jpg")
    face_cropped_path = os.path.join(FACES_FOLDER, f"{roll}_cropped.jpg")
    
    # Save color image
    cv2.imwrite(face_path, img)
    
    # Save processed face
    cv2.imwrite(face_cropped_path, face_processed)
    
    # Add to face database
    face_database["labels"].append(roll)
    face_database["faces"][roll] = {
        "name": name,
        "dept": dept,
        "face_path": face_cropped_path
    }
    
    # Save face database
    with open(FACE_DB_FILE, 'w') as f:
        json.dump(face_database, f)
    
    # Generate QR code
    qr_data = f"Name:{name};Roll:{roll};Dept:{dept}"
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(qr_data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert QR image to numpy array
    qr_np = np.array(qr_img.convert('RGB'))
    qr_np = cv2.cvtColor(qr_np, cv2.COLOR_RGB2BGR)
    
    # Create digital ID
    card = np.ones((500, 700, 3), dtype=np.uint8) * 255
    
    # Add border
    cv2.rectangle(card, (10, 10), (690, 490), (0, 100, 200), 4)
    
    # Add title
    cv2.putText(card, "DIGITAL STUDENT ID", (180, 60), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(card, "CAMPUS SENTINEL", (220, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
    
    # Add student info
    cv2.putText(card, "STUDENT INFORMATION:", (30, 160), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(card, f"Name: {name}", (30, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(card, f"Roll No: {roll}", (30, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(card, f"Department: {dept}", (30, 280), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(card, "Status: VERIFIED ✅", (30, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 150, 0), 2)
    
    # Add face image with border
    if face_cropped is not None:
        face_display = cv2.resize(face_cropped, (150, 150))
        card[150:300, 500:650] = face_display
        cv2.rectangle(card, (500, 150), (650, 300), (0, 150, 0), 2)
        cv2.putText(card, "FACE", (540, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add QR code with border
    if qr_np is not None:
        qr_display = cv2.resize(qr_np, (120, 120))
        card[330:450, 515:635] = qr_display
        cv2.rectangle(card, (515, 330), (635, 450), (0, 0, 200), 2)
        cv2.putText(card, "QR CODE", (530, 470), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Add issue date
    import datetime
    issue_date = datetime.datetime.now().strftime("%Y-%m-%d")
    cv2.putText(card, f"Issue Date: {issue_date}", (30, 400), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Save digital ID
    digital_id_path = os.path.join(OUTPUT_FOLDER, f"digital_id_{roll}.jpg")
    cv2.imwrite(digital_id_path, card)
    
    return jsonify({
        "success": True,
        "filename": f"digital_id_{roll}.jpg",
        "message": "Enrollment completed successfully!"
    })

@app.route("/download/<filename>")
def download(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')