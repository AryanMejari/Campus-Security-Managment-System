import os
import cv2
import base64
import requests
from flask import Flask, request, render_template_string

# ---------------- CONFIG ----------------
API_KEY = "AIzaSyBunDAFzp4uj1Z_YszfEMl-BPQsRmIqpHI"

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent?key=" + API_KEY
)



UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ---------------- BASIC UI ----------------
HTML = """
<h2>Campus Sentinel – Incident Threat Analysis</h2>
<form method="POST" action="/analyze" enctype="multipart/form-data">
    <label>Incident Image:</label><br>
    <input type="file" name="image" required><br><br>

    <label>Location:</label><br>
    <input type="text" name="location" required><br><br>

    <label>Time:</label><br>
    <input type="text" name="time" required><br><br>

    <button type="submit">Analyze Threat</button>
</form>

{% if result %}
<hr>
<h3>AI Threat Analysis</h3>
<pre>{{ result }}</pre>
{% endif %}
"""

@app.route("/")
def home():
    return render_template_string(HTML)

# ---------------- ANALYSIS ENDPOINT ----------------
@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files["image"]
    location = request.form["location"]
    time = request.form["time"]

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    # ---------- READ IMAGE USING OPENCV ----------
    img = cv2.imread(image_path)
    if img is None:
        return "❌ Invalid image file"

    _, buffer = cv2.imencode(".jpg", img)
    image_base64 = base64.b64encode(buffer).decode("utf-8")

    # ---------- PROMPT ----------
    prompt = f"""
You are an AI campus security threat analysis system.

Analyze the uploaded incident image with the following context:

Location: {location}
Time: {time}

Tasks:
1. Give a threat confidence score from 0 to 100
2. Categorize threat strictly as:
   - LOW (0–30)
   - MEDIUM (31–70)
   - HIGH (71–100)
3. Provide a short professional incident summary (2–3 lines)

Return output in EXACT format:

Threat Confidence: <number>
Threat Level: <LOW/MEDIUM/HIGH>
Incident Summary: <text>
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(GEMINI_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return f"❌ AI API Error: {response.text}"

    result = response.json()["candidates"][0]["content"]["parts"][0]["text"]

    return render_template_string(HTML, result=result)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
