# 🛡️ Campus Sentinel

**Advanced AI-Powered Campus Security & Identity Management System**

Campus Sentinel is a comprehensive security solution designed to modernize campus safety using computer vision, generative AI, and automated identity verification. The system integrates student enrollment, real-time access control, and intelligent threat analysis into a unified platform.

---

## 🚀 Key Features

### 1. 🎓 Smart Student Enrollment (`register.py`)
- **Automated Data Extraction**: Uses **OCR (Tesseract)** to extract Name, Roll No, and Department directly from physical ID cards.
- **Biometric Capture**: Captures and processes student face data for facial recognition.
- **Digital Identity**: Generates a secure **QR Code** containing student details for quick access.

### 2. 🔐 Real-Time Verification Station (`verify.py`)
- **Two-Factor Authentication**: Combines **QR Code Scanning** with **Face Verification**.
- **Live Feedback**: Provides instant visual feedback (Access Granted/Denied) with confidence scores.
- **Anti-Spoofing**: Basic checks for face liveness and presence.

### 3. 🚨 AI Threat Analysis (`threat.py`)
- **Generative AI Integration**: Powered by **Google Gemini 2.5 Flash**.
- **Context-Aware Reporting**: Analyzes incident images to determine **Threat Level (Low/Medium/High)** and provides a professional summary.
- **Confidence Scoring**: Assigns a confidence score to every analysis to aid human decision-making.

### 4. 📊 Central Command Dashboard (`app.py`)
- **Live Statistics**: Real-time view of total students, active incidents, and threat levels.
- **Incident Management**: Track, update, and resolve reported security incidents.
- **Database Management**: View registered students and logs.

---

## 🛠️ Technical Architecture

### Tech Stack
- **Backend Framework**: Flask (Python)
- **Database**: SQLite (via SQLAlchemy)
- **Computer Vision**: OpenCV, NumPy
- **OCR Engine**: Tesseract-OCR
- **Generative AI**: Google Gemini API
- **Frontend**: HTML5, CSS3, JavaScript

### Implementation Details

#### Face Recognition Pipeline
The system utilizes a hybrid computer vision approach:
1.  **Detection**: Uses Multi-Scale **Haar Cascade Classifiers** for robust face detection under varying conditions.
2.  **Preprocessing**: Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) and Gaussian Blur to normalize lighting and reduce noise.
3.  **Verification**: Uses **Histogram Correlation** (Bhattacharyya distance / Correl) to compare the captured live face with the stored biometric reference.

#### Threat Analysis Logic
1.  **Input**: Incident image + Location + Time.
2.  **Processing**: The image is encoded to Base64 and sent to the **Gemini 2.5 Flash** model with a specific system prompt context.
3.  **Output**: Structured JSON-like response containing Threat Score, Categorization, and Summary, which is parsed and stored in the database.

---

## ⚙️ Installation & Setup

### Prerequisites
1.  **Python 3.8+**
2.  **Tesseract OCR** installed on your system.
    - *Windows*: [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki) (Add to PATH)
    - *Linux*: `sudo apt install tesseract-ocr`

### Steps
1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/campus-sentinel.git
    cd campus-sentinel
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    - Open `app.py` and `threat.py`.
    - Replace `GEMINI_API_KEY` with your valid Google Gemini API Key.

---

## 🚦 Usage Guide

### 1. Run the Web Dashboard
Start the main server to view stats and manage incidents.
```bash
python app.py
```
*Access at: `http://localhost:5000`*

### 2. Enroll a New Student
Launch the enrollment interface to register students.
```bash
python register.py
```
*Follow the on-screen steps: Upload ID -> Capture Face -> Generate QR.*

### 3. Start Verification Station
Run the specialized verification script for gate/entry points.
```bash
python verify.py
```
*This opens a camera window. Show the generated QR code, then position your face.*

### 4. Threat Analysis Tool
Run the standalone threat analysis tool.
```bash
python threat.py
```

---

## 📂 Project Structure

```
campus_sec/
├── app.py              # Main Flask Application (Dashboard)
├── register.py         # Student Enrollment Logic
├── verify.py           # Standalone Verification System
├── threat.py           # Threat Analysis Module
├── models.py           # Database Models (SQLAlchemy)
├── requirements.txt    # Project Dependencies
├── faces/              # Stored student face images
├── qrcodes/            # Generated Student QR codes
├── static/             # CSS/JS assets
├── templates/          # HTML Templates
└── campus_sec.db       # SQLite Database
```

---

## 📜 License
This project is open-source and available under the MIT License.
