from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Student(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    roll_no = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    face_path = db.Column(db.String(255), nullable=True)
    qr_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "roll_no": self.roll_no,
            "department": self.department,
            "face_path": self.face_path,
            "qr_path": self.qr_path,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(255), nullable=True)
    location = db.Column(db.String(100), nullable=False)
    time = db.Column(db.String(50), nullable=False)
    threat_score = db.Column(db.Integer, default=0)
    threat_level = db.Column(db.String(20), default='LOW')
    summary = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='Pending') # Pending, Resolved, Investigating
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "image_path": self.image_path,
            "location": self.location,
            "time": self.time,
            "threat_score": self.threat_score,
            "threat_level": self.threat_level,
            "summary": self.summary,
            "status": self.status,
            "created_at": self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        }
