# src/data_logger.py

import csv
import sqlite3
import pandas as pd
from datetime import datetime
import os
import json
import logging

class EmotionLogger:
    """Logging system for emotion detection results"""
    
    def __init__(self, log_dir='data/logs', db_path='data/emotion_logs.db'):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for CSV logs
            db_path: Path to SQLite database
        """
        self.log_dir = log_dir
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Current session file
        self.session_file = os.path.join(
            log_dir, 
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        # Initialize CSV file with headers
        with open(self.session_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'emotion', 'confidence', 'face_id', 'frame_id'])
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create emotions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emotion TEXT,
                confidence REAL,
                face_id TEXT,
                frame_id INTEGER,
                session_id TEXT
            )
        ''')
        
        # Create statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                emotion TEXT,
                count INTEGER,
                avg_confidence REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_emotion(self, emotion, confidence, face_id='face_1', frame_id=0):
        """
        Log single emotion detection
        
        Args:
            emotion: Detected emotion
            confidence: Confidence score
            face_id: Face identifier
            frame_id: Frame number
        """
        timestamp = datetime.now().isoformat()
        
        # Log to CSV
        with open(self.session_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, emotion, confidence, face_id, frame_id])
        
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO emotions (timestamp, emotion, confidence, face_id, frame_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, emotion, confidence, face_id, frame_id, os.path.basename(self.session_file)))
        conn.commit()
        conn.close()
    
    def log_multiple_faces(self, detections, frame_id=0):
        """
        Log multiple face detections
        
        Args:
            detections: List of detection dictionaries
            frame_id: Frame number
        """
        for i, detection in enumerate(detections):
            face_id = f"face_{i+1}"
            self.log_emotion(
                detection['emotion'],
                detection['confidence'],
                face_id,
                frame_id
            )
    
    def get_recent_logs(self, limit=100):
        """Get recent logs from database"""
        conn = sqlite3.connect(self.db_path)
        query = f"""
            SELECT timestamp, emotion, confidence, face_id 
            FROM emotions 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_statistics(self, time_range='day'):
        """
        Get emotion statistics
        
        Args:
            time_range: 'day', 'week', 'month', or 'all'
        """
        conn = sqlite3.connect(self.db_path)
        
        if time_range == 'day':
            query = """
                SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM emotions
                WHERE date(timestamp) = date('now')
                GROUP BY emotion
            """
        elif time_range == 'week':
            query = """
                SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM emotions
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY emotion
            """
        elif time_range == 'month':
            query = """
                SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM emotions
                WHERE timestamp >= datetime('now', '-30 days')
                GROUP BY emotion
            """
        else:
            query = """
                SELECT emotion, COUNT(*) as count, AVG(confidence) as avg_confidence
                FROM emotions
                GROUP BY emotion
            """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def export_report(self, format='csv'):
        """
        Export emotion report
        
        Args:
            format: 'csv' or 'json'
        """
        report_dir = 'data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get all data
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM emotions", conn)
        conn.close()
        
        if format == 'csv':
            report_file = os.path.join(report_dir, f'report_{timestamp}.csv')
            df.to_csv(report_file, index=False)
        else:
            report_file = os.path.join(report_dir, f'report_{timestamp}.json')
            df.to_json(report_file, orient='records', date_format='iso')
        
        return report_file