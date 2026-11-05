"""
NotificationAgent: Sends alerts when suspicious activity is detected.
"""

from typing import Dict, List, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import os
from io import BytesIO
import base64
import cv2
import numpy as np

# Twilio is optional; code will degrade gracefully if not installed
try:
    from twilio.rest import Client as TwilioClient
except Exception:  # pragma: no cover
    TwilioClient = None

# requests is optional, used for optional image upload (e.g., to imgbb)
try:
    import requests
except Exception:  # pragma: no cover
    requests = None


class NotificationAgent:
    """Agent responsible for sending alerts and notifications."""
    
    def __init__(self, enable_email: bool = False,
                 enable_sms: bool = False,
                 enable_whatsapp: bool = False,
                 enable_console: bool = True,
                 enable_visual: bool = True,
                 email_recipients: List[str] = None,
                 sms_recipients: List[str] = None,
                 whatsapp_recipients: List[str] = None):
        """
        Initialize NotificationAgent.
        
        Args:
            enable_email: Enable email notifications
            enable_sms: Enable SMS notifications
            enable_console: Enable console logging
            enable_visual: Enable visual on-screen alerts
            email_recipients: List of email addresses
            sms_recipients: List of phone numbers
            whatsapp_recipients: List of phone numbers (E.164, no spaces)
        """
        self.enable_email = enable_email
        self.enable_sms = enable_sms
        self.enable_whatsapp = enable_whatsapp
        self.enable_console = enable_console
        self.enable_visual = enable_visual
        self.email_recipients = email_recipients or []
        self.sms_recipients = sms_recipients or []
        self.whatsapp_recipients = whatsapp_recipients or []
        self.alert_history = []
        # Run network-bound notification channels asynchronously so video processing is not blocked.
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="notif")
        
    def send_alert(self, decision_result: Dict, frame: Optional[np.ndarray] = None,
                   camera_id: str = "Camera#1", timestamp: Optional[datetime] = None):
        """
        Send alert based on decision result.
        
        Args:
            decision_result: Result from DecisionAgent
            frame: Current video frame (for snapshot)
            camera_id: Identifier for the camera
            timestamp: Timestamp of the alert
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        person_id = decision_result['person_id']
        status = decision_result['status']
        threat_level = decision_result['threat_level']
        reason = decision_result.get('reason', 'Unknown')
        
        # Create alert message
        alert_message = self._create_alert_message(
            person_id, status, threat_level, reason, camera_id, timestamp
        )
        
        # Store in history
        alert_entry = {
            'timestamp': timestamp,
            'person_id': person_id,
            'status': status,
            'threat_level': threat_level,
            'reason': reason,
            'camera_id': camera_id,
            'message': alert_message
        }
        self.alert_history.append(alert_entry)
        
        # Send notifications based on status and threat level
        if status in ['Loitering', 'Intrusion']:
            if self.enable_console:
                self._console_alert(alert_message)
            
            if self.enable_email and self.email_recipients:
                self._executor.submit(self._email_alert, alert_message, frame, person_id, timestamp)
            
            if self.enable_sms and self.sms_recipients:
                self._executor.submit(self._sms_alert, alert_message)

            if self.enable_whatsapp and self.whatsapp_recipients:
                self._executor.submit(self._whatsapp_alert, alert_message, frame, person_id, timestamp)
    
    def _create_alert_message(self, person_id: int, status: str, threat_level: str,
                             reason: str, camera_id: str, timestamp: datetime) -> str:
        """Create formatted alert message."""
        time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"🚨 ALERT: {status} Detected\n"
            f"Person ID: {person_id}\n"
            f"Threat Level: {threat_level.upper()}\n"
            f"Reason: {reason}\n"
            f"Camera: {camera_id}\n"
            f"Time: {time_str}"
        )
    
    def _console_alert(self, message: str):
        """Print alert to console."""
        print("\n" + "="*60)
        print(message)
        print("="*60 + "\n")
    
    def _email_alert(self, message: str, frame: Optional[np.ndarray],
                    person_id: int, timestamp: datetime):
        """Send email alert using SMTP settings from environment variables.
        
        Required env:
          - SMTP_HOST
          - SMTP_PORT (int)
          - SMTP_USERNAME
          - SMTP_PASSWORD
          - SMTP_FROM (email)
        Optional:
          - SMTP_STARTTLS (default: 'true')
        """
        try:
            smtp_host = os.getenv("SMTP_HOST")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_user = os.getenv("SMTP_USERNAME")
            smtp_pass = os.getenv("SMTP_PASSWORD")
            smtp_from = os.getenv("SMTP_FROM", smtp_user or "")
            use_starttls = os.getenv("SMTP_STARTTLS", "true").lower() in ("1", "true", "yes")

            if not (smtp_host and smtp_user and smtp_pass and smtp_from):
                print("[EMAIL] SMTP not configured; skipping email send.")
                return

            msg = MIMEMultipart()
            msg["From"] = smtp_from
            msg["To"] = ", ".join(self.email_recipients)
            msg["Subject"] = f"Security Alert: Person {person_id} at {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(message, "plain"))

            # Attach frame if available
            if frame is not None:
                success, buffer = cv2.imencode(".jpg", frame)
                if success:
                    img = MIMEImage(buffer.tobytes(), _subtype="jpeg")
                    img.add_header("Content-Disposition", "attachment", filename=f"alert_{person_id}.jpg")
                    msg.attach(img)

            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
            if use_starttls:
                server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
            server.quit()
            print(f"[EMAIL] Sent email alert to {len(self.email_recipients)} recipient(s).")
        except Exception as e:
            print(f"[EMAIL ERROR] Failed to send email: {e}")
    
    def _sms_alert(self, message: str):
        """Send SMS alert via Twilio using environment variables.
        
        Required env:
          - TWILIO_ACCOUNT_SID
          - TWILIO_AUTH_TOKEN
          - TWILIO_SMS_FROM (E.164)
        """
        try:
            if TwilioClient is None:
                print("[SMS] Twilio SDK not installed; skipping SMS send.")
                return

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            sms_from = os.getenv("TWILIO_SMS_FROM")
            if not (account_sid and auth_token and sms_from):
                print("[SMS] Twilio not configured; skipping SMS send.")
                return

            client = TwilioClient(account_sid, auth_token)
            for recipient in self.sms_recipients:
                client.messages.create(
                    body=message,
                    from_=sms_from,
                    to=recipient
                )
            print(f"[SMS] Sent SMS alert to {len(self.sms_recipients)} recipient(s).")
        except Exception as e:
            print(f"[SMS ERROR] Failed to send SMS: {e}")

    def _whatsapp_alert(self, message: str, frame: Optional[np.ndarray],
                        person_id: int, timestamp: datetime):
        """Send WhatsApp alert (optionally with image) via Twilio WhatsApp.
        
        Required env:
          - TWILIO_ACCOUNT_SID
          - TWILIO_AUTH_TOKEN
          - TWILIO_WHATSAPP_FROM (e.g., 'whatsapp:+14155238886')
        
        To include image, provide a media URL. If IMGBB_API_KEY is set and
        requests is available, the frame will be uploaded to imgbb and the
        resulting URL attached as media.
        """
        try:
            if TwilioClient is None:
                print("[WHATSAPP] Twilio SDK not installed; skipping WhatsApp send.")
                return

            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            wa_from = os.getenv("TWILIO_WHATSAPP_FROM")  # must include 'whatsapp:' prefix
            if not (account_sid and auth_token and wa_from):
                print("[WHATSAPP] Twilio WhatsApp not configured; skipping WhatsApp send.")
                return

            media_url = None
            if frame is not None:
                media_url = self._try_upload_image_get_url(frame, person_id, timestamp)

            client = TwilioClient(account_sid, auth_token)
            for recipient in self.whatsapp_recipients:
                to_number = recipient if recipient.startswith("whatsapp:") else f"whatsapp:{recipient}"
                kwargs = {
                    "body": f"{message}\n\nTimestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
                }
                if media_url:
                    kwargs["media_url"] = [media_url]
                client.messages.create(
                    from_=wa_from if wa_from.startswith("whatsapp:") else f"whatsapp:{wa_from}",
                    to=to_number,
                    **kwargs
                )
            print(f"[WHATSAPP] Sent WhatsApp alert to {len(self.whatsapp_recipients)} recipient(s).")
        except Exception as e:
            print(f"[WHATSAPP ERROR] Failed to send WhatsApp: {e}")

    def _try_upload_image_get_url(self, frame: np.ndarray, person_id: int, timestamp: datetime) -> Optional[str]:
        """Attempt to upload an image to a public host and return a URL.
        
        Current implementation supports imgbb.com if IMGBB_API_KEY is present.
        Returns None if upload is not possible.
        """
        try:
            if requests is None:
                return None
            api_key = os.getenv("IMGBB_API_KEY")
            if not api_key:
                return None

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                return None
            b64_data = base64.b64encode(buffer.tobytes()).decode("utf-8")
            payload = {
                "key": api_key,
                "image": b64_data,
                "name": f"alert_{person_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            }
            resp = requests.post("https://api.imgbb.com/1/upload", data=payload, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("data", {}).get("url")
        except Exception:
            return None
        return None
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict]:
        """Get recent alerts from history."""
        return self.alert_history[-limit:]
    
    def get_alerts_by_status(self, status: str) -> List[Dict]:
        """Get all alerts with specific status."""
        return [alert for alert in self.alert_history if alert['status'] == status]

