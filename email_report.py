"""
=========================================================
AI Face Attendance System
Email Report Module
Author : Malay Maity
=========================================================
"""

import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import config


def send_attendance_report(csv_file, excel_file):
    if not config.ENABLE_EMAIL:
        print("Email sending is disabled in config.")
        return

    if not os.path.exists(csv_file) or not os.path.exists(excel_file):
        print("Attendance files are missing. Cannot send email report.")
        return

    msg = EmailMessage()
    msg['Subject'] = config.EMAIL_SUBJECT
    msg['From'] = config.EMAIL_SENDER
    msg['To'] = config.EMAIL_RECEIVER

    body = f"Dear Sir/Madam,\n\nPlease find attached the attendance report for {datetime.now().strftime(config.DATE_FORMAT)}.\n\nThank you,\nAI Face Attendance System"
    msg.set_content(body)

    with open(csv_file, 'rb') as f:
        msg.add_attachment(
            f.read(),
            maintype='text',
            subtype='csv',
            filename=os.path.basename(csv_file)
        )

    with open(excel_file, 'rb') as f:
        msg.add_attachment(
            f.read(),
            maintype='application',
            subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=os.path.basename(excel_file)
        )

    try:
        server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
        server.starttls()
        server.login(config.EMAIL_SENDER, config.EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email report sent successfully.")
    except Exception as exc:
        print(f"Failed to send email report: {exc}")
