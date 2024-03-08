import smtplib
from email.mime.text import MIMEText


def send_email(subject, password, message="Experiment Complete"):
    msg = MIMEText(message)
    msg["Subject"] = "EXPERIMENT COMPLETE: " + subject
    msg["From"] = "marios1861@gmail.com"
    msg["To"] = "marios1861@gmail.com"
    smtp_server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
    smtp_server.login("marios1861@gmail.com", password)
    smtp_server.sendmail(
        "marios1861@gmail.com", "marios1861@gmail.com", msg.as_string()
    )
    smtp_server.quit()
