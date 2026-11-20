"""Send a backup failure notification using an authenticated TLS connection."""
import os
import smtplib
import ssl
import stat
import sys
from email.message import EmailMessage
from pathlib import Path


def send_failure_alert(sender, recipient, password_file, *, host='smtp.gmail.com', port=465, username=None, test=False):
    path = Path(password_file)
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError('mail password file must be owned by the service user and private')
    password = path.read_text(encoding='utf-8').strip().replace(' ', '')
    if not sender or not recipient or not password:
        raise ValueError('email configuration is incomplete')
    message = EmailMessage()
    message['From'] = sender
    message['To'] = recipient
    message['Subject'] = '[SIOT] Backup alert test' if test else '[SIOT] Backup failed'
    message.set_content('This is a test of SIOT backup failure notifications.' if test else
                        'The scheduled SIOT backup failed. Check: journalctl -u siot-backup.service -n 50 --no-pager')
    with smtplib.SMTP_SSL(host, port, timeout=30, context=ssl.create_default_context()) as smtp:
        smtp.login(username or sender, password)
        smtp.send_message(message)


def main():
    try:
        send_failure_alert(os.environ['SIOT_ALERT_SENDER'], os.environ['SIOT_ALERT_RECIPIENT'],
                           os.environ['SIOT_ALERT_PASSWORD_FILE'],
                           host=os.environ.get('SIOT_ALERT_SMTP_HOST', 'smtp.gmail.com'),
                           port=int(os.environ.get('SIOT_ALERT_SMTP_PORT', '465')),
                           username=os.environ.get('SIOT_ALERT_SMTP_USERNAME'),
                           test=sys.argv[1:] == ['--test'])
    except Exception:
        print('Backup alert email failed; check mail configuration.', file=sys.stderr)
        return 1
    print('Backup alert accepted by mail server.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
