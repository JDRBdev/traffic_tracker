from unittest.mock import patch

import numpy as np

from config import settings
from notifier import EmailNotifier


def test_notifier_cooldown():
    """Test that immediate subsequent calls to notify() are suppressed by the cooldown logic."""
    # Force mock settings
    settings.SENDGRID_API_KEY = ""
    settings.SMTP_USER = "test@test.com"
    settings.SMTP_PASSWORD = "password"
    
    # 60s cooldown is the default, ensuring it will trigger suppression on the second call
    settings.EMAIL_COOLDOWN_SECONDS = 60
    
    notifier = EmailNotifier()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with patch("smtplib.SMTP") as mock_smtp:
        # 1. First notify - should send email via SMTP
        notifier.notify(dummy_crop)
        assert mock_smtp.called
        
        # Get the mock instance returned by SMTP()
        mock_server = mock_smtp.return_value.__enter__.return_value
        assert mock_server.send_message.called
        
    with patch("smtplib.SMTP") as mock_smtp_2:
        # 2. Second notify immediately after - should hit cooldown and not send
        notifier.notify(dummy_crop)
        assert not mock_smtp_2.called

def test_notifier_per_car():
    """Test that notify() uses per-car suppression when track_id is provided."""
    settings.SENDGRID_API_KEY = ""
    settings.SMTP_USER = "test@test.com"
    settings.SMTP_PASSWORD = "password"
    
    notifier = EmailNotifier()
    dummy_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # 1. Notify for car 1
    with patch("smtplib.SMTP") as mock_smtp:
        notifier.notify(dummy_crop, track_id=1)
        assert mock_smtp.called
        
    # 2. Notify for car 1 again - should be suppressed
    with patch("smtplib.SMTP") as mock_smtp_2:
        notifier.notify(dummy_crop, track_id=1)
        assert not mock_smtp_2.called
        
    # 3. Notify for car 2 immediately - should NOT be suppressed
    with patch("smtplib.SMTP") as mock_smtp_3:
        notifier.notify(dummy_crop, track_id=2)
        assert mock_smtp_3.called
