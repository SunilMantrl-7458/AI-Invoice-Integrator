import logging
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver
from django.utils.timezone import now

logger = logging.getLogger(__name__)

@receiver(user_logged_in)
def log_user_login(sender, request, user, **kwargs):
    logger.info(f"User logged in: username={user.username}, role={getattr(user, 'role', 'unknown')}, time={now()}")
