from apscheduler.triggers.cron import CronTrigger
from .scheduler_engine import scheduler
from .tasks import apscheduler_daily_extraction_job, apscheduler_comparison_job

def setup_jobs():
    scheduler.add_job(
        apscheduler_daily_extraction_job,
        trigger=CronTrigger(hour=10, minute=33),  # Daily at 6:00 AM
        id='apscheduler_daily_extraction_job',
        replace_existing=True
    )
    scheduler.add_job(
        apscheduler_comparison_job,
        trigger=CronTrigger(hour=10, minute=34),  # Daily at 7:00 AM
        id='apscheduler_comparison_job',
        replace_existing=True
    )