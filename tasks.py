from .cron import DailyDataExtractionCronJob, DailyComparisonCronJob
import logging

logger = logging.getLogger(__name__)

def apscheduler_daily_extraction_job():
    try:
        cron_job = DailyDataExtractionCronJob()
        cron_job.do()
        logger.info("APScheduler daily extraction job executed successfully.")
    except Exception as e:
        logger.error(f"Error in APScheduler daily extraction job: {e}")

def apscheduler_comparison_job():
    try:
        cron_job = DailyComparisonCronJob()
        cron_job.do()
        logger.info("APScheduler comparison job executed successfully.")
    except Exception as e:
        logger.error(f"Error in APScheduler comparison job: {e}")
