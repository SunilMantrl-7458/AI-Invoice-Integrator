from django_cron import CronJobBase
import logging
from django_cron.models import CronJobLog
from django.utils.timezone import now
from django.core.cache import cache

logger = logging.getLogger(__name__)

class DailyDataExtractionCronJob(CronJobBase):
    code = 'testapp.daily_data_extraction'

    def do(self):
        start_time = now()
        log = CronJobLog.objects.create(
            code=self.code,
            start_time=start_time,
            is_success=False,
            message='Processing',
            end_time=start_time  # Avoid NULL end_time
        )

        # Add debug logs to confirm cache updates
        logger.debug("Setting extraction_job_status to 'processing'")
        cache.set('extraction_job_status', 'processing', timeout=3600)
        cache.set('extraction_job_running', True)

        try:
            from .extraction import run_data_extraction
            run_data_extraction()

            log.is_success = True
            log.message = 'Extraction completed successfully'
            logger.debug("Setting extraction_job_status to 'success'")
            cache.set('extraction_job_status', 'success', timeout=3600)
            logger.info("Extraction completed successfully.")

        except Exception as e:
            log.is_success = False
            log.message = f"Error: {str(e)}"
            logger.debug("Setting extraction_job_status to 'failed'")
            cache.set('extraction_job_status', 'failed', timeout=3600)
            logger.error(f"Error in data extraction cron job: {str(e)}")

        finally:
            logger.debug(f"Before saving log: is_success={log.is_success}, message={log.message}")
            log.end_time = now()
            log.save()
            # Verify the saved log state
            saved_log = CronJobLog.objects.get(id=log.id)
            logger.debug(f"Saved log state: is_success={saved_log.is_success}, message={saved_log.message}")
            cache.set('extraction_job_running', False)
            logger.debug(f"Log saved with ID {log.id}: is_success={log.is_success}, message={log.message}")


class DailyComparisonCronJob(CronJobBase):
    code = 'testapp.daily_comparison'

    def do(self):
        now_time = now()
        log = CronJobLog.objects.create(
            code=self.code,
            start_time=now_time,
            end_time=now_time,
            is_success=False,
            message='Processing',
        )

        cache.set('comparison_job_status', 'processing')
        cache.set('comparison_job_running', True)

        try:
            from .views import run_comparison
            run_comparison()

            log.is_success = True
            log.message = 'Comparison executed successfully'
            cache.set('comparison_job_status', 'success')
            logger.info("Comparison completed successfully.")

        except Exception as e:
            log.is_success = False
            log.message = f"Error: {str(e)}"
            cache.set('comparison_job_status', 'failed')
            logger.error(f"Error in comparison cron job: {str(e)}")

        finally:
            logger.debug(f"Before saving log: is_success={log.is_success}, message={log.message}")
            log.end_time = now()
            log.save()
            # Verify the saved log state
            saved_log = CronJobLog.objects.get(id=log.id)
            logger.debug(f"Saved log state: is_success={saved_log.is_success}, message={saved_log.message}")
            cache.set('comparison_job_running', False)
            logger.debug(f"Log saved with ID {log.id}: is_success={log.is_success}, message={log.message}")