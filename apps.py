# from django.apps import AppConfig

# class TestappConfig(AppConfig):
#     default_auto_field = 'django.db.models.BigAutoField'
#     name = 'testapp'

#     def ready(self):
#         import testapp.signals
#         from testapp.scheduler_engine import scheduler
#         from testapp.apscheduler_setup import setup_jobs
 
#         if not scheduler.running:
#             setup_jobs()
#             scheduler.start()
        

from django.apps import AppConfig
from django.core.cache import cache

class TestappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'testapp'

    def ready(self):
        import testapp.signals
        from testapp.scheduler_engine import scheduler
        from testapp.apscheduler_setup import setup_jobs

        # Clear cron job statuses from cache on app startup
        cache.delete('comparison_job_status')
        cache.delete('extraction_job_status')

        if not scheduler.running:
            setup_jobs()
            scheduler.start()
