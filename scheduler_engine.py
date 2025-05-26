from apscheduler.schedulers.background import BackgroundScheduler

extraction_scheduler = BackgroundScheduler()
comparison_scheduler = BackgroundScheduler()

scheduler = BackgroundScheduler()  # Keep existing for backward compatibility if needed
