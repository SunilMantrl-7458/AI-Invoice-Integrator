INSTALLED_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'testapp',
    'django_cron',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': ':memory:',
    }
}

USE_TZ = True

SECRET_KEY = 'test-secret-key'

MIDDLEWARE = []
