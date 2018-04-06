import os


class Config(object):
    DEBUG = False
    TESTING = False
    MY_DATABASE_HOST = os.environ.get('MY_DATABASE_HOST', None)
    MY_DATABASE_NAME = os.environ.get('MY_DATABASE_NAME', None)
    MY_DATABASE_USER = os.environ.get('MY_DATABASE_USER', None)
    MY_DATABASE_PASS = os.environ.get('MY_DATABASE_PASS', None)

class ProductionConfig(Config):
    pass

class StagingConfig(Config):
    DEBUG = os.environ.get('DEBUG', False)

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
