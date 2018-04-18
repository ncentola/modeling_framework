import os


class Config(object):
    DEBUG = False
    TESTING = False
    GP_DATABASE_HOST = os.environ.get('GP_DATABASE_HOST', None)
    GP_DATABASE_NAME = os.environ.get('GP_DATABASE_NAME', None)
    GP_DATABASE_USER = os.environ.get('GP_DATABASE_USER', None)
    GP_DATABASE_PASS = os.environ.get('GP_DATABASE_PASS', None)

class ProductionConfig(Config):
    pass

class StagingConfig(Config):
    DEBUG = os.environ.get('DEBUG', False)

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
