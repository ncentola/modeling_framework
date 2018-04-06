import os


class Config(object):
    DEBUG = False
    TESTING = False
    CONSUMER_DATABASE_HOST = os.environ.get('CONSUMER_DATABASE_HOST', None)
    CONSUMER_DATABASE_NAME = os.environ.get('CONSUMER_DATABASE_NAME', 'consumer')
    CONSUMER_DATABASE_USER = os.environ.get('CONSUMER_DATABASE_USER', 'consumer')
    CONSUMER_DATABASE_PASS = os.environ.get('CONSUMER_DATABASE_PASS', None)
    FOLIO_DATABASE_HOST = os.environ.get('FOLIO_DATABASE_HOST', None)
    FOLIO_DATABASE_NAME = os.environ.get('FOLIO_DATABASE_NAME', 'folio')
    FOLIO_DATABASE_USER = os.environ.get('FOLIO_DATABASE_USER', 'folio')
    FOLIO_DATABASE_PASS = os.environ.get('FOLIO_DATABASE_PASS', None)
    ROLODEX_DATABASE_HOST = os.environ.get('ROLODEX_DATABASE_HOST', None)
    ROLODEX_DATABASE_NAME = os.environ.get('ROLODEX_DATABASE_NAME', 'rolodex')
    ROLODEX_DATABASE_USER = os.environ.get('ROLODEX_DATABASE_USER', 'rolodex')
    ROLODEX_DATABASE_PASS = os.environ.get('ROLODEX_DATABASE_PASS', None)
    TELEGRAPH_DATABASE_HOST = os.environ.get('TELEGRAPH_DATABASE_HOST', None)
    TELEGRAPH_DATABASE_NAME = os.environ.get('TELEGRAPH_DATABASE_NAME', 'telegraph')
    TELEGRAPH_DATABASE_USER = os.environ.get('TELEGRAPH_DATABASE_USER', 'telegraph')
    TELEGRAPH_DATABASE_PASS = os.environ.get('TELEGRAPH_DATABASE_PASS', None)
    REPORTING_DATABASE_HOST = os.environ.get('REPORTING_DATABASE_HOST', None)
    REPORTING_DATABASE_NAME = os.environ.get('REPORTING_DATABASE_NAME', 'reporting')
    REPORTING_DATABASE_USER = os.environ.get('REPORTING_DATABASE_USER', 'looker_import')
    REPORTING_DATABASE_PASS = os.environ.get('REPORTING_DATABASE_PASS', None)

class ProductionConfig(Config):
    pass

class StagingConfig(Config):
    DEBUG = os.environ.get('DEBUG', False)

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
