from sqlalchemy import create_engine
from flask import current_app

import db


postgres_base_uri = "postgres://{dbuser}:{dbpass}@{dbhost}/{dbname}"


def kill_all():
    for thing in dir(db):
        if 'close_' in thing:
            getattr(db, thing)()

# ----------------------------
# ----------consumer----------
# ----------------------------

consumer_engine = None
consumer_con = None

def get_consumer_con(user=None):
    global consumer_con
    global consumer_engine

    if not consumer_con:
        consumer_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['CONSUMER_DATABASE_USER'],
            dbpass=current_app.config['CONSUMER_DATABASE_PASS'],
            dbhost=current_app.config['CONSUMER_DATABASE_HOST'],
            dbname=current_app.config['CONSUMER_DATABASE_NAME'],
        ))
        consumer_con = consumer_engine.connect()
    return consumer_engine, consumer_con

def close_consumer_con():
    global consumer_con
    global consumer_engine

    if consumer_con and consumer_engine:
        consumer_con.close()
        consumer_engine.dispose()
        consumer_engine = None
        consumer_con = None


# ----------------------------
# -----------folio------------
# ----------------------------

folio_engine = None
folio_con = None

def get_folio_con(user=None):
    global folio_con
    global folio_engine

    if not folio_con:
        folio_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['FOLIO_DATABASE_USER'],
            dbpass=current_app.config['FOLIO_DATABASE_PASS'],
            dbhost=current_app.config['FOLIO_DATABASE_HOST'],
            dbname=current_app.config['FOLIO_DATABASE_NAME'],
        ))
        folio_con = folio_engine.connect()
    return folio_engine, folio_con

def close_folio_con():
    global folio_con
    global folio_engine

    if folio_con and folio_engine:
        folio_con.close()
        folio_engine.dispose()
        folio_engine = None
        folio_con = None


# ----------------------------
# ----------rolodex-----------
# ----------------------------

rolodex_engine = None
rolodex_con = None

def get_rolodex_con(user=None):
    global rolodex_con
    global rolodex_engine

    if not rolodex_con:
        rolodex_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['ROLODEX_DATABASE_USER'],
            dbpass=current_app.config['ROLODEX_DATABASE_PASS'],
            dbhost=current_app.config['ROLODEX_DATABASE_HOST'],
            dbname=current_app.config['ROLODEX_DATABASE_NAME'],
        ))
        rolodex_con = rolodex_engine.connect()
    return rolodex_engine, rolodex_con

def close_rolodex_con():
    global rolodex_con
    global rolodex_engine

    if rolodex_con and rolodex_engine:
        rolodex_con.close()
        rolodex_engine.dispose()
        rolodex_engine = None
        rolodex_con = None


# ----------------------------
# ----------telegraph-----------
# ----------------------------

telegraph_engine = None
telegraph_con = None

def get_telegraph_con(user=None):
    global telegraph_con
    global telegraph_engine

    if not telegraph_con:
        telegraph_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['TELEGRAPH_DATABASE_USER'],
            dbpass=current_app.config['TELEGRAPH_DATABASE_PASS'],
            dbhost=current_app.config['TELEGRAPH_DATABASE_HOST'],
            dbname=current_app.config['TELEGRAPH_DATABASE_NAME'],
        ))
        telegraph_con = telegraph_engine.connect()
    return telegraph_engine, telegraph_con

def close_telegraph_con():
    global telegraph_con
    global telegraph_engine

    if telegraph_con and telegraph_engine:
        telegraph_con.close()
        telegraph_engine.dispose()
        telegraph_engine = None
        telegraph_con = None


# ----------------------------
# ----------looker-----------
# ----------------------------

looker_engine = None
looker_con = None

def get_looker_con(user=None):
    global looker_con
    global looker_engine

    if not looker_con:
        looker_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['REPORTING_DATABASE_USER'],
            dbpass=current_app.config['REPORTING_DATABASE_PASS'],
            dbhost=current_app.config['REPORTING_DATABASE_HOST'],
            dbname=current_app.config['REPORTING_DATABASE_NAME'],
        ))
        looker_con = looker_engine.connect()
    return looker_engine, looker_con

def close_looker_con():
    global looker_con
    global looker_engine

    if looker_con and looker_engine:
        looker_con.close()
        looker_engine.dispose()
        looker_engine = None
        looker_con = None
