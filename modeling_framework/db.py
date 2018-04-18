from sqlalchemy import create_engine
from flask import current_app
import db

postgres_base_uri = "postgres://{dbuser}:{dbpass}@{dbhost}/{dbname}"

'''
implement get and close methods for any db connections you will need here
'''

# ----------------------------
# ---------GREENPLUM----------
# ----------------------------

my_engine = None
gp_con = None

def get_gp_con(user=None):
    global gp_con
    global gp_engine

    if not gp_con:
        gp_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['GP_DATABASE_USER'],
            dbpass=current_app.config['GP_DATABASE_PASS'],
            dbhost=current_app.config['GP_DATABASE_HOST'],
            dbname=current_app.config['GP_DATABASE_NAME'],
        ))
        gp_con = gp_engine.connect()
    return gp_engine, gp_con

def close_gp_con():
    global gp_con
    global gp_engine

    if gp_con and gp_engine:
        gp_con.close()
        gp_engine.dispose()
        gp_engine = None
        gp_con = None



def kill_all():
    for thing in dir(db):
        if 'close_' in thing:
            getattr(db, thing)()

# ----------------------------
# ----------EXAMPLE----------
# ----------------------------

my_engine = None
my_con = None

def get_my_con(user=None):
    global my_con
    global my_engine

    if not my_con:
        my_engine = create_engine(postgres_base_uri.format(
            dbuser=current_app.config['MY_DATABASE_USER'],
            dbpass=current_app.config['MY_DATABASE_PASS'],
            dbhost=current_app.config['MY_DATABASE_HOST'],
            dbname=current_app.config['MY_DATABASE_NAME'],
        ))
        my_con = my_engine.connect()
    return my_engine, my_con

def close_my_con():
    global my_con
    global my_engine

    if my_con and my_engine:
        my_con.close()
        my_engine.dispose()
        my_engine = None
        my_con = None



def kill_all():
    for thing in dir(db):
        if 'close_' in thing:
            getattr(db, thing)()
