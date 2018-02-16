from sqlalchemy import create_engine
import db
'''
Define your db connections here. Each database you need to connect to should have a get and close method

EXAMPLE:

def get_db_con(user):
    global db_con
    global db_engine

    if not db_con:
        db_engine = create_engine('postgres://' + user + '@db-ro.prod.iloan.com/db')
        db_con = db_engine.connect()
    return db_engine, db_con

def close_db_con():
    global db_con
    global db_engine

    if db_con and db_engine:
        db_con.close()
        db_engine.dispose()
        db_engine = None
        db_con = None

'''

def kill_all():
    for thing in dir(db):
        if 'close_' in thing:
            getattr(db, thing)()
