import os
from flask import Flask, request, render_template, jsonify
from modeling_data import ModelingData
from data_classes import *
from scorer import Scorer
from db import get_consumer_con


app = Flask(__name__)
app.config.from_object(os.environ['FLASK_APP_SETTINGS'])

@app.route('/score')
def score():

    '''
    run as a scoring api where some key (application_id for example) is passed in as a query param
    example...

    application_id = request.args.get('application_id')
    application_id = request.args.get('application_id')
    classes = [ApplicantFeatures, AttributionSources, ClarityCbb, ClarityClearfraud, CreditReports, CreditAlerts, FraudFlags, FraudStats, Idology, Iovation, IpAddresses, Neustar, PhoneCalls]
    m = ModelingData(class_list=classes, application_ids = [application_id])
    m.build(target='fraud', debug=app.config['DEBUG'])

    s = Scorer(model='../model.pkl', data=m)
    s.score(calc_contribs=True)

    final_dict = {
        'preds': s.preds.to_dict(orient='records'),
        'contribs': s.ordered_contribs.to_dict(orient='records')
    }
    return jsonify(final_dict)
    '''
    return None

@app.route('/ping')
def ping():
    get_consumer_con()
    return 'OK'

if __name__ == '__main__':
    app.run(debug = True, port = 5000)
