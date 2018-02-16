from flask import Flask, request, render_template
from modeling_data import ModelingData
from data_classes import *
from scorer import Scorer
import json

app = Flask(__name__)

@app.route('/score')
def score():
    application_id = request.args.get('application_id')
    classes = [ApplicantFeatures, AttributionSources, ClarityCbb, ClarityClearfraud, CreditReports, CreditAlerts, FraudFlags, FraudStats, Idology, Iovation, IpAddresses, Neustar, PhoneCalls]
    m = ModelingData(class_list=classes, application_ids = [application_id])
    m.build(model_type='xgboost', target='fraud')

    s = Scorer(model='model.pkl', data=m)
    s.score(calc_contribs=True)

    final_dict = {
        'preds': s.preds.to_json(orient='records'),
        'contribs': s.ordered_contribs.to_json(orient='records')
    }
    return json.dumps(final_dict)
    # return m.modeling_data.to_json(orient='records')
    # return render_template('view.html',tables=[m.modeling_data.to_html()], titles = ['test'])
    # return m.modeling_data

if __name__ == '__main__':
    app.run(debug = True, port = 5000)
