#!/usr/bin/env python3

from __future__ import division

from flask import Blueprint, jsonify, request, abort
import json
import logging

from . import table

###########################################################################################################
###########################################################################################################

logger = logging.getLogger(__name__)

#OEB_base_url = "https://dev-openebench.bsc.es/api/scientific"
OEB_base_url = "https://dev-openebench.bsc.es/sciapi"

def setOEBScientificServer(new_base_url):
    OEB_base_url = new_base_url
    logger.error(f'OEB Scientific root {OEB_base_url}')

# create blueprint and define url
bp = Blueprint('table', __name__)

@bp.route('/')
def index_page():
    return "<b>FLASK BENCHMARKING EVENT API</b><br><br>\
            USAGE:<br><br> \
            http://webpage:8080/bench_event_id/desired_classification"

@bp.route('/<string:bench_id>')
@bp.route('/<string:bench_id>/<string:classificator_id>', methods = ['POST', 'GET'])
def compute_classification(bench_id, classificator_id="diagonals"):
    if request.method == 'POST':
        challenge_list = request.get_data()
    else:
        challenge_list = []
    
    out = table.get_data(OEB_base_url, bench_id, classificator_id, [])
    if out is None:
        abort(404)
        
    response = jsonify(out)
#    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
