#!/usr/bin/env python3

from __future__ import division

from flask import Blueprint, jsonify, request, abort, current_app
import json
import logging
import urllib.parse

from . import table

###########################################################################################################
###########################################################################################################

logger = logging.getLogger(__name__)

#OEB_base_url = "https://dev-openebench.bsc.es/api/scientific"
DEFAULT_oeb_server = 'dev-openebench.bsc.es'
DEFAULT_oeb_sci_path = '/sciapi'

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
    
    auth_header = request.headers.get('Authorization')
    
    oeb_sci = current_app.config.get("oeb_scientific", {})
    parsed = urllib.parse.urlparse(request.base_url)
    oeb_scheme = 'https'
    if parsed.netloc.startswith('localhost'):
        oeb_server = oeb_sci.get('default_server', DEFAULT_oeb_server)
    else:
        oeb_server = parsed.netloc
        #oeb_scheme = parsed.scheme
    oeb_path = oeb_sci.get('path', DEFAULT_oeb_sci_path)
    
    oeb_base_url = urllib.parse.urlunparse(
        urllib.parse.ParseResult(
            scheme=oeb_scheme,
            netloc=oeb_server,
            path=oeb_path,
            params='',
            query='',
            fragment=''
        )
    )
    
    out = table.get_data(oeb_base_url, auth_header, bench_id, classificator_id, challenge_list)
    if out is None:
        abort(404)
        
    response = jsonify(out)
#    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
