#!/usr/bin/env python3

from __future__ import division

from typing import (
    cast,
    TYPE_CHECKING,
)
if TYPE_CHECKING:
    from typing import (
        Optional,
        Sequence,
    )
    
    from flask import (
        Response,
    )

from flask import Blueprint, jsonify, request, abort, current_app
import json
import logging
import urllib.parse

from . import table
from . import auth

###########################################################################################################
###########################################################################################################

logger = logging.getLogger(__name__)

#OEB_base_url = "https://dev-openebench.bsc.es/api/scientific"
DEFAULT_oeb_server = 'dev-openebench.bsc.es'
DEFAULT_oeb_sci_path = '/sciapi'

# create blueprint and define url
bp = Blueprint('table', __name__)

@bp.route('/')
def index_page() -> "str":
    return "<b>FLASK BENCHMARKING EVENT API</b><br><br>\
            USAGE:<br><br> \
            http://webpage:5000/{bench_event_id}/{desired_classification}"

@bp.route('/<string:bench_id>')
@bp.route('/<string:bench_id>/<string:classificator_id>', methods = ['POST', 'GET'])
@bp.route('/<string:bench_id>/<string:classificator_id>/<string:challenge_id>')
def compute_classification(bench_id: "str", classificator_id: "Optional[str]" = "diagonals", challenge_id: "Optional[str]" = None) -> "Response":
    if request.method == 'POST':
        challenge_list_t = request.get_json()
        challenge_list = cast("Sequence[str]", challenge_list_t) if isinstance(challenge_list_t, list) else []
    else:
        challenge_list = []
        if challenge_id is not None:
            challenge_list.append(challenge_id)
    
    auth_header = request.headers.get('Authorization')
    
    oeb_sci = current_app.config.get("oeb_scientific", {})
    parsed = urllib.parse.urlparse(request.base_url)
    oeb_scheme = 'https'
    if parsed.netloc.startswith('localhost') or parsed.netloc.startswith('127.0.0.1'):
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
    
    if auth_header is None:
        oeb_auth = current_app.config.get("oeb_auth")
        if isinstance(oeb_auth, dict):
            try:
                access_token = auth.getAccessToken(oeb_auth)
                auth_header = "Bearer " + access_token
            except:
                logger.exception("Failed to default auth")
    
    out = table.get_data(oeb_base_url, auth_header, bench_id, classificator_id, challenge_list)
    if out is None:
        abort(404)
        
    response = jsonify(out)
#    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
