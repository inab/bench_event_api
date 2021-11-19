#!/usr/bin/env python3

from __future__ import absolute_import

import os
import sys
from flask import Flask
from flask_cors import CORS

from libs.routes import bp , setOEBScientificServer

# Creating the object holding the state of the API
if hasattr(sys, 'frozen'):
        basis = sys.executable
else:
        basis = sys.argv[0]

import logging
def initLogging():
    logging.basicConfig(stream=sys.stderr,level=10)

if __name__ == '__main__':
    initLogging()

api_root = os.path.split(basis)[0]
config_file = basis + '.ini'
if os.path.exists(config_file):
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    oeb_scientific_endpoint = cfg.get('config', 'oeb_scientific_endpoint', fallback=None)
    logging.debug(oeb_scientific_endpoint)
    if oeb_scientific_endpoint is not None:
        setOEBScientificServer(oeb_scientific_endpoint)

app = Flask(__name__)
CORS(app)

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True)