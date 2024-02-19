#!/usr/bin/env python3

from __future__ import absolute_import

import json
import os
import sys
from flask import Flask
from flask_cors import CORS

from libs.routes import bp

# Creating the object holding the state of the API
if hasattr(sys, 'frozen'):
        basis = sys.executable
else:
        basis = sys.argv[0]

# It is being called by mod_wsgi
if basis == 'mod_wsgi':
    basis = __file__

import logging
def initLogging() -> "None":
    logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)

api_root = os.path.split(basis)[0]

app = Flask(__name__)

config_file = basis + '.json'
if os.path.exists(config_file):
    with open(config_file, mode="r", encoding="utf-8") as cF:
        app.config.update(json.load(cF))

auth_config_file = config_file + '.auth'
if os.path.exists(auth_config_file):
    with open(auth_config_file, mode="r", encoding="utf-8") as cF:
        app.config.update({"oeb_auth": json.load(cF)})
else:
    logging.critical(f"The file with the auth setup {auth_config_file} must exist, in order to be granted access to TestActions")

CORS(app)

app.register_blueprint(bp)

if __name__ == '__main__':
    initLogging()
    app.run(debug=True)