#!/usr/bin/env python3

from __future__ import absolute_import

import json
import os
import sys
from flask import Flask
from flask_cors import CORS

from libs.routes import bp

import logging
def initLogging() -> "None":
    logging.basicConfig(stream=sys.stderr,level=logging.DEBUG)

def initApp() -> "Flask":
    # Creating the object holding the state of the API
    if hasattr(sys, 'frozen'):
            basis = sys.executable
    else:
            basis = sys.argv[0]

    # It is being called by either mod_wsgi or uwsgi
    if basis in ('mod_wsgi', 'uwsgi'):
        basis = __file__

    api_root = os.path.split(basis)[0]

    app = Flask(__name__)

    config_file = sys.argv[1] if len(sys.argv)>1 else basis + '.json'
    if os.path.exists(config_file):
        with open(config_file, mode="r", encoding="utf-8") as cF:
            app.config.update(json.load(cF))
    else:
        logging.warning(f"Configuration file {config_file} was not found (or it is not readable)")

    auth_config_file = config_file + '.auth'
    if os.path.exists(auth_config_file):
        with open(auth_config_file, mode="r", encoding="utf-8") as cF:
            app.config.update({"oeb_auth": json.load(cF)})
    else:
        logging.critical(f"The file with the auth setup {auth_config_file} must exist, in order to be granted access to TestActions")

    CORS(app)

    app.register_blueprint(bp)
    
    return app


if __name__ == '__main__':
    initLogging()
    app = initApp()
    app.run(debug=True)
else:
    app = initApp()