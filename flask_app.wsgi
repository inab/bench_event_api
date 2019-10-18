#! /usr/bin/python3.6

import logging
import sys
activate_this = '/path/to/bench_event_api/.pyenv/bin/activate_this.py'
with open(activate_this) as file_:
  exec(file_.read(), dict(__file__=activate_this))
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/path/to/bench_event_api/')
from flask_app import app as application
