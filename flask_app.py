#! /usr/bin/python3.6

from flask import Flask
from table import bp

app = Flask(__name__)
app.register_blueprint(bp)

if __name__ == '__main__':
    app.run()