#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# coding: utf-8

import sys, os

from flup.server.fcgi import WSGIServer

# Creating the object holding the state of the API
if hasattr(sys, 'frozen'):
	basis = sys.executable
else:
	basis = sys.argv[0]

if __name__ == '__main__':
    import app
    theApp = app.create_app()
    WSGIServer(theApp).run()
