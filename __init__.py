
from flask import Flask



# create and configure the app

def create_app():

    app = Flask(__name__, instance_relative_config=True)


    from . import table
    app.register_blueprint(table.bp)
    app.add_url_rule('/', endpoint='index')

    return app


if __name__ == '__main__':

    app = create_app()
    app.run()
