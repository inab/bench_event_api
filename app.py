
from flask import Flask



# create and configure the app

def create_app():

    app = Flask(__name__, instance_relative_config=True)


    #from . import table
    import table
    app.register_blueprint(table.bp)
    app.add_url_rule('/', endpoint='index')

    return app


if __name__ == '__main__':

    
    theApp = create_app()
    theApp.run(host='0.0.0.0', port=8080, debug=False)
