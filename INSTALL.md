# Install instructions of Benchmarking Event API

The source code of this API is written using the [Flask framework](http://flask.pocoo.org/) for Python 3. It depends on standard libreries, plus the ones declared in [requirements.txt](requirements.txt).

-   In order to install the dependencies you need `pip3` and `venv` Python modules. - `pip3` is available in many Linux distributions (Ubuntu package `python3-pip`, CentOS EPEL package `python3-pip`), and also as [pip](https://pip.pypa.io/en/stable/) Python package.

    ```bash
    sudo apt-get install python3-pip python-dev build-essential
    ```

        	- `venv` for Python 3 can be installed with:

    ```bash
    sudo apt-get install python3-venv
    ```

-   The creation of a virtual environment and installation of the dependencies in that environment is done running:

```bash
python3 -m venv .pyenv
source .pyenv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt -c constraints.txt
```

## API deployment
As the API can be used both for development and production, this code looks for the configuration file `flask_app.py.json`
at the very same directory as it is installed. There are a couple of pre-configured files `flask_app.py.json.dev` and `flask_app.py.json.prod`
which can be symlinked to `flask_app.py.json`.

## API integration into Apache

This API can be integrated into an Apache instance. The instance must have the module [WSGI](https://modwsgi.readthedocs.io/en/develop/) installed (package `libapache2-mod-wsgi-py3` in Ubuntu).

```bash
sudo apt-get install libapache2-mod-wsgi-py3
sudo a2enmod wsgi
sudo service apache2 restart
```

```apache config
	<VirtualHost *:80>
		ServerAdmin webmaster@localhost
        DocumentRoot /home/<USERNAME>/public_html

		ErrorLog ${APACHE_LOG_DIR}/error.log
        CustomLog ${APACHE_LOG_DIR}/access.log combined
		
		# Added to ensure Apache forwards authentication headers to Flask
		WSGIPassAuthorization On
		WSGIDaemonProcess flask_app python-path=/path/to/bench_event_api python-home=/path/to/bench_event_api/.pyenv
		WSGIProcessGroup flask_app
		WSGIApplicationGroup %{GLOBAL}
		
		Alias /rest/bench_event_api   /path/to/bench_event_api/flask_app.wsgi/
		<Location /rest/bench_event_api>
			Require all granted
			SetHandler wsgi-script
			Options +ExecCGI
		</Location>

     	ErrorLog ${APACHE_LOG_DIR}/error.log
     	LogLevel warn
     	CustomLog ${APACHE_LOG_DIR}/access.log combined
	</VirtualHost>
```
