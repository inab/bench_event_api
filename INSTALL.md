# Install instructions of Benchmarking Event API

The source code of this API is written using the [Flask framework](http://flask.pocoo.org/) for Python 2.7. It depends on standard libreries, plus the ones declared in [requirements.txt](requirements.txt).

* In order to install the dependencies you need `pip` and `venv` Python modules.
	- `pip` is available in many Linux distributions (Ubuntu package `python-pip`, CentOS EPEL package `python-pip`), and also as [pip](https://pip.pypa.io/en/stable/) Python package.
    ```bash
    sudo apt-get install python-pip python-dev build-essential
    sudo pip install virtualenv virtualenvwrapper
    ```
	- `venv` for Python 2 can be installed with:
    ```bash
    sudo pip install virtualenv virtualenvwrapper
    ```

* The creation of a virtual environment and installation of the dependencies in that environment is done running:

```bash
virtualenv -p python2 .pyenv
source .pyenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -c constraints.txt
```

## API integration into Apache

This API can be integrated into an Apache instance. The instance must have the module [FCGID](https://httpd.apache.org/mod_fcgid/) installed (package `libapache2-mod-fcgid` in Ubuntu).

```bash
sudo apt install apache2 libapache2-mod-fcgid
sudo a2enmod mod-fcgid
sudo service apache2 restart
sudo service apache2 enable
```

```apache
	FcgidMaxProcessesPerClass	5
	ScriptAlias / "/path/to/bench_event_api.fcgi/"

	<Location />
		SetHandler fcgid-script
		Options +ExecCGI
		Require all granted
	</Location>
```
