# OpenEBench Benchmarking Event API

API that gets data from the [OpenEBench scientific API](https://openebench.bsc.es/api/scientific/Community.html) for a particular community and benchmarking event and applies a certain results classification method in order to retrieve a 'summary' for that event.

-   **Before running** the Benchmarking Event API, please follow the instructions in [INSTALL.md](INSTALL.md).

*   The API can be run at http://localhost:5000/ in debug mode using the next command line:

```bash
source .pyenv/bin/activate
python flask_app.py
```

-   If you pass any Authorization header to the service, it will be forwarded to internal GraphQL queries, as some additional data could be returned.
-   In order to get data from a specific benchmarking event go to: http://localhost:5000/'<bench_event_id>'/'<desired_classification>' (e.g. http://localhost:5000/OEBE0020000001/squares)
-   This directory holds a WSGI executable, so it can be integrated into an Apache instance. Please follow the instructions of API integration into Apache in [INSTALL.md](INSTALL.md).

## Deployment in production

There is a `flask_app.py.json.*` for each one of the deployments: production, preprod, test1 and test2.

Each one of these deployments needs its own auth file. For instance, a `flask_app.py.json.prod.auth` file is paired with setup at [flask_app.py.json.prod](flask_app.py.json.prod).

The format of the auth files is the next, where the `REALM`, `THEUSER` and `THEPASS` have to be substituted by the legit values:

```json
{
        "authURI": "https://inb.bsc.es/auth/realms/REALM/protocol/openid-connect/token",
        "clientId": "oeb-api-rest",
        "user": "THEUSER",
        "pass": "THEPASS"
}
```

There is a sample file at [flask_app.py.json.auth.template](flask_app.py.json.auth.template).

### nginx reverse proxy setup

Remember to adjust the IP and the port, depending on whether it is being pointed out production (5000), preprod (5001), test1 (5002) or test2 (5003).

```
    location /rest/bench_event_api {
	rewrite /rest/bench_event_api/(.*) /$1 break;
        uwsgi_pass THE_API_IP:5000;
        include uwsgi_params;
    }

```

### Apache reverse proxy setup

**TO BE DONE**