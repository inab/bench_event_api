# Benchmarking Event API

API that gets data from the [OpenEBench scientific API](https://openebench.bsc.es/api/scientific/Community.html) for a particular community and benchmarking event and applies a certain results classification method in order to retrieve a 'summary' for that event.
* Before running the Benchmarking Event API, please follow the instructions in [INSTALL.md](INSTALL.md).


* The API can be run at http://localhost:8080/ in debug mode using the next command line:
```bash
source .pyenv/bin/activate
python app.py
```
* In order to get data from a specific benchmarking event go to: http://localhost:8080/'<bench_event_id>'/'<desired_classification>' (e.g. http://localhost:8080/TCGA:2018-04-05/squares)
* This directory holds a FastCGI executable, so it can be integrated into an Apache instance. Please follow the instructions of API integration into Apache in [INSTALL.md](INSTALL.md). 
