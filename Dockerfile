FROM alpine:3.21 AS build

WORKDIR /usr/src/app
RUN apk add --no-cache python3-dev py3-pip build-base

COPY requirements.txt LICENSE /usr/src/app

RUN python3 -m venv /usr/src/app/.py3env && source /usr/src/app/.py3env/bin/activate && pip install --no-cache-dir --upgrade pip wheel && pip install --no-cache-dir -r requirements.txt

FROM alpine:3.21 AS deploy

EXPOSE 5000
RUN apk add --no-cache uwsgi-python3 python3 libgomp

COPY --from=build /usr/src/app /usr/src/app
COPY uwsgi.ini flask_app.wsgi flask_app.py /usr/src/app
COPY ./libs /usr/src/app/libs

CMD [ "uwsgi", "--ini", "/usr/src/app/uwsgi.ini" ]

# CMD [ "uwsgi", "--socket", "0.0.0.0:5000", \
#                "--uid", "uwsgi", \
#                "--plugins", "python3", \
#                "--protocol", "uwsgi", \
#                "--wsgi", "flask_app:app" ]
# 
# --socket 127.0.0.1:3031 --callable app --wsgi-file myflaskapp.py --processes 4 --threads 2 --stats 127.0.0.1:9191