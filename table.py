#! /usr/bin/python3.6
from __future__ import division
from flask import (
    Blueprint, jsonify, request
)
from sklearn.cluster import KMeans
import numpy as np
import pandas
import json
import requests


# funtion that gets quartiles for x and y values
def plot_square_quartiles(tools_dict, better, percentile=50):

    # generate 3 lists: 
    x_values = []
    means = []
    tools = []
    for key, metrics in tools_dict.items():
        tools.append(key)
        x_values.append(metrics[0])
        means.append(metrics[1])

    x_percentile, y_percentile = (np.nanpercentile(x_values, percentile), np.nanpercentile(means, percentile))

    # create a dictionary with tools and their corresponding quartile
    tools_quartiles = {}
    if better == "bottom-right":
        for i, val in enumerate(tools, 0):
            if x_values[i] >= x_percentile and means[i] <= y_percentile:
                tools_quartiles[tools[i]] = 1
            elif x_values[i] >= x_percentile and means[i] > y_percentile:
                tools_quartiles[tools[i]] = 3
            elif x_values[i] < x_percentile and means[i] > y_percentile:
                tools_quartiles[tools[i]] = 4
            elif x_values[i] < x_percentile and means[i] <= y_percentile:
                tools_quartiles[tools[i]] = 2
    elif better == "top-right":
        for i, val in enumerate(tools, 0):
            if x_values[i] >= x_percentile and means[i] < y_percentile:
                tools_quartiles[tools[i]] = 3
            elif x_values[i] >= x_percentile and means[i] >= y_percentile:
                tools_quartiles[tools[i]] = 1
            elif x_values[i] < x_percentile and means[i] >= y_percentile:
                tools_quartiles[tools[i]] = 2
            elif x_values[i] < x_percentile and means[i] < y_percentile:
                tools_quartiles[tools[i]] = 4
    return (tools_quartiles)


# function to normalize the x and y axis to 0-1 range
def normalize_data(x_values, means):
    maxX = max(x_values)
    maxY = max(means)

    x_norm = [x / maxX for x in x_values]
    means_norm = [y / maxY for y in means]
    return x_norm, means_norm


# funtion that splits the analysed tools into four quartiles, according to the asigned score
def get_quartile_points(scores_and_values, first_quartile, second_quartile, third_quartile):
    tools_quartiles = {}
    for i, val in enumerate(scores_and_values, 0):
        if scores_and_values[i][0] > third_quartile:
            tools_quartiles[scores_and_values[i][3]] = 1
        elif second_quartile < scores_and_values[i][0] <= third_quartile:
            tools_quartiles[scores_and_values[i][3]] = 2
        elif first_quartile < scores_and_values[i][0] <= second_quartile:
            tools_quartiles[scores_and_values[i][3]] = 3
        elif scores_and_values[i][0] <= first_quartile:
            tools_quartiles[scores_and_values[i][3]] = 4

    return (tools_quartiles)


# funtion that separate the points through diagonal quartiles based on the distance to the 'best corner'
def plot_diagonal_quartiles( tools_dict, better):

    # generate 3 lists: 
    x_values = []
    means = []
    tools = []
    for key, metrics in tools_dict.items():
        tools.append(key)
        x_values.append(metrics[0])
        means.append(metrics[1])

    # normalize data to 0-1 range
    x_norm, means_norm = normalize_data(x_values, means)

    # compute the scores for each of the tool. based on their distance to the x and y axis
    scores = []
    for i, val in enumerate(x_norm, 0):
        if better == "bottom-right":
            scores.append(x_norm[i] + (1 - means_norm[i]))
        elif better == "top-right":
            scores.append(x_norm[i] + means_norm[i])

    # region sort the list in descending order
    scores_and_values = sorted([[scores[i], x_values[i], means[i], tools[i]] for i, val in enumerate(scores, 0)],
                               reverse=True)
    scores = sorted(scores, reverse=True)

    first_quartile, second_quartile, third_quartile = (
        np.nanpercentile(scores, 25), np.nanpercentile(scores, 50), np.nanpercentile(scores, 75))

    # split in quartiles
    tools_quartiles = get_quartile_points(scores_and_values, first_quartile, second_quartile, third_quartile)

    return (tools_quartiles)


# function that clusters participants using the k-means algorithm
def cluster_tools(tools_dict, better):

    # generate 3 lists: 
    x_values = []
    means = []
    tools = []
    for key, metrics in tools_dict.items():
        tools.append(key)
        x_values.append(metrics[0])
        means.append(metrics[1])


    X = np.array(zip(x_values, means))
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=0).fit(X)

    cluster_no = kmeans.labels_

    centroids = kmeans.cluster_centers_

    # normalize data to 0-1 range
    x_values = []
    y_values = []
    for centroid in centroids:
        x_values.append(centroid[0])
        y_values.append(centroid[1])
    x_norm, y_norm = normalize_data(x_values, y_values)

    # get distance from centroids to better corner
    distances = []
    if better == "top-right":

        for x, y in zip(x_norm, y_norm):
            distances.append(x + y)

    elif better == "bottom-right":

        for x, y in zip(x_norm, y_norm):
            distances.append(x + (1 - y))

    # assign ranking to distances array
    output = [0] * len(distances)
    for i, x in enumerate(sorted(range(len(distances)), key=lambda y: distances[y], reverse=True)):
        output[x] = i

    # reorder the clusters according to distance
    for i, val in enumerate(cluster_no):
        for y, num in enumerate(output):
            if val == y:
                cluster_no[i] = num

    tools_clusters = {}
    for (x, y), num, name in zip(X, cluster_no, tools):
        tools_clusters[name] = num + 1

    return tools_clusters


###########################################################################################################
###########################################################################################################


def build_table(data, classificator_id, tool_names, challenge_list):

    # this dictionary will store all the information required for the quartiles table
    quartiles_table = []

    for challenge in data:
        
        challenge_id = challenge['acronym']
        challenge_OEB_id = challenge['_id']
        challenge_X_metric = challenge['metrics_categories'][0]['metrics'][0]['metrics_id']
        challenge_Y_metric = challenge['metrics_categories'][0]['metrics'][1]['metrics_id']

        if challenge_list == [] or str.encode(challenge_OEB_id) in challenge_list:

            challenge_object = {}
            tools = {}
            better = 'top-right'
            # loop over all assessment datasets and create a dictionary like -> { 'tool': [x_metric, y_metric], ..., ... }
            for dataset in challenge['datasets']:
                if dataset['type'] == "assessment":
                    #get tool which this dataset belongs to
                    tool_id = dataset['depends_on']['tool_id']
                    tool_name = tool_names[tool_id]
                    if tool_name not in tools:
                        tools[tool_name] = [0]*2
                    # get value of the two metrics
                    metric = float(dataset['datalink']['inline_data']['value'])
                    if dataset['depends_on']['metrics_id'] == challenge_X_metric:
                        tools[tool_name][0] = metric
                    elif dataset['depends_on']['metrics_id'] == challenge_Y_metric:
                        tools[tool_name][1] = metric

            # get quartiles depending on selected classification method

            if classificator_id == "squares":
                tools_quartiles = plot_square_quartiles(tools, better)

            elif classificator_id == "clusters":
                tools_quartiles = cluster_tools(tools, better)

            else:
                tools_quartiles = plot_diagonal_quartiles( tools, better)

            challenge_object["_id"] = challenge_OEB_id
            challenge_object["acronym"] = challenge_id
            challenge_object["participants"] = tools_quartiles
            quartiles_table.append(challenge_object)
    
    return quartiles_table

def get_data(base_url, bench_id, classificator_id, challenge_list):
    try:
        url = base_url + "sciapi/graphql"
        # get datasets for provided benchmarking event
        json = { 'query' : '{\
                                getBenchmarkingEvents(benchmarkingEventFilters:{id:"'+ bench_id + '"}) {\
                                    _id\
                                    community_id\
                                }\
                                getChallenges(challengeFilters: {benchmarking_event_id: "'+ bench_id + '"}) {\
                                    _id\
                                    acronym\
                                    metrics_categories{\
                                        metrics {\
                                            metrics_id\
                                        }\
                                    }\
                                    datasets {\
                                        _id\
                                        datalink{\
                                            inline_data\
                                        }\
                                        depends_on{\
                                            tool_id\
                                            metrics_id\
                                        }\
                                        type\
                                    }\
                                }\
                            }' }

        r = requests.post(url=url, json=json, verify=False )
        response = r.json()
        if response["data"]["getBenchmarkingEvents"] == []:

            return { 'data': None}

        else:
            data = response["data"]["getChallenges"]
            # get tools for provided benchmarking event
            community_id = response["data"]["getBenchmarkingEvents"][0]["community_id"]
            json2 = { 'query' : '{\
                                    getTools(toolFilters:{community_id:"'+ community_id + '"}) {\
                                        _id\
                                        name\
                                    }\
                                }' }

            r = requests.post(url=url, json=json2, verify=False )
            response2 = r.json()
            tool_list = response2["data"]["getTools"]
            # iterate over the list of tools to generate a dictionary
            tool_names = {}
            for tool in tool_list:
                tool_names[tool["_id"]] = tool["name"]

            # compute the classification
            result = build_table(data, classificator_id, tool_names, challenge_list)

            return result

    except Exception as e:

        print (e)


# create blueprint and define url
bp = Blueprint('table', __name__)



@bp.route('/')
def index_page():
    return "<b>FLASK BENCHMARKING EVENT API</b><br><br>\
            USAGE:<br><br> \
            http://webpage:8080/bench_event_id/desired_classification"

@bp.route('/<string:bench_id>')
@bp.route('/<string:bench_id>/<string:classificator_id>', methods = ['POST', 'GET'])
def compute_classification(bench_id, classificator_id="diagonals"):
	
    mode = "dev"
    if mode == "production":
	    base_url = "https://openebench.bsc.es/"
    else:
	    base_url = "https://dev-openebench.bsc.es/"
    
    if request.method == 'POST':
        challenge_list = request.get_data()
        out = get_data(base_url, bench_id, classificator_id, challenge_list)
        response = jsonify(out)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    else:
        out = get_data(base_url, bench_id, classificator_id, [])
        response = jsonify(out)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        # return send_from_directory("/home/jgarrayo/public_html/flask_table/", "table.svg", as_attachment=False)
        # return send_file(out, mimetype='svg')
        # return render_template('index.html', data=out)

# if __name__ == "__main__":
#     app.run()
