#!/usr/bin/env python3

from __future__ import division

import logging
from flask import abort
from sklearn.cluster import KMeans
import numpy as np
import pandas
import json
import requests

logger = logging.getLogger(__name__)

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
    
    # Are all values 0?
    if maxX != 0:
        x_norm = [x / maxX for x in x_values]
    else:
        x_norm = list(x_values)
    
    # Are all values 0?
    if maxY != 0:
        means_norm = [y / maxY for y in means]
    else:
        means_norm = list(means)
        
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

    X = np.array(list(zip(x_values, means)))
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
        tools_clusters[name] = int(num + 1)

    return tools_clusters


###########################################################################################################
###########################################################################################################


def build_table(data, classificator_id, tool_names, metrics, challenge_list):

    # this dictionary will store all the information required for the quartiles table
    quartiles_table = []
    if classificator_id == "squares":
        classifier = plot_square_quartiles

    elif classificator_id == "clusters":
        classifier = cluster_tools

    else:
        classifier = plot_diagonal_quartiles

    for challenge in data:
        
        challenge_id = challenge['acronym']
        challenge_OEB_id = challenge['_id']
        
        if len(challenge_list) == 0 or str.encode(challenge_OEB_id) in challenge_list:
            # Metrics categories is optional
            metrics_categories = challenge.get('metrics_categories')
            
            # Skip it!
            if metrics_categories is None:
                # FIXME: Tell something more meaningful
                continue
            
            for metrics_category in metrics_categories:
                # Right now, we are skipping aggregation metrics
                if metrics_category['category'] != 'assessment':
                    continue
                
                # And we are also skipping the cases where we don't have 
                # enough information to do the work
                if len(metrics_category['metrics']) <= 1:
                    continue
                
                for i_metrics_X, metrics_X in enumerate(metrics_category['metrics']):
                    for metrics_Y in metrics_category['metrics'][i_metrics_X+1:]:
                        challenge_X_metric = metrics_X['metrics_id']
                        challenge_Y_metric = metrics_Y['metrics_id']


                        challenge_object = {}
                        tools = {}
                        better = 'top-right'
                        # loop over all assessment datasets and create a dictionary like -> { 'tool': [x_metric, y_metric], ..., ... }
                        for dataset in challenge['datasets']:
                            if dataset['type'] == "assessment":
                                logger.debug(json.dumps(dataset, indent=4))
                                #get tool which this dataset belongs to
                                tool_id = dataset['depends_on']['tool_id']
                                tool_name = tool_names[tool_id]
                                if tool_name not in tools:
                                    tools[tool_name] = [0]*2
                                # get value of the two metrics
                                inline_data = dataset['datalink']['inline_data']
                                if isinstance(inline_data, str):
                                    inline_data = json.loads(inline_data)
                                metric = float(inline_data['value'])
                                if dataset['depends_on']['metrics_id'] == challenge_X_metric:
                                    tools[tool_name][0] = metric
                                elif dataset['depends_on']['metrics_id'] == challenge_Y_metric:
                                    tools[tool_name][1] = metric

                        # get quartiles depending on selected classification method

                        tools_quartiles = classifier(tools, better)

                        challenge_object["_id"] = challenge_OEB_id
                        challenge_object["acronym"] = challenge_id
                        challenge_object['metrics_x'] = metrics[challenge_X_metric]
                        challenge_object['metrics_y'] = metrics[challenge_Y_metric]
                        challenge_object["participants"] = tools_quartiles
                        quartiles_table.append(challenge_object)
    
    return quartiles_table

# Get datasets from given benchmarking event
CHALLENGES_FROM_BE_GRAPHQL = '''query DatasetsFromBenchmarkingEvent($bench_event_id: String) {
    getBenchmarkingEvents(benchmarkingEventFilters:{id: $bench_event_id}) {
        _id
        community_id
    }
    getChallenges(challengeFilters: {benchmarking_event_id: $bench_event_id}) {
        _id
        acronym
        metrics_categories{
            category
            metrics {
                metrics_id
            }
        }
        datasets {
            _id
            datalink{
                inline_data
            }
            depends_on{
                tool_id
                metrics_id
            }
            type
        }
    }
}'''

TOOLS_AND_METRICS_FROM_COMMUNITY_GRAPHQL = '''query ToolsFromCommunity($community_id: String) {
    getTools(toolFilters:{community_id: $community_id}) {
        _id
        name
    }
    getMetrics {
        _id
        orig_id
        title
        description
        representation_hints
        _metadata
    }
}'''

import urllib.request
#import http.client
#
#http.client.HTTPConnection.debuglevel = 1

def get_data(base_url, auth_header, bench_id, classificator_id, challenge_list):
    #logging.getLogger().setLevel(logging.DEBUG)
    #requests_log = logging.getLogger("requests.packages.urllib3")
    #requests_log.setLevel(logging.DEBUG)
    #requests_log.propagate = True
    try:
        url = base_url + "/graphql"
        # get datasets for provided benchmarking event
        query1 = {
            'query': CHALLENGES_FROM_BE_GRAPHQL,
            'variables': {
                'bench_event_id': bench_id
            }
        }
        logger.debug(f"Getting challenges from {bench_id}")
        #data1 = json.dumps(query1,indent=4,sort_keys=True)
        #logger.error(data1)
        #data1b = data1.encode('utf-8')
        #req1 = urllib.request.Request(
        #    url,
        #    data=data1b,
        #    method='POST',
        #    headers={
        #        'Accept': '*/*',
        #        'Content-Type': 'application/json;charset=UTF-8',
        #        'Content-Length': len(data1b)
        #    }
        #)
        #with urllib.request.urlopen(req1) as res1:
        #    resto1 = res1.read()
        #    r1 = resto1.decode('utf-8')
        #    print(r1)
        #    response = json.loads(r1)
        
        common_headers = {
                'Content-Type': 'application/json'
        }
        if auth_header is not None:
            common_headers['Authorization'] = auth_header
        
        r = requests.post(url=url, json=query1, verify=True, headers=common_headers)
        response = r.json()
        if len(response["data"]["getBenchmarkingEvents"]) == 0:
            logger.error(f"{bench_id} not found")
            return None

        data = response["data"]["getChallenges"]
        # get tools for provided benchmarking event
        community_id = response["data"]["getBenchmarkingEvents"][0]["community_id"]
        logger.debug(f'Benchmarking event {bench_id} belongs to community {community_id}')
        
        jsonTM = {
            'query': TOOLS_AND_METRICS_FROM_COMMUNITY_GRAPHQL,
            'variables': {
                'community_id': community_id
            }
        }

        r = requests.post(url=url, json=jsonTM, verify=True, headers=common_headers)
        responseTM = r.json()
        import sys
        json.dump(responseTM, sys.stderr, indent=4)
        tool_list = responseTM["data"]["getTools"]
        if len(tool_list) == 0:
            logger.error(f"Tools for {community_id} not found")
            return None
        metrics_list = responseTM["data"]["getMetrics"]
        if len(metrics_list) == 0:
            logger.error(f"Metrics for {community_id} not found")
            return None
        logger.debug(f'{len(tool_list)} tools and {len(metrics_list)} metrics for {community_id}')
        
        # iterate over the list of tools to generate a dictionary
        tool_names = {}
        for tool in tool_list:
            tool_names[tool["_id"]] = tool["name"]
        
        # And the same for metrics
        metrics = {
            m['_id']: m
            for m in metrics_list
        }
        
        # compute the classification
        result = build_table(data, classificator_id, tool_names, metrics, challenge_list)

        return result

    except Exception as e:
        logger.exception("Unexpected exception")
        abort(500)
