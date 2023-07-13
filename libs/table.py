#!/usr/bin/env python3

from __future__ import division

import logging
from flask import abort
from sklearn.cluster import KMeans
import statistics
import numpy as np
import pandas
import json
import requests

logger = logging.getLogger(__name__)

# funtion that gets quartiles for x and y values
def plot_square_quartiles(tools_dict, better, percentile=50):

    # generate 2 lists: 
    the_values = [] 
    tools = []
    dims = None
    for key, metrics in tools_dict.items():
        tools.append(key)
        the_values.append(metrics)
        if dims is None:
            dims = len(metrics)
    
    dim_percentiles = tuple(np.nanpercentile(col_vals, percentile) for col_vals in zip(*the_values))

    dimcomp = None
    if dims == 1:
        if better.endswith("right"):
            dimcomp = [ True ]
        else:
            dimcomp = [ False ]
    elif dims == 2:
        if better == "top-right":
            dimcomp = [ True, True ]
        elif better == "bottom-right":
            # bottom => y-axis
            dimcomp = [ True, False ]
        elif better == "top-left":
            # left => x-axis
            dimcomp = [ False, True ]
        elif better == "bottom-left":
            dimcomp = [ False, False ]
    else:
        dimcomp = dims * [ True ]
    
    # Create a dictionary with tools and their corresponding quartile
    # Due the changes in dimensionality, for two dimensions are
    # 4 possible square quartiles, for three dimensions are 8 cubic quartiles,
    # for four dimensions are 16 hypercubic quartiles, and so on
    tools_quartiles = {}
    for tool, t_values in zip(tools, the_values):
        t_quart = 0
        for d_comp, t_value, d_percentile in zip(reversed(dimcomp), reversed(t_values), reversed(dim_percentiles)):
            t_quart <<= 1
            if d_comp:
                if t_value < d_percentile:
                    t_quart |= 1
            elif t_value >= d_percentile:
                t_quart |= 1
            
        
        tools_quartiles[tool] = t_quart + 1
    return tools_quartiles


# function to normalize the x and y axis to 0-1 range
def normalize_series(values: "Sequence[float]") -> "Sequence[float]":
    maxV = max(values)
    
    # Are all values 0?
    if maxV != 0:
        v_norm = [v / maxV for v in values]
    else:
        v_norm = list(values)
        
    return v_norm

def normalize_data(args: "Sequence[Sequence[float]]") -> "Tuple[Tuple[float, ...], ...]":
    return tuple(zip(*[normalize_series(zipped_arg) for zipped_arg in zip(*args)]))


# funtion that splits the analysed tools into four quartiles, according to the asigned score
def get_quartile_points(scores_and_values):
    scores = list(map(lambda s: s[0], scores_and_values))

    first_quartile, second_quartile, third_quartile = (
        np.nanpercentile(scores, 25), np.nanpercentile(scores, 50), np.nanpercentile(scores, 75))

    tools_quartiles = {}
    for i, scores_and_value in enumerate(scores_and_values):
        sv = scores_and_value[0]
        if sv > third_quartile:
            quar_number = 1
        elif second_quartile < sv <= third_quartile:
            quar_number = 2
        elif first_quartile < sv <= second_quartile:
            quar_number = 3
        elif sv <= first_quartile:
            quar_number = 4
        else:
            # Should never happen
            quar_number = 0
        tools_quartiles[scores_and_value[2]] = quar_number

    return tools_quartiles


# funtion that separate the points through diagonal quartiles based on the distance to the 'best corner'
def plot_diagonal_quartiles(tools_dict: "Mapping[str,Sequence[float]]", better: "str"):

    # generate 2 lists: 
    the_values = [] 
    tools = []
    dims = None
    for key, metrics in tools_dict.items():
        tools.append(key)
        the_values.append(metrics)
        if dims is None:
            dims = len(metrics)
    
    # normalize data to 0-1 range
    norm_values = normalize_data(the_values)

    # compute the scores for each of the tool. based on their distance to the x and y axis
    dimcorr = None
    if dims == 1:
        if better.endswith("right"):
            dimcorr = None
        else:
            dimcorr = [ True ]
    elif dims == 2:
        if better == "top-right":
            dimcorr = None
        elif better == "bottom-right":
            # bottom => y-axis
            dimcorr = [ False, True ]
        elif better == "top-left":
            # left => x-axis
            dimcorr = [ True, False ]
        elif better == "bottom-left":
            dimcorr = [ True, True ]

    scores = []
    for norm_value in norm_values:
        if dimcorr is None:
            corr_norm_value = norm_value
        else:
            corr_norm_value = tuple(map(lambda dc_norm: 1.0 - dc_norm[1] if dc_norm[0] else dc_norm[1], zip(dimcorr, norm_value)))
        score = sum(corr_norm_value)
        scores.append(score)

    # region sort the list in descending order
    scores_and_values = sorted(zip(scores, the_values, tools), reverse=True)
    # split in quartiles
    tools_quartiles = get_quartile_points(scores_and_values)

    return tools_quartiles


# function that clusters participants using the k-means algorithm
def cluster_tools(tools_dict, better):
    # generate 2 lists:
    the_values = [] 
    tools = []
    dims = None
    for key, metrics in tools_dict.items():
        tools.append(key)
        the_values.append(metrics)
        if dims is None:
            dims = len(metrics)
    
    kmeans = KMeans(n_clusters=4, n_init=50, random_state=0).fit(the_values)

    cluster_no = kmeans.labels_

    centroids = kmeans.cluster_centers_.tolist()

    # normalize data to 0-1 range
    
    norm_centroids = normalize_data(centroids)

    # get distance from centroids to better corner
    distances = []
    dimcorr = None
    if dims == 1:
        if better.endswith("right"):
            dimcorr = None
        else:
            dimcorr = [ True ]
    elif dims == 2:
        if better == "top-right":
            dimcorr = None
        elif better == "bottom-right":
            dimcorr = [ False, True ]
        elif better == "top-left":
            dimcorr = [ True, False ]
        elif better == "bottom-left":
            dimcorr = [ True, True ]
    
    for norm_centroid in norm_centroids:
        if dimcorr is None:
            corr_norm_centroid = norm_centroid
        else:
            corr_norm_centroid = tuple(map(lambda dc_norm: 1.0 - dc_norm[1] if dc_norm[0] else dc_norm[1], zip(dimcorr, norm_centroid)))
        distance = sum(corr_norm_centroid)
        distances.append(distance)

    # assign ranking to distances array
    output = [0] * len(distances)
    for i, x in enumerate(sorted(range(len(distances)), key=lambda y: distances[y], reverse=True)):
        output[x] = i

    # reorder the clusters according to distance
    for i, val in enumerate(cluster_no):
        for y, num in enumerate(output):
            if val == y:
                cluster_no[i] = num

    tools_clusters = {
        name: int(num + 1)
        for num, name in zip(cluster_no, tools)
    }

    return tools_clusters


###########################################################################################################
###########################################################################################################


def build_table(data, classificator_id, tool_names, metrics: "Mapping[str, Mapping[str, Any]]", challenge_list):

    # this dictionary will store all the information required for the quartiles table
    quartiles_table = []
    if classificator_id == "squares":
        classifier = plot_square_quartiles

    elif classificator_id == "clusters":
        classifier = cluster_tools

    elif classificator_id is None or (classificator_id in ("diagonals", "")):
        classifier = plot_diagonal_quartiles
    else:
        raise KeyError(f"'{classificator_id}' is not a valid classificator")

    for challenge in data:
        challenge_OEB_id = challenge['_id']
        
        if len(challenge_list) == 0 or (challenge_OEB_id in challenge_list):
            challenge_metadata = challenge.get("_metadata")
            challenge_id = None
            if challenge_metadata is not None:
                challenge_metadata = json.loads(challenge_metadata)
                challenge_id = challenge_metadata.get("level_2:challenge_id")
            if challenge_id is None:
                challenge_id = challenge['acronym']
            # Metrics categories is optional
            metrics_categories = challenge.get('metrics_categories')
            
            # Skip it!
            if not isinstance(metrics_categories, list):
                # FIXME: Tell something more meaningful
                logger.error(f"FIXME {challenge_OEB_id}")
                continue
            
            # Enumerate the different datasets
            participant_datasets = {
                participant_dataset["_id"]: participant_dataset
                for participant_dataset in challenge["participant_datasets"]
            }
            assessment_datasets = {
                assessment_dataset["_id"]: assessment_dataset
                for assessment_dataset in challenge["assessment_datasets"]
            }
            aggregation_datasets = {
                aggregation_dataset["_id"]: aggregation_dataset
                for aggregation_dataset in challenge["aggregation_datasets"]
            }
            
            participant_from_assessment = dict()
            for m_event in challenge["metrics_test_actions"]:
                part_datasets = []
                ass_datasets = []
                for i_dataset in m_event["involved_datasets"]:
                    # We are capturing the assessment dataset
                    if i_dataset["role"] == "outgoing":
                        pos_ass_dataset = assessment_datasets.get(i_dataset["dataset_id"])
                        if pos_ass_dataset is not None:
                            ass_datasets.append(pos_ass_dataset)
                    elif i_dataset["role"] == "incoming":
                        # We are capturing the participant dataset
                        pos_part_dataset = participant_datasets.get(i_dataset["dataset_id"])
                        if pos_part_dataset is not None:
                            part_datasets.append(pos_part_dataset)
                if len(part_datasets) > 0 and len(ass_datasets) > 0:
                    for part_dataset in part_datasets:
                        for ass_dataset in ass_datasets:
                            participant_from_assessment[ass_dataset["_id"]] = part_dataset
            
            if len(challenge["aggregation_test_actions"]) > 0:
                # First, let's dig in the assessment metrics to
                # build a map from metrics label to metrics entry
                metrics_by_label = dict()
                for m_cat in metrics_categories:
                    if m_cat['category'] == 'assessment':
                        for m_pair in m_cat["metrics"]:
                            # Now, the entry
                            the_metrics = metrics[m_pair["metrics_id"]]
                            metrics_metadata = the_metrics.get("_metadata")
                            metrics_labels = []
                            # Metrics labels, ordered by precedence
                            if metrics_metadata is not None:
                                metrics_label = metrics_metadata.get("level_2:metric_id")
                                metrics_labels.append(metrics_label)
                            
                            _ , metrics_label = the_metrics["orig_id"].split(":" , 1)
                            if metrics_label is not None:
                                metrics_labels.append(metrics_label)
                            
                            # Last chance
                            metrics_labels.append(the_metrics["orig_id"])
                            
                            for metrics_label in metrics_labels:
                                metrics_by_label.setdefault(metrics_label, []).append(the_metrics)
                    #    continue
                    
                
                # Let's chain everything together
                for agg_event in challenge["aggregation_test_actions"]:
                    agg_dataset = None
                    inline_data = None
                    
                    challenge_X_metric_label = None
                    challenge_Y_metric_label = None
                    
                    challenge_X_metric_entry = None
                    challenge_Y_metric_entry = None
                    
                    challenge_metric_labels = []
                    challenge_metric_entries = []
                    
                    better = None
                    ass_part_datasets = []
                    tools = {}
                    agg_label = challenge_id
                    for i_dataset in agg_event["involved_datasets"]:
                        # We are capturing the aggregation dataset
                        if i_dataset["role"] == "outgoing":
                            # Filtering out aggregations which are not 2D-plot
                            agg_dataset = aggregation_datasets.get(i_dataset["dataset_id"])
                            
                            if agg_dataset is None:
                                break
                            
                            inline_data = agg_dataset.get("datalink", {}).get("inline_data")
                            # Filtering out the ones without inline data
                            if inline_data is None:
                                agg_dataset = None
                                break
                            if isinstance(inline_data, str):
                                inline_data = json.loads(inline_data)
                            # Filtering out the ones which are not 2D-plots
                            visualization = inline_data.get("visualization", {})
                            vis_type = visualization.get("type")
                            if vis_type == "2D-plot":
                                challenge_X_metric_label = visualization["x_axis"]
                                challenge_X_metric_entry = metrics_by_label.get(challenge_X_metric_label)
                                if isinstance(challenge_X_metric_entry, list):
                                    challenge_X_metric_entry = challenge_X_metric_entry[0]
                                
                                challenge_Y_metric_label = visualization["y_axis"]
                                challenge_Y_metric_entry = metrics_by_label.get(challenge_Y_metric_label)
                                if isinstance(challenge_Y_metric_entry, list):
                                    challenge_Y_metric_entry = challenge_Y_metric_entry[0]
                                
                                challenge_metric_labels.append(challenge_X_metric_label)
                                challenge_metric_labels.append(challenge_Y_metric_label)
                                
                                challenge_metric_entries.append(challenge_X_metric_entry)
                                challenge_metric_entries.append(challenge_Y_metric_entry)
                                
                                better = visualization.get("optimization", "top-right")
                                
                                # Now, the values
                                for cha_par in inline_data["challenge_participants"]:
                                    tools[cha_par["tool_id"]] = [ cha_par["metric_x"], cha_par["metric_y"] ]
                            elif vis_type == "bar-plot":
                                challenge_metric_label = visualization["metric"]
                                challenge_metric_entry = metrics_by_label.get(challenge_metric_label)
                                if isinstance(challenge_metric_entry, list):
                                    challenge_metric_entry = challenge_metric_entry[0]
                                
                                challenge_metric_labels.append(challenge_metric_label)
                                challenge_metric_entries.append(challenge_metric_entry)
                                
                                # TODO: implement this???
                                better = None
                                
                                # Now, the values
                                for cha_par in inline_data["challenge_participants"]:
                                    tools[cha_par["tool_id"]] = [ cha_par["metric_value"] ]
                            elif vis_type == "box-plot":
                                # It could be more than one
                                metric_label_pos = dict()
                                for i_label, challenge_metric_label in enumerate(visualization["available_metrics"]):
                                    challenge_metric_entry = metrics_by_label.get(challenge_metric_label)
                                    if isinstance(challenge_metric_entry, list):
                                        challenge_metric_entry = challenge_metric_entry[0]
                                    
                                    challenge_metric_labels.append(challenge_metric_label)
                                    challenge_metric_entries.append(challenge_metric_entry)
                                    metric_label_pos[challenge_metric_label] = i_label
                                
                                # TODO: implement this???
                                better = None
                                
                                # Now, the values
                                failed = False
                                for cha_par in inline_data["challenge_participants"]:
                                    participant_label = cha_par["label"]
                                    means = tools.get(participant_label)
                                    if means is None:
                                        means = len(challenge_metric_labels) * [0.0]
                                        tools[participant_label] = means
                                    
                                    metrics_label = cha_par["metric_id"]
                                    # Detecting ill dataset cases
                                    if metrics_label not in metric_label_pos:
                                        failed = True
                                        logger.error(f"Dataset {agg_dataset['_id']} has an ill inline box-plot dataset, as metrics label {metrics_label} matches no valid metric")
                                        break
                                    means[metric_label_pos[metrics_label]] = statistics.mean(
                                        map(
                                            lambda v:
                                            v if not isinstance(v, dict) else v["v"]
                                            ,
                                            cha_par["values"]
                                        )
                                    )
                                # Ill box-plot dataset
                                if failed:
                                    agg_dataset = None
                            else:
                                agg_dataset = None
                            
                            agg_label += "<br/>" + "<br/>vs<br/>".join(challenge_metric_labels)
                            if agg_dataset is None:
                                break
                        elif i_dataset["role"] == "incoming":
                            # Getting the participant dataset from the assessment dataset
                            ass_dataset = assessment_datasets.get(i_dataset["dataset_id"])
                            if ass_dataset is not None:
                                part_dataset = participant_from_assessment.get(i_dataset["dataset_id"])
                                if part_dataset is not None:
                                    ass_part_datasets.append((ass_dataset, part_dataset))
                    
                    if agg_dataset is not None and len(ass_part_datasets) > 0:
                        # Default value
                        if better is None:
                            better = "top-right"
                        tools_quartiles = classifier(tools, better)
                        challenge_object = {
                            "_id": challenge_OEB_id,
                            "aggregation_id": agg_dataset["_id"],
                            "acronym": agg_label,
                            'metrics': challenge_metric_entries,
                            'metrics_x': challenge_X_metric_entry,
                            'metrics_y': challenge_Y_metric_entry,
                            'metrics_category': "aggregation",
                            "participants": tools_quartiles
                        }
                        quartiles_table.append(challenge_object)
            
            # This is a fallback to the original code
            if len(quartiles_table) == 0:
                logger.error(f"Fix Challenge {challenge_OEB_id}")
                for metrics_category in metrics_categories:
                    ## Right now, we are skipping aggregation metrics
                    #if metrics_category['category'] != 'assessment':
                    #    continue
                    
                    # And we are also skipping the cases where we don't have 
                    # enough information to do the work
                    if len(metrics_category['metrics']) <= 1:
                        continue
                    
                    for i_metrics_X, metrics_X in enumerate(metrics_category['metrics']):
                        for metrics_Y in metrics_category['metrics'][i_metrics_X+1:]:
                            challenge_X_metric = metrics_X['metrics_id']
                            challenge_Y_metric = metrics_Y['metrics_id']


                            tools = {}
                            better = 'top-right'
                            # loop over all assessment datasets and create a dictionary like -> { 'tool': [x_metric, y_metric], ..., ... }
                            possible_participant_datasets = {}
                            for dataset in challenge['participant_datasets']:
                                #logger.debug(json.dumps(dataset, indent=4))
                                #get tool which this dataset belongs to
                                tool_id = dataset['depends_on']['tool_id']
                                tool_name = tool_names[tool_id]
                                possible_participant_datasets[dataset["_id"]] = (tool_id, tool_name)
                                
                            possible_assessments = set(map(lambda ad: ad["_id"], challenge['assessment_datasets']))
                            # It maps assessment datasets to the tool(s) which
                            # generated the participant dataset which was
                            # assessed
                            assessment_actions = {}
                            for event in challenge["metrics_test_actions"]:
                                tools_ids = []
                                assessment_ids = []
                                for i_dataset in event["involved_datasets"]:
                                    i_role = i_dataset["role"]
                                    if i_role == "incoming":
                                        incoming_dataset_id = i_dataset["dataset_id"]
                                        tool_ids = possible_participant_datasets.get(incoming_dataset_id)
                                        if tool_ids is not None:
                                            tools_ids.append(tool_ids)
                                    elif i_role == "outgoing":
                                        outgoing_dataset_id = i_dataset["dataset_id"]
                                        if outgoing_dataset_id in possible_assessments:
                                            assessment_ids.append(outgoing_dataset_id)
                                
                                for assessment_id in assessment_ids:
                                    assessment_actions.setdefault(assessment_id, []).extend(tools_ids)
                            
                            # At last
                            for dataset in challenge['assessment_datasets']:
                                #logger.debug(json.dumps(dataset, indent=4))
                                #get tool which this dataset belongs to
                                tools_ids = assessment_actions.get(dataset["_id"], [])
                                for tool_ids in tools_ids:
                                    tool_id , tool_name = tool_ids
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
                            
                            challenge_object = {
                                "_id": challenge_OEB_id,
                                "acronym": challenge_id,
                                'metrics_x': metrics[challenge_X_metric],
                                'metrics_y': metrics[challenge_Y_metric],
                                'metrics_category': metrics_category['category'],
                                "participants": tools_quartiles
                            }
                            quartiles_table.append(challenge_object)
    
    # Sort by acronym
    quartiles_table.sort(key=lambda co: co["acronym"])
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
        assessment_datasets: datasets(datasetFilters: {type: "assessment"}) {
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
        participant_datasets: datasets(datasetFilters: {type: "participant"}) {
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
        metrics_test_actions: test_actions(testActionFilters: {action_type: "MetricsEvent"}) {
          _id
          action_type
          challenge_id
          _metadata
          orig_id
          _schema
          status
          tool_id
          involved_datasets {
              dataset_id
              role
          }
        }
        aggregation_datasets: datasets(datasetFilters: {type: "aggregation"}) {
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
        aggregation_test_actions: test_actions(testActionFilters: {action_type: "AggregationEvent"}) {
          _id
          action_type
          challenge_id
          _metadata
          orig_id
          _schema
          status
          tool_id
          involved_datasets {
              dataset_id
              role
          }
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
    logging.getLogger().setLevel(logging.DEBUG)
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
        logger.debug(f"Got {len(response['data']['getBenchmarkingEvents'])} benchmarking events and {len(response['data']['getChallenges'])} challenges")
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
        #import sys
        #json.dump(responseTM, sys.stderr, indent=4)
        
        tool_list = responseTM["data"]["getTools"]
        if len(tool_list) == 0:
            logger.error(f"Tools for {community_id} not found")
            return None
        metrics_list = responseTM["data"]["getMetrics"]
        if len(metrics_list) == 0:
            logger.error(f"Metrics for {community_id} not found")
            return None
        logger.debug(f'{len(tool_list)} tools for {community_id} and {len(metrics_list)} possible metrics')
        
        # iterate over the list of tools to generate a dictionary
        tool_names = {}
        for tool in tool_list:
            tool_names[tool["_id"]] = tool["name"]
        
        # And the same for metrics
        metrics = dict()
        for m in metrics_list:
            metadata = m.get("_metadata")
            if metadata is not None:
                m["_metadata"] = json.loads(metadata)
            
            metrics[m["_id"]] = m
        
        # compute the classification
        #logger.info("data")
        #logger.info(data)
        #logger.info("classificator_id")
        #logger.info(classificator_id)
        #logger.info("tool_names")
        #logger.info(tool_names)
        #logger.info("metrics")
        #logger.info(metrics)
        #logger.info("challenge_list")
        #logger.info(challenge_list)
        result = build_table(data, classificator_id, tool_names, metrics, challenge_list)

        return result

    except Exception as e:
        logger.exception("Unexpected exception")
        abort(500)
