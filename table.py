from __future__ import division
from flask import (
    Blueprint, jsonify
)
from sklearn.cluster import KMeans
from base64 import b64decode
import numpy as np
import pandas
import matplotlib.pyplot as plt
import urllib2
import json
import StringIO


# funtion that gets quartiles for x and y values
def plot_square_quartiles(x_values, means, tools, better, percentile=50):
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
def plot_diagonal_quartiles(x_values, means, tools, better):
    # get distance to lowest score corner

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


# function that prints a table with the list of tools and the corresponding quartiles
def print_full_table(quartiles_table):
    row_names = sorted(quartiles_table[next(iter(quartiles_table))].keys())

    # build matrix with all results
    quartiles_list = []

    for name in sorted(quartiles_table.iterkeys()):

        quartiles = []

        for row in row_names:
            quartiles.append(quartiles_table[name][row])

        quartiles_list.append(quartiles)

    text = []
    for tool in row_names:
        text.append([tool])

    for num, name in enumerate(row_names):
        for i in range(len(quartiles_table.keys())):
            text[num].append(quartiles_list[i][num])

    # get total score for all methods

    quartiles_sums = {}

    for num, val in enumerate(text):
        total = sum(text[num][i] for i in range(1, len(text[num]), 1))

        quartiles_sums[text[num][0]] = total

    # sort tools by that score to rank them

    sorted_quartiles_sums = sorted(quartiles_sums.items(), key=lambda x: x[1])

    # append to the final table

    for i, val in enumerate(sorted_quartiles_sums):
        for j, lst in enumerate(text):
            if val[0] == text[j][0]:
                text[j].append("# " + str(i + 1))

    df = pandas.DataFrame(text)
    vals = df.values

    # green color scale
    colors = df.applymap(lambda x: '#238b45' if x == 1 else '#ffffff')
    colors = colors.values

    ## build matplotlib image
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    fig.tight_layout()
    method_names = sorted(quartiles_table.iterkeys())

    method_names = ["TOOL / CHALLENGE -->"] + method_names
    method_names.append("# RANKING #")

    the_table = ax.table(cellText=vals,
                         colLabels=method_names,
                         cellLoc='center',
                         loc='center',
                         # bbox=[1.1, 0.15, 0.5, 0.8])
                         colWidths=[0.16, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06],
                         cellColours=colors,
                         colColours=['#ffffff'] * len(df.columns))
    fig.tight_layout()
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    the_table.scale(1, 1.5)
    plt.subplots_adjust(right=0.95, left=0.04, top=0.9, bottom=0.1)


# function that clusters participants using the k-means algorithm
def cluster_tools(my_array, tools, better):
    X = np.array(my_array)
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

def build_table(bench_id, classificator_id):
    try:
        response = urllib2.urlopen(
            'https://dev-openebench.bsc.es/api/scientific/Dataset/?query=' + bench_id + '+assessment&fmt=json')
        data = json.loads(response.read())['Dataset']

        challenges = set()

        for dataset in data:
            challenge_id = dataset['_id'].split('_')[1]
            challenges.add(challenge_id)

        # this dictionary will store all the information required for the quartiles table
        quartiles_table = {}

        for challenge in challenges:
            x_values = []
            y_values = []
            tools = set()
            better = 'top-right'
            for dataset in data:
                tools.add(dataset['depends_on']['tool_id'].split(":")[1])
                challenge_id = dataset['_id'].split('_')[1]
                if challenge == challenge_id:
                    data_uri = dataset['datalink']['uri']
                    encoded = data_uri.split(",")[1]
                    metric = float(b64decode(encoded))
                    if dataset['depends_on']['metrics_id'] == "TCGA:TPR":
                        x_values.append(metric)
                    elif dataset['depends_on']['metrics_id'] == "TCGA:PPV":
                        y_values.append(metric)

            # get quartiles depending on selected classification method

            if classificator_id == "squares":
                tools_quartiles = plot_square_quartiles(x_values, y_values, sorted(list(tools)), better)

            elif classificator_id == "clusters":
                tools_quartiles = cluster_tools(zip(x_values, y_values), tools, better)

            else:
                tools_quartiles = plot_diagonal_quartiles(x_values, y_values, sorted(list(tools)), better)

            quartiles_table[challenge] = tools_quartiles

        print_full_table(quartiles_table)

        fig = plt.gcf()
        fig.set_size_inches(20, 11.1)
        imgdata = StringIO.StringIO()
        fig.savefig(imgdata, format='svg')
        imgdata.seek(0)  # rewind the data

        plt.close("all")

        return quartiles_table

    except urllib2.URLError as e:

        print e.reason


# create blueprint and define url
bp = Blueprint('table', __name__)

# build_table('TCGA:2018-04-05', 'diagonals')

@bp.route('/<string:bench_id>')
@bp.route('/<string:bench_id>/<string:classificator_id>')
def compute_classification(bench_id, classificator_id="diagonals"):
    out = build_table(bench_id, classificator_id)
    response = jsonify(out)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response
    # return send_from_directory("/home/jgarrayo/public_html/flask_table/", "table.svg", as_attachment=False)
    # return send_file(out, mimetype='svg')
    # return render_template('index.html', data=out)

# @app.route('your route', methods=['GET'])
# def yourMethod(params):
#    response = flask.jsonify({'some': 'data'})
#    response.headers.add('Access-Control-Allow-Origin', '*')
#    return response