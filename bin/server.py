import cv2
import os
import feature_saver
import data_generator
import locator

from flask import Flask
from flask import request, redirect, url_for, send_from_directory, send_file, jsonify
from werkzeug import secure_filename

import requests # for performing our own HTTP requests
import math
from PIL import Image

###### ACCESS CONTROL ##########

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

################################

# Create the flask REST server application
app = Flask(__name__)

# a set of allowed file extensions and the path to the folder to save them in
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'JPG'])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['UPLOAD_FILENAME'] = 'query.jpg'
app.config['SV_FOLDER'] = 'sv/'
app.config['SV_FEATURES_FOLDER'] = 'sv/features/'
app.config['SV_FILENAMES'] = 'filenames.txt'
app.config['SV_QUERY'] = 'query.jpg'
app.config['SV_DATA'] = 'data.csv'
app.config['BINS_FILENAME'] = 'bins.txt'

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# main route
# GET       Return hello message
# POST      include image file in body of request with key 'file'
#           returns the number of SIFT matches
@app.route('/', methods=['GET', 'POST'])
@crossdomain(origin='*')
def hello_world():
    if request.method == 'GET':
        return 'Hello World! Do a POST to match against Wills!'
    elif request.method == 'POST':
        # Get the file
        file = request.files['file']
        # Build the path to save it to
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], app.config['UPLOAD_FILENAME'])

        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Save file to upload folder as query.jpg
            file.save(filepath)

            # Run the recogniser on this query image to get the number of SIFT matches
            matches = r.query(filepath)
            print matches
            return jsonify(success='true',matches=matches);
        else:
            return jsonify(success='false');

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

def saveSVImagesAndFeatures(lat,lng,theta,f_saver,filenameFile):
    # Street View api key
    key = 'AIzaSyCP5BKla9RY0aObtlovjVzIBV2XEsfYj48'

    img_filenames = ""
    # For each heading
    for i in range(0, int(math.floor(360/theta)),1):
        heading = i * theta

        # Build the url to get the SV image at this location and heading
        url = 'https://maps.googleapis.com/maps/api/streetview?size=640x640'
        url += ('&location={},{}'.format(lat,lng))
        url += ('&fov={}'.format(theta))
        url += ('&heading={}'.format(heading))
        url += '&pitch=35'
        url += ('&key={}'.format(key))

        # Get the image, stream it to file
        r = requests.get(url,stream=True)
        if r.status_code == 200:
            # open new file for writing and reading, binary
            filename = '{},{},{}'.format(lat,lng,heading)
            with open(app.config['SV_FOLDER'] + filename + '.jpg', 'wb+') as f:
                for chunk in r:
                    f.write(chunk)
            # Open the newly written image to read pixel data
            im = Image.open(app.config['SV_FOLDER'] + filename + '.jpg')
            # if first pixel is "no image" grey then no SV data at this location
            # so delete image and return
            if im.load()[0,0] == (228,227,223):
                print "...no imagery"
                os.remove(app.config['SV_FOLDER'] + filename + '.jpg')
                return
            else:
                if(i == 0):
                    img_filenames += (filename + '.jpg');
                else:
                    img_filenames += (":" + filename + '.jpg');
        else:
            print "...error fetching Street View image!"
    filenameFile.write('{},{}\n'.format(lat,lng))
    f_saver.saveFeatures(app.config['SV_FOLDER'], img_filenames, app.config['SV_FEATURES_FOLDER'], app.config['BINS_FILENAME'])

@app.route('/sv', methods=['POST'])
def sv():
    print "*** Fetching and processing Street View Images ***"

    # read args from POST form data
    args = request.form
    lat1 = float(args.get('lat1'))
    lng1 = float(args.get('lng1'))
    lat2 = float(args.get('lat2'))
    lng2 = float(args.get('lng2'))
    density = int(args.get('density'))
    theta = int(args.get('theta'))

    # My C++ library to compute and save image features
    f_saver = feature_saver.FeatureSaver()

     # Open file for writing filenames
    filenameFile = open(app.config['SV_FOLDER'] + app.config['SV_FILENAMES'], 'w')

    # Delete bins file if already exists
    if os.path.exists(app.config['BINS_FILENAME']):
        os.remove(app.config['BINS_FILENAME'])

    # iterate over mesh of lat-lngs at specified density,
    # producing SV images at each point
    pc_change = 1.0 / (density+1) / (density+1)
    pc_complete = 0
    lat_step = (lat1 - lat2)/density
    lng_step = (lng1 - lng2)/density
    for i in range(0,density+1,1):
        for j in range(0,density+1,1):
            lat = lat2 + j * lat_step
            lng = lng2 + i * lng_step
            saveSVImagesAndFeatures(lat,lng,theta,f_saver,filenameFile)
            pc_complete += pc_change
            print "{}%".format(pc_complete * 100)

    filenameFile.close()
    return jsonify(success='true')

# produces a csv file detailing number of matches for query image against saved SV data
@app.route('/sv/csv', methods=['POST'])
def analyse():
    # Save the query image to file
    queryFilePath = os.path.join(app.config['SV_FOLDER'], app.config['SV_QUERY'])
    file = request.files['file']
    if file:
        file.save(queryFilePath)

    # My C++ library to process saved image features
    dg = data_generator.DataGenerator()
    filenameFile = app.config['SV_FOLDER'] + app.config['SV_FILENAMES']
    dg.generate(queryFilePath, filenameFile, app.config['SV_FEATURES_FOLDER'], app.config['SV_DATA'])
    return send_file(app.config['SV_DATA'], mimetype="text/csv")

@app.route('/sv/location', methods=['GET'])
def locate():
    #l = locator.Locator()
    #l.locate(app.config['SV_DATA'])
    #return jsonify(lat=l.getLat(),lng=l.getLng())

    # My C++ library to compute and save image features
    f_saver = feature_saver.FeatureSaver()
    f_saver.saveBigTree(app.config['SV_FOLDER'] + app.config['SV_FILENAMES'], app.config['SV_FEATURES_FOLDER'])

    l = locator.Locator()
    l.locateWithBigTree(app.config['SV_FOLDER'] + app.config['SV_QUERY'], app.config['BINS_FILENAME']);

    return jsonify(success='true')




if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
