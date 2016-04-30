import os
import feature_saver
import data_generator
import locator

from flask import Flask
from flask import request, send_file, jsonify
from flask.ext.cors import CORS, cross_origin
from werkzeug import secure_filename

import requests # for performing our own HTTP requests for SV
import math
import PIL
from PIL import Image

from pymongo import MongoClient
import json
from bson import json_util
from bson.objectid import ObjectId

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
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['UPLOAD_FILENAME'] = 'query.jpg'
app.config['SV_FOLDER'] = 'sv/'
app.config['SV_FEATURES_FOLDER'] = 'sv/features/'
app.config['SV_FILENAMES'] = 'filenames.txt'
app.config['SV_QUERY'] = 'query.jpg'
app.config['SV_DATA'] = 'data.csv'
app.config['SV_LOCATIONS_FILENAME'] = 'locations.txt'

# TODO: just return the filename (easier)
# Given a location, fetch the SV images for each heading and pitch,
# compute the features and save to disk, and add filename to filenameFile
def saveSVImagesAndFeatures(lat,lng,theta,f_saver,filenameFile):
    # Street View api key
    key = 'AIzaSyCP5BKla9RY0aObtlovjVzIBV2XEsfYj48'

    img_filenames = ""
    # For each heading
    for i in range(0, int(math.floor(360/theta)),1):
        heading = i * theta
        # For each pitch
        for pitch in range(10, 50, 10):
            # Build the url to get the SV image at this location, heading and pitch
            url = 'https://maps.googleapis.com/maps/api/streetview?size=640x640'
            url += ('&location={},{}'.format(lat,lng))
            url += ('&fov={}'.format(theta))
            url += ('&heading={}'.format(heading))
            url += ('&pitch={}'.format(pitch))
            url += ('&key={}'.format(key))

            filename = '{},{},{},{}'.format(lat,lng,heading,pitch)

            # Get the image, stream it to file
            r = requests.get(url,stream=True)
            if r.status_code == 200:
                # open new file for writing and reading, binary
                with open(app.config['SV_FOLDER'] + filename + '.jpg', 'wb+') as f:
                    for chunk in r:
                        f.write(chunk)
                # Open the newly written image to read pixel data
                im = Image.open(app.config['SV_FOLDER'] + filename + '.jpg')
                # if first pixel is "no image" grey then no SV data at this location
                # so delete image and return
                if im.load()[0,0] == (228,227,223):
                    # print "...no imagery"
                    os.remove(app.config['SV_FOLDER'] + filename + '.jpg')
                    return
                else:
                    filenameFile.write(filename + '\n')
                    if(i == 0 and pitch == 10):
                        img_filenames += (filename + '.jpg');
                    else:
                        img_filenames += (":" + filename + '.jpg');
            else:
                print "...error fetching Street View image!"
    f_saver.saveFeatures(app.config['SV_FOLDER'], img_filenames, app.config['SV_FEATURES_FOLDER'])

#TODO: check for missing/invalid params
@app.route('/sv', methods=['POST'])
def sv():
    print "*** Fetching and processing Street View Images ***"
    # My C++ library to compute and save image features
    f_saver = feature_saver.FeatureSaver()
    # Open file for writing filenames
    filenameFile = open(app.config['SV_FOLDER'] + app.config['SV_FILENAMES'], 'w')
    # read args from POST form data
    args = request.form
    theta = int(args.get('theta'))

    # Fetch SV images from bounding box specified by
    # southwest=(lat1,lng1) and northwest=(lat2,lng2) corners
    if args.get('lat1') and args.get('lat2') and args.get('lng1') and args.get('lng2'):
        lat1 = float(args.get('lat1'))
        lng1 = float(args.get('lng1'))
        lat2 = float(args.get('lat2'))
        lng2 = float(args.get('lng2'))
        density = int(args.get('density'))

        # iterate over mesh of lat-lngs at specified density,
        # producing SV images at each point
        pc_change = 1.0 / (density+1) / (density+1)     # TODO: Is this a bug? Accidental paste?
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
    # Fetch SV images by coords specified in the uploaded file
    else:
        file = request.files['file']
        if file:
            file.save(app.config['SV_FOLDER'] + app.config['SV_LOCATIONS_FILENAME'])
            fileHandle = open(app.config['SV_FOLDER'] + app.config['SV_LOCATIONS_FILENAME'], 'r')
            line = fileHandle.readline()
            locations = line.split(':')
            print len(locations)
            locations = set(locations)
            print len(locations)
            for idx, location in enumerate(locations):
                print "{}%".format((float(idx)/len(locations))*100)
                lat = location.split(',')[0]
                lng = location.split(',')[1]
                saveSVImagesAndFeatures(lat,lng,theta,f_saver,filenameFile)
            fileHandle.close()
        else:
            print "Missing params"
            return jsonify(success='false')

    # My C++ library to compute and save image features
    filenameFile.close()
    f_saver = feature_saver.FeatureSaver()
    print app.config['SV_FOLDER'] + app.config['SV_FILENAMES']
    print app.config['SV_FEATURES_FOLDER']
    f_saver.saveBigTree(app.config['SV_FOLDER'] + app.config['SV_FILENAMES'], app.config['SV_FEATURES_FOLDER'])
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

##################### PRODUCTION ROUTES ##########################
@app.route('/place', methods=['POST', 'OPTIONS'])
@cross_origin(origin='*')
def add_place():
    args = request.get_json()
    name = args['name']
    address = args['address']
    description = args['description']
    location = args['location']

    #TODO: check for places too close by
    if name and address and location and 'lat' in location and 'lng' in location and description:
        try:
            place = {
                'name': name,
                'address': address,
                'loc': [ location['lng'], location['lat'] ],
                "description": description
            }
            db.places.insert_one(place)
        except:
            return jsonify(success="False")
        return jsonify(success=True,place=json_util.dumps(place))
    else:
        return jsonify(success=False)

@app.route('/place', methods=['GET'])
def get_place():
    _id = request.args.get('_id')
    if _id:
        try:
            place = db.places.find({'_id': ObjectId(_id)})
        except:
            return jsonify(success="False")
        return jsonify(success=True, place=json_util.dumps(place))
    else:
        return jsonify(success=False)


@app.route('/locate', methods=['POST'])
def locate():
    # Get the file
    file = request.files['file']
    # Path to save query image to sv/query.jpg
    filepath = os.path.join(app.config['SV_FOLDER'], app.config['SV_QUERY'])

    if file:
        # Ensure filename is safe and save
        filename = secure_filename(file.filename)
        file.save(filepath)

        # resize the image (too large causes out-of-memory error)
        img = Image.open(filepath)
        if img.size[1] < img.size[0]:
            width = 800
            wpercent = (width/float(img.size[0]))
            height = int((float(img.size[1]) * float(wpercent)))
        else:
            height = 800
            hpercent = (height/float(img.size[1]))
            width = int((float(img.size[0]) * float(hpercent)))

        img = img.resize((width, height), PIL.Image.ANTIALIAS)
        img.save(filepath)
    else:
        return jsonify(success=False)

    # locate the object in the query image and send response
    if l.locate(app.config['SV_FOLDER'] + app.config['SV_QUERY'], app.config['SV_FOLDER'], app.config['SV_FOLDER'] + app.config['SV_FILENAMES']):
        lat=l.getLat()
        lng=l.getLng()
        print "Looking for places near {},{}".format(lat, lng)
        try:
            places = db.places.find({
                'loc' : {
                    '$near': {
                        '$geometry': {
                            'type':"Point",
                            'coordinates':[lng, lat]
                        },
                    '$maxDistance': 100
                    }
                }
            })
            return jsonify(success=True,lat=l.getLat(),lng=l.getLng(),places=json_util.dumps(places))
        except:
            return jsonify(success=False)
    else:
        return jsonify(success=False)


if __name__ == '__main__':
    app.debug = False
    print "Loading..."
    l = locator.Locator()   #pre-load the locator
    print "Loaded!"
    db = MongoClient().identisnap
    print "Connected!"
    app.run(host='0.0.0.0')
