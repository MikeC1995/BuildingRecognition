import cv2
import os
import recogniser

from flask import Flask
from flask import request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename

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




# Load a recogniser for Wills Memorial Building
r = recogniser.Recogniser("christchurch", "ROOTSIFT")

# Create the flask REST server application
app = Flask(__name__)

# a set of allowed file extensions and the path to the folder to save them in
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'JPG'])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['UPLOAD_FILENAME'] = 'query.jpg'

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

if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0')
