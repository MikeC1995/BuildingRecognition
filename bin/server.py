import cv2
import os
import recogniser

from flask import Flask
from flask import request, redirect, url_for, send_from_directory, jsonify
from werkzeug import secure_filename

# Load a recogniser for Wills Memorial Building
r = recogniser.Recogniser("wills")

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
            return jsonify(success='true',matches=matches);
        else:
            return jsonify(success='false');

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
