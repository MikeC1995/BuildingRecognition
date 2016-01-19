import cv2
import os
import recogniser

from flask import Flask
from flask import request, redirect, url_for, send_from_directory
from werkzeug import secure_filename

# Create recogniser for Wills Memorial Building
r = recogniser.Recogniser("wills")


matches = r.query("/root/server/images/wills/query/0001.jpg")
print matches


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg'])
app.config['UPLOAD_FOLDER'] = 'uploads/'

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return 'Hello World! Post to me to recognise a building!'
    elif request.method == 'POST':
        file = request.files['file']
        # Check if the file is one of the allowed types/extensions
        if file and allowed_file(file.filename):
            # Make the filename safe, remove unsupported chars
            filename = secure_filename(file.filename)
            # Move the file form the temporal folder to
            # the upload folder we setup
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # Redirect the user to the uploaded_file route, which
            # will basicaly show on the browser the uploaded file
            return redirect(url_for('uploaded_file',
                                    filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
