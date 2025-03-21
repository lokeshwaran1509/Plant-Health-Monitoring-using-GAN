import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, render_template

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase Admin SDK with credentials
cred = credentials.Certificate("pass.json")  # Path to your Firebase credentials JSON file
firebase_admin.initialize_app(cred, {
    'databaseURL': 'Your URL goes here'
})

# Route to display data from Firebase
@app.route('/')
def index():
    # Get data from Firebase Realtime Database
    ref = db.reference('/')  # Refers to the root of your Firebase database
    data = ref.get()  # This will retrieve all the data at the root level

    # If you want to display more specific data, you can change the path in reference
    # Example: data = db.reference('some/path').get()

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
