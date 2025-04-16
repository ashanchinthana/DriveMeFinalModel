from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import datetime
from integrated_predictions import predict_with_location_data

# Initialize Flask app
app = Flask(__name__, static_folder='./', static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Home route serving HTML dashboard
@app.route('/')
def index():
    return send_file('integrated_dashboard.html')

# API endpoint to get integrated predictions
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        # Get date range from query parameters or use default (today + 10 days)
        days = request.args.get('days', default=10, type=int)
        today = datetime.datetime.now().date()
        start_date = today
        end_date = today + datetime.timedelta(days=days)
        
        # Get predictions
        predictions = predict_with_location_data(start_date, end_date)
        
        if predictions:
            return jsonify({
                "success": True,
                "predictions": predictions
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to generate predictions."
            }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("Starting integrated prediction API server...")
    print("Access the dashboard at http://127.0.0.1:8000")
    app.run(debug=True, port=8000, host='0.0.0.0') 