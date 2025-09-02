from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import uuid
from datetime import date
import os

app = Flask(__name__)

# -------------------- Database Config --------------------
# Use PostgreSQL in production (Railway), SQLite for local development
database_url = os.environ.get('DATABASE_URL', 'sqlite:///fitness.db')
# Fix PostgreSQL URL format if needed
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- Session Model --------------------
class Session(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_session_id = db.Column(db.String(36), nullable=False)  # UUID for user session
    date = db.Column(db.Date, nullable=False, default=date.today)
    week = db.Column(db.Integer, nullable=False)
    year = db.Column(db.Integer, nullable=False)
    calories = db.Column(db.Float, nullable=False)

with app.app_context():
    db.create_all()

# -------------------- Load ANN Checkpoints --------------------
import os

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model weights
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "best_model.npz")
checkpoint = np.load(checkpoint_path)
W1, b1, W2, b2 = checkpoint["W1"], checkpoint["b1"], checkpoint["W2"], checkpoint["b2"]

# Load the StandardScaler used during training
scaler_path = os.path.join(BASE_DIR, "checkpoints", "scaler.npz")
scaler = np.load(scaler_path)
scaler_mean = scaler["mean"]
scaler_scale = scaler["scale"]

# -------------------- ANN Prediction --------------------
def relu(x):
    return np.maximum(0, x)

def predict_calories(input_features):
    # Apply the same StandardScaler normalization used during training
    X = np.array(input_features, dtype=np.float32).reshape(1, -1)
    X_scaled = (X - scaler_mean) / scaler_scale
    Z1 = np.dot(X_scaled, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    return float(Z2[0,0])

def encode_gender(gender_str):
    return 0 if gender_str.lower() == 'male' else 1

def get_week_number(d):
    return d.isocalendar()[1]

# -------------------- Routes --------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_session_id', methods=['GET'])
def get_session_id():
    """Generate a new session ID for the user"""
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})

@app.route('/get_totals', methods=['GET'])
def get_totals():
    """Get weekly and yearly totals for display"""
    try:
        user_session_id = request.headers.get('X-Session-ID')
        if not user_session_id:
            return jsonify({"error": "Session ID required"}), 400
            
        current_year = date.today().year
        
        # Weekly totals for current year (filtered by user session)
        weekly_totals_query = (
            db.session.query(Session.week, db.func.sum(Session.calories))
            .filter(Session.year == current_year, Session.user_session_id == user_session_id)
            .group_by(Session.week)
            .all()
        )
        weekly_totals = {f"Week {w}": float(total) for w, total in weekly_totals_query}
        # Ensure all 52 weeks are present
        weekly_totals_full = {f"Week {i}": 0 for i in range(1, 53)}
        weekly_totals_full.update(weekly_totals)
        
        # Yearly totals (filtered by user session)
        yearly_totals_query = (
            db.session.query(Session.year, db.func.sum(Session.calories))
            .filter(Session.user_session_id == user_session_id)
            .group_by(Session.year)
            .all()
        )
        yearly_totals = {str(y): float(total) for y, total in yearly_totals_query}
        
        return jsonify({
            "weekly_totals": weekly_totals_full,
            "yearly_totals": yearly_totals
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_session_id = request.headers.get('X-Session-ID')
        if not user_session_id:
            return jsonify({"error": "Session ID required"}), 400
            
        # Extract input
        gender = encode_gender(request.form['gender'])
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])

        # Prepare input features in the same order as training data
        X = [
            gender,
            age,
            height,
            weight,
            duration,
            heart_rate,
            body_temp
        ]
        calories_pred = predict_calories(X)

        # Save session with user_session_id
        today = date.today()
        week = get_week_number(today)
        year = today.year

        new_session = Session(
            user_session_id=user_session_id,
            date=today, 
            week=week, 
            year=year, 
            calories=calories_pred
        )
        db.session.add(new_session)
        db.session.commit()

        # Get updated totals (filtered by user session)
        current_year = date.today().year
        
        # Weekly totals for current year (filtered by user session)
        weekly_totals_query = (
            db.session.query(Session.week, db.func.sum(Session.calories))
            .filter(Session.year == current_year, Session.user_session_id == user_session_id)
            .group_by(Session.week)
            .all()
        )
        weekly_totals = {f"Week {w}": float(total) for w, total in weekly_totals_query}
        # Ensure all 52 weeks are present
        weekly_totals_full = {f"Week {i}": 0 for i in range(1, 53)}
        weekly_totals_full.update(weekly_totals)

        # Yearly totals (filtered by user session)
        yearly_totals_query = (
            db.session.query(Session.year, db.func.sum(Session.calories))
            .filter(Session.user_session_id == user_session_id)
            .group_by(Session.year)
            .all()
        )
        yearly_totals = {str(y): float(total) for y, total in yearly_totals_query}

        return jsonify({
            "today_calories": calories_pred,
            "weekly_totals": weekly_totals_full,
            "yearly_totals": yearly_totals
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/sessions', methods=['GET'])
def get_sessions():
    user_session_id = request.headers.get('X-Session-ID')
    if not user_session_id:
        return jsonify({"error": "Session ID required"}), 400
        
    sessions = Session.query.filter_by(user_session_id=user_session_id).order_by(Session.date).all()
    output = []
    for s in sessions:
        output.append({
            "id": s.id,
            "date": s.date.isoformat(),
            "week": s.week,
            "year": s.year,
            "calories": s.calories
        })
    return jsonify(output)

@app.route('/delete_session/<int:session_id>', methods=['DELETE'])
def delete_session(session_id):
    user_session_id = request.headers.get('X-Session-ID')
    if not user_session_id:
        return jsonify({"error": "Session ID required"}), 400
        
    session = Session.query.filter_by(id=session_id, user_session_id=user_session_id).first()
    if not session:
        return jsonify({"error": "Session not found"}), 404
    
    # Store session info before deletion
    session_date = session.date
    session_year = session.year
    
    db.session.delete(session)
    db.session.commit()
    
    # Get updated totals after deletion
    current_year = date.today().year
    
    # Weekly totals for current year
    weekly_totals_query = (
        db.session.query(Session.week, db.func.sum(Session.calories))
        .filter(Session.year == current_year, Session.user_session_id == user_session_id)
        .group_by(Session.week)
        .all()
    )
    weekly_totals = {f"Week {w}": float(total) for w, total in weekly_totals_query}
    # Ensure all 52 weeks are present
    weekly_totals_full = {f"Week {i}": 0 for i in range(1, 53)}
    weekly_totals_full.update(weekly_totals)
    
    # Yearly totals
    yearly_totals_query = (
        db.session.query(Session.year, db.func.sum(Session.calories))
        .filter(Session.user_session_id == user_session_id)
        .group_by(Session.year)
        .all()
    )
    yearly_totals = {str(y): float(total) for y, total in yearly_totals_query}
    
    return jsonify({
        "message": f"Session on {session_date} deleted successfully.",
        "weekly_totals": weekly_totals_full,
        "yearly_totals": yearly_totals
    })

# -------------------- Run App --------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
