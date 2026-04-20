
from flask import Flask, redirect, render_template, request
from flask import Flask, render_template, request, redirect

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


app = Flask(__name__)

app.secret_key = "mysecretkey123"

# -----------------------------
# Database Config
# -----------------------------
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# -----------------------------
# Database Model
# -----------------------------
class News(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.utcnow)


# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = "Fake_Real_Data.csv"
MODEL_PATH = "fake_news_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


# -----------------------------
# Train Model
# -----------------------------
def train_model():

    print("🧠 Training model...")

    df = pd.read_csv(DATA_PATH)

    # Convert column names to lowercase
    df.columns = df.columns.str.lower()

    df = df[["text", "label"]]

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )

    tfidf_train = vectorizer.fit_transform(x_train)
    tfidf_test = vectorizer.transform(x_test)

    model = SGDClassifier(
        loss="hinge",
        learning_rate="optimal",
        max_iter=1000
    )

    model.fit(tfidf_train, y_train)

    # Save Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    y_pred = model.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print("✅ Accuracy:", round(score * 100, 2), "%")


# -----------------------------
# Prediction
# -----------------------------
def predict_news(text):

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    vec = vectorizer.transform([text])

    prediction = model.predict(vec)[0]

    confidence = np.max(model.decision_function(vec))
    confidence = round(abs(confidence) * 10, 2)

    return prediction, confidence


# Login Manager 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(200))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register
@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect("/login")
    return render_template("register.html")

# Login
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect("/")
    return render_template("login.html")

# Logout
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")


# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        news = request.form["news"]

        prediction, confidence = predict_news(news)

        # Save to Database
        new_news = News(
            text=news,
            prediction=prediction,
            confidence=confidence
        )

        db.session.add(new_news)
        db.session.commit()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


# -----------------------------
# History Route
# -----------------------------
@app.route("/history")
@login_required
def history():

    data = News.query.order_by(News.date.desc()).all()

    return render_template("history.html", data=data)


# -----------------------------
# About Route
# -----------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------
# Contact Route
# -----------------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# team route
@app.route("/team")
def team():
    return render_template("team.html")
    
# Delete Single History
@app.route("/delete/<int:id>")
def delete(id):

    news = News.query.get_or_404(id)

    db.session.delete(news)
    db.session.commit()

    return redirect("/history")

# Delete All History
@app.route("/delete_all")
def delete_all():

    News.query.delete()
    db.session.commit()

    return redirect("/history")


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    if not os.path.exists(MODEL_PATH):
        train_model()

    app.run(debug=True)
