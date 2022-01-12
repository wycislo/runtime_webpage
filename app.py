# import necessary libraries
from models import create_classes
import os
import pickle
import joblib
import pandas as pd

from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    redirect)

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# Database Setup
#################################################

from flask_sqlalchemy import SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] =  "sqlite:///db.sqlite"

# Remove tracking modifications
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

Pet = create_classes(db)

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")
    
@app.route("/playerpos")
def playerpos():
    return render_template("playerpos.html")

@app.route("/draftcard")
def draftcard():
    return render_template("draftcard.html")

@app.route("/predmodel", endpoint='predmodel', methods=["GET", "POST"])
def predmodel():
    result={'intercept':' ','passingyards':' ','fumble':' ','passingcomplete':' ','score':' '}
    if request.method == "POST" and request.endpoint == 'predmodel':
        intercept = request.form["PlayerIntercept"]
        passingyards = request.form["PlayerPassingYards"]
        fumble = request.form["PlayerFumble"]
        passingcomplete = request.form["PlayerPassingComplete"]

        x_scaler = joblib.load('QBscaler.gz')

        datadict = {"Interceptions": intercept, "Passing Yards Per TD": passingyards, "Fumbles": fumble, "Passing Completion Per Attempts": passingcomplete}
        
        xdf = pd.DataFrame([datadict])
        xdf_scaled = x_scaler.transform(xdf)

        filename = 'QuaterbackModel_trained.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        newguypoint = loaded_model.predict(xdf_scaled)

        
        result={'intercept':intercept,'passingyards':passingyards,'fumble':fumble,'passingcomplete':passingcomplete,'score':newguypoint[0]}
        
        return render_template("predmodel.html",result=result)
        
    return render_template("predmodel.html",result=result)

@app.route("/wrpredmodel", endpoint='wrpredmodel' , methods=["GET", "POST"])
def wrpredmodel():
    result={'receptions':' ','recyardsperse':' ','recyardsperrec':' ','recyardspertd':' ','fumble':' ', 'score':' '}
    if request.method == "POST" and request.endpoint == 'wrpredmodel':
        receptions = request.form["PlayerReceptions"]
        recyardsperse = request.form["PlayerRecyardsperse"]
        recyardsperrec = request.form["PlayerRecyardsperrec"]
        recyardspertd = request.form["PlayerRecyardspertd"]
        fumble = request.form["PlayerFumble"]
        
        x_scaler = joblib.load('WRscaler.gz')

        datadict = {'receptions':receptions,'recyardsperse':recyardsperse,'recyardsperrec':recyardsperrec,'recyardspertd':recyardspertd,'fumble':fumble}
        
        xdf = pd.DataFrame([datadict])
        xdf_scaled = x_scaler.transform(xdf)

        filename = 'Widereceiver_trained.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        newguypoint = loaded_model.predict(xdf_scaled)

        
        result={'receptions':receptions,'recyardsperse':recyardsperse,'recyardsperrec':recyardsperrec,'recyardspertd':recyardspertd,'fumble':fumble,'score':newguypoint[0]}
        
        return render_template("wrpredmodel.html",result=result)
        
    return render_template("wrpredmodel.html",result=result)

@app.route("/rbpredmodel", endpoint='rbpredmodel', methods=["GET", "POST"])
def rbpredmodel():
    result={'rushattempts':' ','rushyards':' ','recyardspertd':' ','fumble':' ','score':' '}
    if request.method == "POST" and request.endpoint == 'rbpredmodel':
        rushattempts = request.form["PlayerRushattempts"]
        rushyards = request.form["PlayerRushyards"]
        recyardspertd = request.form["PlayerRecyardspertd"]
        fumble = request.form["PlayerFumble"]

        x_scaler = joblib.load('RBscaler.gz')

        datadict = {"Rushattempts": rushattempts, "Rushyards": rushyards, "Recyardspertd": recyardspertd, "Fumble": fumble}
        
        xdf = pd.DataFrame([datadict])
        xdf_scaled = x_scaler.transform(xdf)

        filename = 'Runningback_trained.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        newguypoint = loaded_model.predict(xdf_scaled)

        
        result={'rushattempts':rushattempts,'rushyards':rushyards,'recyardspertd':recyardspertd,'fumble':fumble,'score':newguypoint[0]}
        
        return render_template("rbpredmodel.html",result=result)
        
    return render_template("rbpredmodel.html",result=result)

# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        name = request.form["petName"]
        lat = request.form["petLat"]
        lon = request.form["petLon"]

        pet = Pet(name=name, lat=lat, lon=lon)
        db.session.add(pet)
        db.session.commit()
        return redirect("/", code=302)

    return render_template("form.html")


@app.route("/api/pals")
def pals():
    results = db.session.query(Pet.name, Pet.lat, Pet.lon).all()

    hover_text = [result[0] for result in results]
    lat = [result[1] for result in results]
    lon = [result[2] for result in results]

    pet_data = [{
        "type": "scattergeo",
        "locationmode": "USA-states",
        "lat": lat,
        "lon": lon,
        "text": hover_text,
        "hoverinfo": "text",
        "marker": {
            "size": 50,
            "line": {
                "color": "rgb(8,8,8)",
                "width": 1
            },
        }
    }]

    return jsonify(pet_data)


if __name__ == "__main__":
    app.run()
