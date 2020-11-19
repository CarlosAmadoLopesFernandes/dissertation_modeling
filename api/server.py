from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from kanpai import Kanpai
import numpy as np
from datetime import datetime
import mysql.connector
from pytz import timezone
import requests
from Config import configs



app = Flask(__name__)
CORS(app)
model = pickle.load(open('naive_bayes.pkl', 'rb'))

mydb = mysql.connector.connect(
    host=configs.config['mysql']['dev']['host'],
    user=configs.config['mysql']['dev']['user'],
    password=configs.config['mysql']['dev']['password']
)

mycursor = mydb.cursor()

schema = Kanpai.Object({
    "park": Kanpai.String().required("Park is required"),
    "date": Kanpai.String().required("Date is required")
})

schema_future = Kanpai.Object({
    "latitude": Kanpai.Number().required("Latitude is required"),
    "longitude":Kanpai.Number().required("Longitude is required"),
    "date": Kanpai.String().required("Latitude is required"),
})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        columns = pickle.load(open('model_columns.pkl', 'rb'))

        validation_result = schema.validate(request.json)
        if validation_result.get('success', False) is False:
            return jsonify({
                "status": "Error",
                "errors": validation_result.get("error")
            }), 400
        data = validation_result.get("data")
        #print(data)
        #data = data.values()
        #data = list(data)

        date_time_obj = datetime.strptime(data["date"], '%Y-%m-%d %H:%M:%S')
        data["date"] = date_time_obj

        ## GET PARK INFO FROM DATABASE
        query_park = "SELECT * FROM `dissertation`.`parks` where name= %s"
        mycursor.execute(query_park, (data["park"],))
        park_data = mycursor.fetchone()

        #GET HOLIDAYS FROM DATABASE WITH GIVEN DATE
        query_holidays = "SELECT * FROM `dissertation`.`holidays`where date = %s"
        mycursor.execute(query_holidays, (data["date"].date(), ))
        holiday_data = mycursor.fetchone()

        ##GET METEOS
        date_with_timezone = timezone("Europe/Brussels").localize(data["date"])
        query_meteo = "SELECT * FROM `dissertation`.`meteorologies`where time = %s"
        mycursor.execute(query_meteo, (date_with_timezone.timestamp(),))
        meteo_data = mycursor.fetchone()

        ##GET PLACES
        query_places = "SELECT * FROM `dissertation`.`places` where park_name = %s LIMIT 5";
        park_name = park_data[3]
        mycursor.execute(query_places, (park_name,))
        places_data = mycursor.fetchall()


        print(columns)

        for index, col in enumerate(columns):
            if col == "day":
                columns[index] = data["date"].day
            elif col == "month":
                columns[index] = data["date"].month
            elif col == "year":
                columns[index] = data["date"].year
            elif col == "hour":
                columns[index] = data["date"].hour
            elif col == "weekDay":
                columns[index] = data["date"].weekday()
            elif col == "isWeekend":
                if data["date"].weekday() == 5 or data["date"].weekday() == 6:
                    columns[index] = 1
                else:
                    columns[index] = 0

            elif col == "isHoliday":
                if holiday_data is not None:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col == "precipIntensity":
                columns[index] = meteo_data[7]
            elif col == "precipProbability":
                columns[index] = meteo_data[8]
            elif col == "temperature":
                columns[index] = meteo_data[9]
            elif col == "apparentTemperature":
                columns[index] = meteo_data[10]
            elif col == "windSpeed":
                columns[index] = meteo_data[14]
            elif col == "numberOfSpaces":
                columns[index] = park_data[8]
            elif col == "hourlyPrice":
                columns[index] = park_data[17]
            elif col.startswith("parkName"):
                if col == "parkName_" + park_data[3]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("parkingGroupName"):
                if col == "parkingGroupName_" + park_data[2]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("place1Type"):
                if col == "place1Type_" + places_data[0][7]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("place2Type"):
                if col == "place2Type_" + places_data[1][7]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("place3Type"):
                if col == "place3Type_" + places_data[2][7]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("place4Type"):
                if col == "place4Type_" + places_data[3][7]:
                    columns[index] = 1
                else:
                    columns[index] = 0
            elif col.startswith("place5Type"):
                if col == "place5Type_" + places_data[4][7]:
                    columns[index] = 1
                else:
                    columns[index] = 0

        prediction = model.predict([np.array(columns)])
        print(prediction)
        print(columns)
        prediction = {'prediction': str(prediction)}
        return jsonify(prediction)
        #return jsonify(columns)
    except Exception as e:
        if park_data is None or  holiday_data is None or meteo_data is None or places_data is None:
            return jsonify({
                "error": "Problem retrieving data"
            }), 400
        else:
            return jsonify({
                "error": "Error, try again"
            }), 400


@app.route('/predict_future', methods=['POST'])
def predict_future():
    print(configs.config)
    columns = pickle.load(open('model_columns.pkl', 'rb'))

    validation_result = schema_future.validate(request.json)
    if validation_result.get('success', False) is False:
        return jsonify({
            "status": "Error",
            "errors": validation_result.get("error")
        }), 400
    data = validation_result.get("data")
    date_with_timezone = timezone("Europe/Brussels").localize(data["date"])

    #GET ALL PARKS FROM DATABASE
    query_parks = "SELECT * FROM `dissertation`.`parks`"
    mycursor.execute(query_parks)
    park_data = mycursor.fetchall()

    # GET HOLIDAYS FROM DATABASE WITH GIVEN DATE
    query_holidays = "SELECT * FROM `dissertation`.`holidays`where date = %s"
    mycursor.execute(query_holidays, (data["date"].date(),))
    holiday_data = mycursor.fetchone()


    # GET METEO INFO FROM API
    url = "https://api.darksky.net/forecast/" + configs.config['dark_sky']['api_key'] + "/" + configs.config['general']['brussels_latitude'] + "," + configs.config['general']['brussels_longitude'] + "," + date_with_timezone.timestamp() + "?units=si&exclude=currently,flags"
    weather_info = requests.get(url).json()
    if weather_info:
        print(weather_info)
    else:
        print("ERROR retrieving data from darksky")

    return data
app.run()
