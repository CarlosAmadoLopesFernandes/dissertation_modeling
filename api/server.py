from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
from kanpai import Kanpai
import numpy as np
from datetime import datetime
import mysql.connector
from pytz import timezone


app = Flask(__name__)
CORS(app)
model = pickle.load(open('naive_bayes.pkl', 'rb'))

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="test"
)

mycursor = mydb.cursor()



'''
park,
date,
'''


schema = Kanpai.Object({
    "park": Kanpai.String().required("Park is required"),
    "date": Kanpai.String().required("Date is required")
})

'''schema = Kanpai.Object({
    "day": Kanpai.Number().required("day is required"),
    "month": Kanpai.Number().required("month is required"),
    "year": Kanpai.Number().required("year is required"),
    "hour": Kanpai.Number().required("hour is required"),
    "weekDay": Kanpai.Number().required("weekDay is required"),
    "isWeekend": Kanpai.Number().required("isWeekend is required"),
    "isHoliday": Kanpai.Number().required("isHoliday is required"),
    "precipIntensity": Kanpai.Number().required("precipIntensity is required"),
    "precipProbability": Kanpai.Number().required("precipProbability is required"),
    "temperature": Kanpai.Number().required("temperature is required"),
    "apparentTemperature": Kanpai.Number().required("apparentTemperature is required"),
    "windSpeed": Kanpai.Number().required("windSpeed is required"),
    "numberOfSpaces": Kanpai.Number().required("numberOfSpaces is required"),
    "hourlyPrice": Kanpai.Number().required("hourlyPrice is required"),
    "parkName_CCN": Kanpai.Number().required("parkName_CCN is required"),
    "parkName_Docks Bruxsel": Kanpai.Number().required("parkName_Docks Bruxsel is required"),
    "parkName_Hippocrate": Kanpai.Number().required("parkName_Hippocrate is required"),
    "parkName_Lennik": Kanpai.Number().required("parkName_Lennik is required"),
    "parkingGroupName_Interparking": Kanpai.Number().required("parkingGroupName_Interparking is required"),
    "parkingGroupName_Q-Park Belgium": Kanpai.Number().required("parkingGroupName_Q-Park Belgium is required"),
    "parkingGroupName_Rauwers Parkingshop": Kanpai.Number().required("parkingGroupName_Rauwers Parkingshop is required"),
    "parkingGroupName_SIPE (Indigo Belgium)": Kanpai.Number().required("parkingGroupName_SIPE (Indigo Belgium) is required"),
    "place1Type_locality": Kanpai.Number().required("place1Type_locality is required"),
    "place2Type_lodging": Kanpai.Number().required("place2Type_lodging is required"),
    "place3Type_hospital": Kanpai.Number().required("place3Type_hospital is required"),
    "place3Type_lodging": Kanpai.Number().required("place3Type_lodging is required"),
    "place4Type_point_of_interest": Kanpai.Number().required("place4Type_point_of_interest is required"),
    "place4Type_restaurant": Kanpai.Number().required("place4Type_restaurant is required"),
    "place4Type_tourist_attraction": Kanpai.Number().required("place4Type_tourist_attraction is required"),
    "place5Type_lodging": Kanpai.Number().required("place5Type_lodging is required"),
    "place5Type_point_of_interest": Kanpai.Number().required("place5Type_point_of_interest is required")
})'''


@app.route('/predict', methods=['POST'])
def predict():
    try:
        columns = pickle.load(open('model_columns.pkl', 'rb'))

        '''
            ['parkName', 'day', 'month', 'year', 'hour', 'weekDay',
            'isWeekend', 'isHoliday', 'precipIntensity', 'precipProbability',
            'temperature', 'apparentTemperature', 'windSpeed', 'place1Type',
            'place2Type', 'place3Type', 'place4Type', 'place5Type',
            'parkingGroupName', 'numberOfSpaces', 'hourlyPrice',
            'class']
    
    
            ['day', 'month', 'year', 'hour', 'weekDay', 'isWeekend', 'isHoliday',
            'precipIntensity', 'precipProbability', 'temperature',
            'apparentTemperature', 'windSpeed', 'numberOfSpaces',
            'hourlyPrice', 'parkName_CCN',
            'parkName_Docks Bruxsel',
            'parkName_Hippocrate',
            'parkName_Lennik',
            'parkingGroupName_Interparking', 'parkingGroupName_Q-Park Belgium',
            'parkingGroupName_Rauwers Parkingshop', 'parkingGroupName_SIPE (Indigo Belgium)',
            'place1Type_locality', 'place2Type_lodging', 'place3Type_hospital',
            'place3Type_lodging', 'place4Type_point_of_interest', 'place4Type_restaurant',
            'place4Type_tourist_attraction', 'place5Type_lodging', 'place5Type_point_of_interest',
            'class_code']
        '''



        validation_result = schema.validate(request.json)
        if validation_result.get('success', False) is False:
            return jsonify({
                "status": "Error",
                "errors": validation_result.get("error")
            })
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

        model_entry_data = [
            data["date"].day,
            data["date"].month,
            data["date"].year,
            data["date"].hour,
            data["date"].weekday(),
            1 if data["date"].weekday() == 5 or data["date"].weekday() == 6 else 0,
            1 if holiday_data is not None else 0,
            meteo_data[7],
            meteo_data[8],
            meteo_data[9],
            meteo_data[10],
            meteo_data[14],
            park_data[8],
            park_data[17]
        ]

        print(model_entry_data)
        print(columns)

        #prediction = model.predict([np.array(data)])
        #print(prediction)
        #print(columns)
        #prediction = {'prediction': str(prediction)}
        #return jsonify(prediction)
        return jsonify(columns)
    except Exception as e:
        if park_data is None or  holiday_data is None or meteo_data is None or places_data is None:
            return jsonify({
                "error": "Problem retrieving data"
            }), 400
        else:
            return jsonify({
                "error": "Error, try again"
            }), 400

app.run()
