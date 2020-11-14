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
columns = pickle.load(open('model_columns.pkl', 'rb'))

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
    print(data)
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



    print(data)
    print(park_data)
    print(holiday_data)
    print(meteo_data)
    print(columns)

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

    #prediction = model.predict([np.array(data)])
    #print(prediction)
    #print(columns)
    #prediction = {'prediction': str(prediction)}
    #return jsonify(prediction)
    return jsonify(data)

app.run()
