author = "Cheung Hei Wa, Yeung King Yiu"
import numpy as np
import pandas as pd
import subprocess
from beautifultable import BeautifulTable
from pymongo import MongoClient
import datetime
import re
import model

def DroppingAndCreating():
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["comp4332_Project"]
        db.course.drop()
        print("Collection dropping and empty collection creating are successful")
        client.close()
    except pymongo.errors.ConnectionFailure as error:
        print(error)

def DataCrawling(URL):
    if URL == "default":
        strCommand = 'scrapy crawl course_spider -a urls={}'.format("http://comp4332.com/realistic/")
        subprocess.run(strCommand, shell=True)
        print("Data Crawling is successful and all data are inserted into the database")
    else:
        strCommand = 'scrapy crawl course_spider -a urls={}'.format(URL)
        subprocess.run(strCommand, shell=True)
        print("Data crawling is completed!")
        

def CourseSearchByKeywords(keywords):
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["comp4332_Project"]

        # split the string: keywowrds into a list
        keywords = str(keywords)
        keywords = keywords.split(" ")
        regx_list = []
        # convert keywords into a list of regular expression
        for i in range(len(keywords)):
            to_be_append_1 = "[^A-Za-z0-9]{}[^A-Za-z0-9]".format(keywords[i])
            to_be_append_2 = "[^A-Za-z0-9]{}$".format(keywords[i])
            to_be_append_3 = "^{}[^A-Za-z0-9]".format(keywords[i])
            to_be_append_4 = "^{}$".format(keywords[i])
            regx_list.append(to_be_append_1)
            regx_list.append(to_be_append_2)
            regx_list.append(to_be_append_3)
            regx_list.append(to_be_append_4)

        result = db.course.aggregate([
                {"$match": {"$or": [
                            {
                                "CName": {
                                    "$in": [re.compile(element) for element in regx_list]
                                }
                            },
                            {
                                "CDescription": {
                                    "$in": [re.compile(element) for element in regx_list]
                                }
                            },
                            {
                                "Remark": {
                                    "$in": [re.compile(element) for element in regx_list]
                                }
                            } 
                            ]}},
                {
                    "$unwind": "$SectionListWithTimeslot"
                },
                {"$sort": {
                    "CID": 1,
                    "SectionListWithTimeslot.Section": 1,
                    "SectionListWithTimeslot.Timeslot": 1
                    }
                },
                {"$project": {
                    "_id": 0,
                    "CID": 1,
                    "CName": 1,
                    "Credit": 1,
                    "SectionListWithTimeslot.Timeslot": 1,
                    "SectionListWithTimeslot.Section": 1,
                    "SectionListWithTimeslot.Date_and_Time": 1,
                    "SectionListWithTimeslot.Quota": 1,
                    "SectionListWithTimeslot.Enrol": 1,
                    "SectionListWithTimeslot.Avail": 1,
                    "SectionListWithTimeslot.Wait": 1}},
                {
                    "$group": {
                        "_id": {
                            "CID": "$CID",
                            "CName": "$CName",
                            "Credit": "$Credit"
                        },
                        "SectionList": {
                            "$push": "$SectionListWithTimeslot"
                        }
                    }
                },
                {
                    "$sort": {"_id.CID":1}
                }
                ])

        course_list = []
        for courseList in result:
            cID = courseList['_id']['CID']
            course_list.append(cID)
            cName = courseList['_id']['CName']
            credit = courseList['_id']['Credit']
            print("Course_Code:\t{:s}\nCourse_Title:\t{:s}\nNo._of_Credits:\t{}".format(cID, cName, credit))
            table = BeautifulTable()
            table.column_headers = ["Timeslot", "Section", "Date_and_Time", "Quota", "Enrol", "Avail", "Wait"]
            for section in courseList["SectionList"]:
                table.append_row([section['Timeslot'].strftime("%Y-%m-%dT%H:%M:%S"), section['Section'],
                                  section['Date_and_Time'], section['Quota'], section['Enrol'],
                                  section['Avail'], section['Wait']])
            print(table)
            print("")

        print(course_list, "\n")
        print("---------Results End Here---------")
        print("")

        client.close()
        
    except pymongo.errors.ConnectionFailure as error:
        print(error)
    return

def CourseSearchByWaitList(f, start_ts, end_ts):
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client["comp4332_Project"]

        lowerbound = datetime.datetime.strptime(start_ts, "%Y-%m-%dT%H:%M:%S")
        upperbound = datetime.datetime.strptime(end_ts, "%Y-%m-%dT%H:%M:%S")

        firstStep = db.course.aggregate([
        {
            "$unwind": "$SectionListWithTimeslot"
        },
        {
            "$match": {
                "$and": [
                    {"SectionListWithTimeslot.Timeslot": {"$gte": lowerbound}},
                    {"SectionListWithTimeslot.Timeslot": {"$lte": upperbound}},
                    {"SectionListWithTimeslot.Section": {"$regex": '^L[0-9]'}}
                    ]}
        },
        {
            "$project": {
                "Cutoff": {
                    "$multiply": [ "$SectionListWithTimeslot.Enrol", f]},
                    "_id": 0,
                    "CID": 1,
                    "SectionListWithTimeslot": 1
                    }
        },
        {
            "$project": {
                "Satisfied": {"$gte" : ["$SectionListWithTimeslot.Wait", "$Cutoff"]},
                "_id": 0,
                "CID": 1,
                "SectionListWithTimeslot.Timeslot": 1
                }
        }, 
        {
            "$match": {"Satisfied": True}
        },
        {
            "$group": {
                "_id": "$CID",
                "LatestTimeslot": {"$max": "$SectionListWithTimeslot.Timeslot"}
                    }
        }, 
        {
            "$out": "TargetCourseAndTimeslot"
        }])

        secondStep = db.course.aggregate([
        {
            "$lookup": {
                "from": "TargetCourseAndTimeslot",
                "localField": "CID",
                "foreignField": "_id",
                "as": "course_TargetCourseAndTimeslot"
            }
        },
        {
            "$unwind": "$course_TargetCourseAndTimeslot"
        },
        {
            "$unwind": "$SectionListWithTimeslot"
        },
        {
            "$project": {
                "IsLastTimeslot": {
                    "$eq": [
                            "$course_TargetCourseAndTimeslot.LatestTimeslot",
                            "$SectionListWithTimeslot.Timeslot" ]
                },
                "_id": 0,
                "CID": 1,
                "CName": 1,
                "Credit": 1,
                "Timeslot": "$course_TargetCourseAndTimeslot.LatestTimeslot",
                "SectionListWithTimeslot": 1,
            }
        },
        {
            "$match": {
                "IsLastTimeslot": True
            }
        },
        {
            "$project": {
                "CID": 1,
                "CName": 1,
                "Credit": 1,
                "Timeslot": 1,
                "Cutoff": {
                    "$multiply": [
                            "$SectionListWithTimeslot.Enrol",
                            f]
                },
                "SectionListWithTimeslot.Section": 1,
                "SectionListWithTimeslot.Date_and_Time": 1,
                "SectionListWithTimeslot.Quota": 1,
                "SectionListWithTimeslot.Enrol": 1,
                "SectionListWithTimeslot.Avail": 1,
                "SectionListWithTimeslot.Wait": 1
            }
        },
        {
            "$project": {
                "CID": 1,
                "CName": 1,
                "Credit": 1,
                "Timeslot": 1,
                "SectionListWithTimeslot.Satisfied": {
                    "$gte": [
                            "$SectionListWithTimeslot.Wait",
                            "$Cutoff"]
                },
                "SectionListWithTimeslot.Section": 1,
                "SectionListWithTimeslot.Date_and_Time": 1,
                "SectionListWithTimeslot.Quota": 1,
                "SectionListWithTimeslot.Enrol": 1,
                "SectionListWithTimeslot.Avail": 1,
                "SectionListWithTimeslot.Wait": 1
            }
        },
        {
            "$sort": {
                "CID": 1,
                "SectionListWithTimeslot.Section": 1
            }
        },
        {
            "$group": {
                "_id": {
                    "CID": "$CID",
                    "CName": "$CName",
                    "Credit": "$Credit",
                    "Timeslot": "$Timeslot"
                },
                "SectionList": {
                    "$push": "$SectionListWithTimeslot"
                }
            }
        },
        {
            "$sort": {
                "_id.CID": 1,
            }
        }
        ])

        print("Course Search By Waitlist")
        for courseList in secondStep:
            table = BeautifulTable()
            table.column_headers = ["Section", "Date_and_Time", "Quota", "Enrol", "Avail", "Wait", "Satisfied"]
            cID = courseList['_id']['CID']
            cName = courseList['_id']['CName']
            credit = courseList['_id']['Credit']
            timeslot = courseList['_id']['Timeslot'].strftime("%Y-%m-%dT%H:%M:%S")
            print("Course_Code:\t{:s}\nCourse_Title:\t{:s}\nNo._of_Credits:\t{}\nMatched_Time_Slot:\t{:s}".format(cID, cName, credit, timeslot))
            for section in courseList["SectionList"]:
                table.append_row([section['Section'], section['Date_and_Time'], section['Quota'],
                                  section['Enrol'], section['Avail'], section['Wait'], section["Satisfied"]])
            print(table)
            print("")

        print("---------Results End Here---------")
        print("")
        client.close()
    
    except pymongo.errors.ConnectionFailure as error:
        print(error)
    return

def WaitingListTraining():
    print("Waiting list size training is successful")
    return

working = 1
while (working == 1):
    print("1. Collection Dropping and Empty collection creating\n"
          "2. Data Crawling\n"
          "3. Course Search\n"
          "4. Waiting List Size Prediction\n"
          "5. Waiting List Size Training\n"
          "6. Exit")
    text = input("Please text the fucntion you want! (e.g. Data Crawling) \n")

    if text == "Collection Dropping and Empty collection creating":
        DroppingAndCreating()
    elif text == "Data Crawling":
        link = input("Please type in URL:\n")
        DataCrawling(link)
    elif text == "Course Search":
        search = 0
        while (search == 0):
            choice = input("Key in 1 for Course Search by Keywords,\nKey in 2 for Course Search by Wait List,\nKey in 3 for Exit")
            if choice == "1":
                keywords = input("Please key in the keyword\n")
                CourseSearchByKeywords(keywords)
                search = 1
            elif choice == "2":
                print("Please indicate courses between which time you are interested in first.\n")
                start_ts = "0"
                end_ts = "0"
                time_pattern = re.compile('^\d\d\d\d\-\d\d\-\d\dT\d\d\:\d\d\:\d\d$')

                # input for start_ts
                while (start_ts == "0"):
                    print("Please type in a starting timeslot in format %Y-%M-%DT%h:%m:%s (e.g. 2012-01-01T00:00:00)")
                    start_ts = input("Please noted that 09:00:00 refers to 09:00:00 - 09:30:00\n")
                    if not time_pattern.match(start_ts):
                        start_ts = "0"
                        print("Wrong datetime format\n")

                # input for end_ts
                while (end_ts == "0"):
                    print("Please type in a ending timeslot in format %Y-%M-%DT%h:%m:%s (e.g. 2012-01-01T00:00:00)")
                    end_ts = input("Please noted that 09:00:00 refers to 09:00:00 - 09:30:00\n")
                    if not time_pattern.match(end_ts):
                        end_ts = "0"
                        print("Wrong datetime format\n")

                # input put for f
                print("Courses with waiting list size >= F * number of enrolled students would be shown\n")
                f = -1
                while (f < 0):
                    f = float(input("Please type in a non-negative real number for F.\n"))

                CourseSearchByWaitList(f, start_ts, end_ts)
                search = 1
            elif (choice == "3"):
                break
            else:
                print("Input error\n")
    elif text == "Waiting List Size Prediction":
        cc = 0
        course_code_format = re.compile('^[A-Z][A-Z][A-Z][A-Z] \d\d\d\d[A-Za-z]?$')
        # input for cc
        while (cc == 0):
            cc = input("Please type in a course code (e.g. COMP 4332)\n")
            if not course_code_format.match(cc):
                cc = 0
                print("Wrong course code format\n")

            # input for ln
        ln = 0
        pattern = re.compile('\d+')
        while (ln == 0):
            ln = input("Please type in an positive integer to specify a section of {}. ie. 1 refers to L1\n".format(cc))
            if not pattern.match(ln) or float(ln) <= 0:
                ln = 0
                print("Wrong input format")

        # input for ts
        ts = 0
        time_pattern = re.compile('^\d\d\d\d\-\d\d\-\d\dT\d\d\:\d\d\:\d\d$')
        while (ts == 0):
            print("Please type in you would like to predict the waiting list size of {} in which timeslot\n in format %Y-%M-%DT%h:%m:%s (e.g. 2012-01-01T00:00:00)".format(cc))
            ts = input("ie. 09:00:00 refers to 09:00:00 - 09:30:00\n")
            if not time_pattern.match(ts):
                ts = 0
                print("Wrong datetime format\n")

        a = model.Model1Prediction(cc,ln,ts)
        b = model.Model2Prediction(cc,ln,ts)
        c = model.Model3Prediction(cc,ln,ts)
        d = model.Model4Prediction(cc,ln,ts)
        e = model.Model5Prediction(cc,ln,ts)
        print([a,b,c,d,e])

    elif text == "Waiting List Size Training":
        df = model.GetDFfromMongo()
        action1 = input("Key in 1.Model 1 2.Model 2 3.Model 3 4. Model 4 5. Model 5:")
        if action1 == '1':
            model.StaticANNModel(df)
        elif action1 == '2':
            model.PerCourseANNTimeSeries(df)
        elif action1 == '3':
            model.CourseDummyANNModel(df)
        elif action1 == '4':
            model.SimpleTimeSeriesLSTMModel(df)
        elif action1 == '5':
            model.SimpleTimeSeriesGRUModel(df)
    elif text == "Exit":
        working = 0
    else:
        print("Wrong input")
        
