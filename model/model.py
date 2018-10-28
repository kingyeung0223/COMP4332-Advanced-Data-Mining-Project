import pymongo
import pprint
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.models import model_from_json
from datetime import date, datetime, timedelta
import datetime

def saveModel(model, modelFilenamePrefix):
	structureFilename = modelFilenamePrefix + ".json"
	model_json = model.to_json()
	with open(structureFilename, "w") as f:
		f.write(model_json)
	weightFilename = modelFilenamePrefix + ".h5"
	model.save_weights(weightFilename)
	return 1

def readModel(modelFilenamePrefix):
	structureFilename = modelFilenamePrefix + ".json"
	with open(structureFilename, "r") as f:
		model_json = f.read()
	model = model_from_json(model_json)
	weightFilename = modelFilenamePrefix + ".h5"
	model.load_weights(weightFilename)
	return model

def checkCC(l):
    if len(l) == 0:
        return False
    else:
        for sen in l:
            if 'Common Core' in sen:
                return True
        return False
    
def checkSSC(l):
    if len(l) == 0:
        return False
    else:
        for sen in l:
            if 'SSC' in sen:
                return True
        return False
    
def courseClusters(name):
    dummy = {'BUS':0,'SENG':0,'SSCI':0,'SHSS':0,'IPO':0,'OTR':0,'startCode':int(name[5])}
    OTR = set(['GNED','HLTH','LANG','PDEV','PPOL','SUST','UROP'])
    BUS = set(['ACCT','ECON','FINA','GBUS','ISOM','LABU','MARK','MGMT','MIMT','SBMT','WBBA'])
    IPO = set(['EVSM','IDPO','IIMP','RMBI','TEMG'])
    SHSS = set(['HMMA','HUMA','MGCS','MILE','SHSS','SOSC','SSMA','HART'])
    SENG = set(['AESF','BIEN','CBME','CENG','CIEM','MSBD','JEVE''CIVL','COMP','CPEG','CSIC','CSIT',
               'EEMT','EESM','ELEC','ENEG','ENGG','EVNG','IBTM','IELM','MECH','MESF','NANO'])
    SSCI = set(['BIBU','BIPH','BTEC','CHEM','CHMS','ENVR','ENVS','EVSM','LIFS','MAED','MAFS','MATH','SCIE'])
    if name[:4] in BUS:
        dummy['BUS'] = 1
    elif name[:4] in SENG:
        dummy['SENG'] = 1
    elif name[:4] in SSCI:
        dummy['SSCI'] = 1
    elif name[:4] in SHSS:
        dummy['SHSS'] = 1
    elif name[:4] in IPO:
        dummy['IPO'] = 1
    else:
        dummy['OTR'] = 1
    return dummy

def courseInfo(entry):
    processing = {}
    a = entry['Attributes']
    if checkCC(a):
        processing['IsCommonCore'] = 1
    else:
        processing['IsCommonCore'] = 0
    
    if checkSSC(a):
        processing['IsSSCCommonCore'] = 1
    else:
        processing['IsSSCCommonCore'] = 0
    
    if isinstance(entry['Pre_requisite'],str):
        processing['hasPre-re'] = 1
    else:
        processing['hasPre-re'] = 0
        
    if isinstance(entry['CoList'], str):
        processing['hasCoList'] = 1
    else:
        processing['hasCoList'] = 0
        
    if isinstance(entry['Exclusion'], str):
        processing['hasExclusion'] = 1
    else:
        processing['hasExclusion'] = 0
    return processing

def create_dataset(df, look_back=1):
	for i in range(1,look_back+1):
		try:
			df["t-"+str(i)+" Enroll"] = df.Enrol.shift(i)
			df["t-"+str(i)+" Wait"] = df.Wait.shift(i)
			df["t-"+str(i)+" Emptiness"] = df.Emptiness.shift(i)
			df["t-"+str(i)+" Prob"] = df.Probability.shift(i)
		except:
			print("Not enough data for shifting")
			newDF = df
	newDF = df.iloc[look_back:,:]
	return newDF

def fitANNModel(lastDF,predictCol):
    np.random.seed(4332)

    known = lastDF.drop(predictCol,axis = 1)
    X = known.values
    Y = lastDF[predictCol].values
    print(lastDF.head())
    print(X)
    print(Y)

    model = keras.models.Sequential()
    model.add(Dense(12, input_dim = len(X[0]), activation = 'relu'))
    model.add(Dense(1, activation = 'relu'))

    model.compile(loss="mse", optimizer="adam",  metrics = ["mean_squared_error"])
    model.fit(X, Y, epochs=20, batch_size = 512, validation_split=0.2)

    scores = model.evaluate(X, Y)
    print("")
    print("{}: {}".format(model.metrics_names[1], scores[1]*100))
    return model

def LSTMModel(aDF,predictCol):
    np.random.seed(4332)

    known = aDF.drop(predictCol,axis = 1)
    beforeRe = known.values
    le = beforeRe.shape[0]
    X = beforeRe.reshape(le,2,4)
    Y = aDF[predictCol].values
    print(aDF.head())
    print(X)
    print(Y)
    model = keras.models.Sequential()
    model.add(LSTM(3, input_shape = (2,4)))
    model.add(Dense(1, activation = 'relu'))

    model.compile(loss="mean_squared_error", optimizer="adam",  metrics=["mean_squared_error"])
    model.fit(X, Y, epochs=20, batch_size = 512, validation_split=0.2)

    scores = model.evaluate(X, Y)
    print("")
    print("{}: {}".format(model.metrics_names[1], scores[1]*100))
    return model

def GRUModel(aDF,predictCol):
    np.random.seed(4332)

    known = aDF.drop(predictCol,axis = 1)
    beforeRe = known.values
    le = beforeRe.shape[0]
    X = beforeRe.reshape(le,2,4)
    Y = aDF[predictCol].values
    print(aDF.head())
    print(X)
    print(Y)
    model = keras.models.Sequential()
    model.add(GRU(3, input_shape = (2,4)))
    model.add(Dense(1, activation = 'relu'))

    model.compile(loss="mean_squared_error", optimizer="adam",  metrics=["mean_squared_error"])
    model.fit(X, Y, epochs=20, batch_size = 512, validation_split=0.2)

    scores = model.evaluate(X, Y)
    print("")
    print("{}: {}".format(model.metrics_names[1], scores[1]*100))
    return model

def predictANNModel(NNModel,l):
    predictY = NNModel.predict(l, batch_size = 4)
    print(predictY)

def GetDFfromMongo():
	try:
		client = pymongo.MongoClient('localhost',27017)
		db = client.comp4332_Project

		query = db.course.aggregate([{"$unwind":"$SectionListWithTimeslot"},
									{"$project":{'CID':1,'CName': 1,'CoList': 1,'Co_requisite': 1,'Credit': 1, 'Exclusion': 1,
									'Pre_requisite': 1, 'Attributes':1, 'Avail':'$SectionListWithTimeslot.Avail',
									'Date_Time':'$SectionListWithTimeslot.Date_and_Time','Enrol':'$SectionListWithTimeslot.Enrol',
									'Quota':'$SectionListWithTimeslot.Quota','Room':'$SectionListWithTimeslot.Room',
									'Section':'$SectionListWithTimeslot.Section','Timeslot':'$SectionListWithTimeslot.Timeslot',
									'Wait':'$SectionListWithTimeslot.Wait'
									}},
									{"$sort":{"Timeslot":1}}
									],
                                   allowDiskUse = True)

		processed = []
		for entry in query:

			del entry['_id']
			n = entry["CID"]
			CIDbreak = courseClusters(n)
			processing = courseInfo(entry)

			trans = {}
			trans["CID"] = n
			trans["Timeslot"] = entry["Timeslot"]
			trans["Section"] = entry["Section"]
			trans["Credit"] = entry["Credit"]
			trans["Quota"] = entry["Quota"]
			trans["Enrol"] = entry["Enrol"]
			trans["Avail"] = entry["Avail"]

			if entry["Quota"] == 0:
			    trans["Probability"] = 0
			    trans["Emptiness"] = 0
			else:
			    trans["Probability"] = entry["Wait"]/entry["Quota"]
			    trans["Emptiness"] = entry["Avail"]/entry["Quota"]
			trans["Wait"] = entry["Wait"]

			newDict = {}
			newDict.update(CIDbreak)
			newDict.update(processing)
			newDict.update(trans)
			processed.append(newDict)

	except pymongo.errors.ConnectionFailure as error:
		print(error)

	df = pd.DataFrame(processed)
	df = df[df["Section"].str.contains('^L\d',regex = True)]
	return df

def StaticANNModel(df):
	course = set(df["CID"].values)
	listOfDF = []
	while len(course) >0:
		st = course.pop()
		sameCourseDF = df.loc[df['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			listOfDF.append(create_dataset(sameSectionDF,2))
	bigDF = pd.concat(listOfDF)

	DFs = {}
	DFs["Wait"] = bigDF.drop(["Avail","CID","Emptiness","Enrol","Probability","Section","Timeslot"],axis = 1)
	DFs["Avail"] = bigDF.drop(["Wait","CID","Emptiness","Enrol","Probability","Section","Timeslot"],axis = 1)
	DFs["Emptiness"] = bigDF.drop(["Avail","CID","Wait","Enrol","Probability","Section","Timeslot"],axis = 1)
	DFs["Enrol"] = bigDF.drop(["Avail","CID","Emptiness","Wait","Probability","Section","Timeslot"],axis = 1)
	DFs["Probability"] = bigDF.drop(["Avail","CID","Emptiness","Enrol","Wait","Section","Timeslot"],axis = 1)
	for key in DFs:
		model = fitANNModel(DFs[key],key)
		if saveModel(model,"ANN Model 1 for "+key):
			print("Save ANN Model 1 for "+key+" Successfully!")

def PerCourseANNTimeSeries(df2):
	df2 = df2.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
                "SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)

	dictOfModel = {}
	code = input("Please key in the course code ")
	course = set(df2["CID"].values)
	key = pd.Series(list(course))
	for st in key[key.str.contains(code)]:
		sameCourseDF = df2.loc[df2['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			newDF2 = create_dataset(sameSectionDF,2)

			DFs = {}
			DFs["Wait"] = newDF2.drop(["CID","Avail","Emptiness","Enrol","Probability","Timeslot","Section"],axis = 1)
			DFs["Avail"] = newDF2.drop(["CID","Wait","Emptiness","Enrol","Probability","Timeslot","Section"],axis = 1)
			DFs["Emptiness"] = newDF2.drop(["CID","Avail","Wait","Enrol","Probability","Timeslot","Section"],axis = 1)
			DFs["Enrol"] = newDF2.drop(["CID","Avail","Emptiness","Wait","Probability","Timeslot","Section"],axis = 1)
			DFs["Probability"] = newDF2.drop(["CID","Avail","Emptiness","Enrol","Wait","Timeslot","Section"],axis = 1)
			for key in DFs:
				DFs[key] = fitANNModel(DFs[key],key)
                
			SectionN = OneSection[:2]
			if st in dictOfModel.keys():
				dictOfModel[st][SectionN] = DFs
			else:
				dictOfModel[st] = {SectionN:DFs}
				
	for cr in dictOfModel:
		for ln in dictOfModel[cr]:
			for key in dictOfModel[cr][ln]:
				if saveModel(dictOfModel[cr][ln][key],cr+"_"+ln+"_"+key):
					print("Save "+cr+"_"+ln+"_"+key+" Successfully!")

def CourseDummyANNModel(df3): 
	df3 = df3.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
					"SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	course = set(df3["CID"].values)
	listOfDF1 = []
	while len(course) >0:
		st = course.pop()
		sameCourseDF = df3.loc[df3['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			listOfDF1.append(create_dataset(sameSectionDF,2))
	bigDF3 = pd.concat(listOfDF1)
	bigDF3["newSection"] = bigDF3.Section.str.slice(0, 2)

	sectionDummy = pd.get_dummies(bigDF3["newSection"])
	courseDummy = pd.get_dummies(bigDF3["CID"])

	DFs = {}
	DFs["Wait"] = bigDF3.drop(["Avail","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Avail"] = bigDF3.drop(["Wait","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Emptiness"] = bigDF3.drop(["Avail","Enrol","Quota","Wait","Probability","Section","Timeslot"],axis = 1)
	DFs["Enrol"] = bigDF3.drop(["Avail","Wait","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Probability"] = bigDF3.drop(["Avail","Enrol","Quota","Emptiness","Wait","Section","Timeslot"],axis = 1)
	for key in DFs:
		tempDF = pd.concat([DFs[key], courseDummy, sectionDummy], axis=1)
		DFs[key] = tempDF.drop(["CID","newSection"],axis = 1)
		model = fitANNModel(DFs[key],key)
		if saveModel(model,"CourseDummyANN Model 3 for "+key):
			print("Save CourseDummyANN Model 3 for "+key+" Successfully!")

def SimpleTimeSeriesLSTMModel(df4):
	df4 = df4.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
					"SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"], axis = 1)
	course = set(df4["CID"].values)
	listOfDF4 = []
	while len(course) >0:
		st = course.pop()
		sameCourseDF = df4.loc[df4['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			listOfDF4.append(create_dataset(sameSectionDF,2))
	bigDF4 = pd.concat(listOfDF4)

	DFs = {}
	DFs["Wait"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Avail"] = bigDF4.drop(["CID","Wait","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Emptiness"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Wait","Probability","Section","Timeslot"],axis = 1)
	DFs["Enrol"] = bigDF4.drop(["CID","Avail","Wait","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Probability"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Emptiness","Wait","Section","Timeslot"],axis = 1)
	for key in DFs:
		orderDF = DFs[key].reindex(sorted(DFs[key].columns))
		model = LSTMModel(DFs[key],key)
		if saveModel(model,"SimpleTimeSeries LSTM Model 4 for "+key):
			print("SSimpleTimeSeries LSTM Model 4 for "+key+" Successfully!")

def SimpleTimeSeriesGRUModel(df5):
	df5 = df5.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
					"SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"], axis = 1)
	course = set(df5["CID"].values)
	listOfDF4 = []
	while len(course) >0:
		st = course.pop()
		sameCourseDF = df5.loc[df5['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			listOfDF4.append(create_dataset(sameSectionDF,2))
	bigDF4 = pd.concat(listOfDF4)

	DFs = {}
	DFs["Wait"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Avail"] = bigDF4.drop(["CID","Wait","Enrol","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Emptiness"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Wait","Probability","Section","Timeslot"],axis = 1)
	DFs["Enrol"] = bigDF4.drop(["CID","Avail","Wait","Quota","Emptiness","Probability","Section","Timeslot"],axis = 1)
	DFs["Probability"] = bigDF4.drop(["CID","Avail","Enrol","Quota","Emptiness","Wait","Section","Timeslot"],axis = 1)
	for key in DFs:
		orderDF = DFs[key].reindex(sorted(DFs[key].columns))
		model = GRUModel(DFs[key],key)
		if saveModel(model,"SimpleTimeSeries GRU Model 5 for "+key):
			print("SSimpleTimeSeries GRU Model 5 for "+key+" Successfully!")

def timelist(start, end, delta):
	l = []
	curr = start
	while curr < end:
		l.append(curr)
		curr = curr + delta
	return l
        
def getAllTimeslot(): 
	client = pymongo.MongoClient('localhost',27017)
	db = client.comp4332_Project
	allTSquery = db.course.distinct("SectionListWithTimeslot.Timeslot")
	allTS = []
	for entry in allTSquery:
		allTS.append(entry)
	return [min(allTS),max(allTS)]

def searchCourse_Section(courseID, section):
	try:
		client = pymongo.MongoClient('localhost',27017)
		db = client.comp4332_Project
		query = db.course.aggregate([{"$match": {"CID":courseID,"SectionListWithTimeslot.Section":{'$regex':'^L'+section}}},
                                    {"$unwind":"$SectionListWithTimeslot"},
									{"$project":{'CID':1,'CName': 1,'CoList': 1,'Co_requisite': 1,'Credit': 1, 'Exclusion': 1,
									'Pre_requisite': 1, 'Attributes':1, 'Avail':'$SectionListWithTimeslot.Avail',
									'Date_Time':'$SectionListWithTimeslot.Date_and_Time','Enrol':'$SectionListWithTimeslot.Enrol',
									'Quota':'$SectionListWithTimeslot.Quota','Room':'$SectionListWithTimeslot.Room',
									'Section':'$SectionListWithTimeslot.Section','Timeslot':'$SectionListWithTimeslot.Timeslot',
									'Wait':'$SectionListWithTimeslot.Wait'
									}},
									{"$sort":{"Timeslot":1}}
									],
                                   allowDiskUse = True)

		processed = []
		for entry in query:

			del entry['_id']
			n = entry["CID"]
			CIDbreak = courseClusters(n)
			processing = courseInfo(entry)

			trans = {}
			trans["CID"] = n
			trans["Timeslot"] = entry["Timeslot"]
			trans["Section"] = entry["Section"]
			trans["Credit"] = entry["Credit"]
			trans["Quota"] = entry["Quota"]
			trans["Enrol"] = entry["Enrol"]
			trans["Avail"] = entry["Avail"]

			if entry["Quota"] == 0:
			    trans["Probability"] = 0
			    trans["Emptiness"] = 0
			else:
			    trans["Probability"] = entry["Wait"]/entry["Quota"]
			    trans["Emptiness"] = entry["Avail"]/entry["Quota"]
			trans["Wait"] = entry["Wait"]

			newDict = {}
			newDict.update(CIDbreak)
			newDict.update(processing)
			newDict.update(trans)
			processed.append(newDict)

	except pymongo.errors.ConnectionFailure as error:
		print(error)

	df = pd.DataFrame(processed)
	df = df[df["Section"].str.contains('^L\d',regex = True)]
	return df

def Model1Prediction(cc, ln ,ts):
	required = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
	startend = getAllTimeslot()
	TSrange = timelist(startend[0],startend[1],timedelta(minutes = 30))

	df = searchCourse_Section(cc, ln)
	df.index = df.Timeslot
	df = df.reindex(TSrange)
	predictDF = df.drop(["CID","Section","Timeslot"],axis = 1)

	waitM = readModel("ANN Model 1 for Wait")
	emptiM = readModel("ANN Model 1 for Emptiness")
	enrolM = readModel("ANN Model 1 for Enrol")
	ProbM = readModel("ANN Model 1 for Probability")

	answer = predictDF.interpolate('linear')
	prediction = create_dataset(answer,2)
	if required > startend[1]:
		curr = pd.Series(prediction.iloc[-1,:])
		print(curr)
		k = int((required-startend[1])/timedelta(minutes = 30))
		for i in range(0,k):
			print(i)
			newCurr =  pd.Series(curr)
			newCurr["t-2 Enroll"] = curr["t-1 Enroll"]
			newCurr["t-2 Wait"] = curr["t-1 Wait"]
			newCurr["t-2 Emptiness"] = curr["t-1 Emptiness"]
			newCurr["t-2 Prob"] = curr["t-1 Prob"]
			newCurr["t-1 Enroll"] = curr["Enrol"]
			newCurr["t-1 Wait"] = curr["Wait"]
			newCurr["t-1 Emptiness"] = curr["Emptiness"]
			newCurr["t-1 Prob"] = curr["Probability"]
			X = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail"]).values.reshape(1,22)
			newCurr["Enrol"] = enrolM.predict(X, batch_size = 16)
			newCurr["Wait"] = waitM.predict(X, batch_size = 16)
			newCurr["Emptiness"] = emptiM.predict(X, batch_size = 16)
			newCurr["Probability"] = ProbM.predict(X, batch_size = 16)
			print(newCurr)
			curr = newCurr

		return curr["Wait"]
	elif required < startend[0]:
		print("Not enough data")
		return 0
	else:
		curr = pd.Series(prediction.loc[required])
		v = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail"]).values.reshape(1,22)
		ans = waitM.predict(v, batch_size = 16)
		return ans

def Model2Prediction(cc, ln ,ts):
	required = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
	startend = getAllTimeslot()
	TSrange = timelist(startend[0],startend[1],timedelta(minutes = 30))

	df = searchCourse_Section(cc, ln)
	df.index = df.Timeslot
	df = df.reindex(TSrange)
	predictDF = df.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
                "SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	predictDF = predictDF.drop(["CID","Section","Timeslot"],axis = 1)

	waitM = readModel(cc+"_L"+ln+"_Wait")
	emptiM = readModel(cc+"_L"+ln+"_Emptiness")
	enrolM = readModel(cc+"_L"+ln+"_Enrol")
	ProbM = readModel(cc+"_L"+ln+"_Probability")

	answer = predictDF.interpolate('linear')
	prediction = create_dataset(answer,2)
	if required > startend[1]:
		curr = pd.Series(prediction.iloc[-1,:])
		print(curr)
		k = int((required-startend[1])/timedelta(minutes = 30))
		for i in range(0,k):
			print(i)
			newCurr =  pd.Series(curr)
			newCurr["t-2 Enroll"] = curr["t-1 Enroll"]
			newCurr["t-2 Wait"] = curr["t-1 Wait"]
			newCurr["t-2 Emptiness"] = curr["t-1 Emptiness"]
			newCurr["t-2 Prob"] = curr["t-1 Prob"]
			newCurr["t-1 Enroll"] = curr["Enrol"]
			newCurr["t-1 Wait"] = curr["Wait"]
			newCurr["t-1 Emptiness"] = curr["Emptiness"]
			newCurr["t-1 Prob"] = curr["Probability"]
			X = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail"]).values.reshape(1,9)
			newCurr["Enrol"] = enrolM.predict(X, batch_size = 16)
			newCurr["Wait"] = waitM.predict(X, batch_size = 16)
			newCurr["Emptiness"] = emptiM.predict(X, batch_size = 16)
			newCurr["Probability"] = ProbM.predict(X, batch_size = 16)
			print(newCurr)
			curr = newCurr

		return curr["Wait"]
	elif required < startend[0]:
		print("Not enough data")
		return 0
	else:
		curr = pd.Series(prediction.loc[required])
		v = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail"]).values.reshape(1,9)
		ans = waitM.predict(v, batch_size = 16)
		return ans

def Model3Prediction(cc, ln ,ts):
	required = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
	startend = getAllTimeslot()
	TSrange = timelist(startend[0],startend[1],timedelta(minutes = 30))

	df = searchCourse_Section(cc, ln)
	df.index = df.Timeslot
	df = df.reindex(TSrange)
	predictDF = df.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
                "SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	predictDF = predictDF.drop(["CID","Section","Timeslot"],axis = 1)

	waitM = readModel("CourseDummyANN Model 3 for Wait")
	emptiM = readModel("CourseDummyANN Model 3 for Emptiness")
	enrolM = readModel("CourseDummyANN Model 3 for Enrol")
	ProbM = readModel("CourseDummyANN Model 3 for Probability")

	df3 = GetDFfromMongo()
	df3 = df3.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
					"SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	df3["newSection"] = df3["Section"][0:2]
	course = set(df3["CID"].values)
	listOfDF1 = []
	while len(course) >0:
		st = course.pop()
		sameCourseDF = df3.loc[df3['CID'] == st]
		section = set(sameCourseDF["Section"].values)
		while len(section) >0:
			OneSection = section.pop()
			sameSectionDF = sameCourseDF.loc[sameCourseDF['Section'] == OneSection]
			listOfDF1.append(create_dataset(sameSectionDF,2))
	bigDF3 = pd.concat(listOfDF1)

	client = pymongo.MongoClient('localhost',27017)
	db = client.comp4332_Project
	cidall = []
	allCIDQ = db.course.find({"SectionListWithTimeslot.Section":{'$regex':'^L\d'}}).distinct("CID")
	for entry in allCIDQ:
		cidall.append(entry)
	cidall = (bigDF3.CID.values)
	cidDict = dict.fromkeys(sorted(cidall), 0)
	cidDict[cc] = 1
	print(len(cidall))

	allSecQ = db.course.find({"SectionListWithTimeslot.Section":{'$regex':'^L\d'}}).distinct("SectionListWithTimeslot.Section")
	secall = []
	for entry in allSecQ:
		n = entry[0:2]
		secall.append(n)
	secall = set(bigDF3.newSection.values)
	if len(secall) == 1:
		secall = ["L0","L1"]
	secDict = dict.fromkeys(set(sorted(secall)), 0)
	secDict["L"+ln] = 1
	cidsr = pd.Series(list(cidDict.values()))
	secsr = pd.Series(list(secDict.values()))

	answer = predictDF.interpolate('linear')
	prediction = create_dataset(answer,2)
	if required > startend[1]:
		curr = pd.Series(prediction.iloc[-1,:])
		curr = curr.append(cidsr)
		curr = curr.append(secsr)
		print(curr)
		k = int((required-startend[1])/timedelta(minutes = 30))
		for i in range(0,k):
			print(i)
			newCurr =  pd.Series(curr)
			newCurr["t-2 Enroll"] = curr["t-1 Enroll"]
			newCurr["t-2 Wait"] = curr["t-1 Wait"]
			newCurr["t-2 Emptiness"] = curr["t-1 Emptiness"]
			newCurr["t-2 Prob"] = curr["t-1 Prob"]
			newCurr["t-1 Enroll"] = curr["Enrol"]
			newCurr["t-1 Wait"] = curr["Wait"]
			newCurr["t-1 Emptiness"] = curr["Emptiness"]
			newCurr["t-1 Prob"] = curr["Probability"]
			X = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,8+len(cidDict)+len(secDict))
			newCurr["Enrol"] = enrolM.predict(X, batch_size = 16)
			newCurr["Wait"] = waitM.predict(X, batch_size = 16)
			newCurr["Emptiness"] = emptiM.predict(X, batch_size = 16)
			newCurr["Probability"] = ProbM.predict(X, batch_size = 16)
			print(newCurr)
			curr = newCurr

		return curr["Wait"]
	elif required < startend[0]:
		print("Not enough data")
		return 0
	else:
		curr = pd.Series(prediction.loc[required])
		curr = curr.append(cidsr)
		curr = curr.append(secsr)
		v = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,8+len(cidDict)+len(secDict))
		ans = waitM.predict(v, batch_size = 16)
		return ans

def Model4Prediction(cc, ln ,ts):
	required = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
	startend = getAllTimeslot()
	TSrange = timelist(startend[0],startend[1],timedelta(minutes = 30))

	df = searchCourse_Section(cc, ln)
	df.index = df.Timeslot
	df = df.reindex(TSrange)
	predictDF = df.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
                "SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	predictDF = predictDF.drop(["CID","Section","Timeslot"],axis = 1)

	waitM = readModel("SimpleTimeSeries LSTM Model 4 for Wait")
	emptiM = readModel("SimpleTimeSeries LSTM Model 4 for Emptiness")
	enrolM = readModel("SimpleTimeSeries LSTM Model 4 for Enrol")
	ProbM = readModel("SimpleTimeSeries LSTM Model 4 for Probability")

	answer = predictDF.interpolate('linear')
	prediction = create_dataset(answer,2)
	if required > startend[1]:
		curr = pd.Series(prediction.iloc[-1,:])
		print(curr)
		k = int((required-startend[1])/timedelta(minutes = 30))
		for i in range(0,k):
			print(i)
			newCurr =  pd.Series(curr)
			newCurr["t-2 Enroll"] = curr["t-1 Enroll"]
			newCurr["t-2 Wait"] = curr["t-1 Wait"]
			newCurr["t-2 Emptiness"] = curr["t-1 Emptiness"]
			newCurr["t-2 Prob"] = curr["t-1 Prob"]
			newCurr["t-1 Enroll"] = curr["Enrol"]
			newCurr["t-1 Wait"] = curr["Wait"]
			newCurr["t-1 Emptiness"] = curr["Emptiness"]
			newCurr["t-1 Prob"] = curr["Probability"]
			X = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,2,4)
			newCurr["Enrol"] = enrolM.predict(X, batch_size = 16)
			newCurr["Wait"] = waitM.predict(X, batch_size = 16)
			newCurr["Emptiness"] = emptiM.predict(X, batch_size = 16)
			newCurr["Probability"] = ProbM.predict(X, batch_size = 16)
			print(newCurr)
			curr = newCurr

		return curr["Wait"]
	elif required < startend[0]:
		print("Not enough data")
		return 0
	else:
		curr = pd.Series(prediction.loc[required])
		v = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,2,4)
		ans = waitM.predict(v, batch_size = 16)
		return ans

def Model5Prediction(cc, ln ,ts):
	required = datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
	startend = getAllTimeslot()
	TSrange = timelist(startend[0],startend[1],timedelta(minutes = 30))

	df = searchCourse_Section(cc, ln)
	df.index = df.Timeslot
	df = df.reindex(TSrange)
	predictDF = df.drop(["BUS","Credit","IPO","IsCommonCore","IsSSCCommonCore","OTR",
                "SENG","SHSS","SSCI","hasCoList","hasExclusion","hasPre-re","startCode"],axis = 1)
	predictDF = predictDF.drop(["CID","Section","Timeslot"],axis = 1)

	waitM = readModel("SimpleTimeSeries GRU Model 5 for Wait")
	emptiM = readModel("SimpleTimeSeries GRU Model 5 for Emptiness")
	enrolM = readModel("SimpleTimeSeries GRU Model 5 for Enrol")
	ProbM = readModel("SimpleTimeSeries GRU Model 5 for Probability")

	answer = predictDF.interpolate('linear')
	prediction = create_dataset(answer,2)
	if required > startend[1]:
		curr = pd.Series(prediction.iloc[-1,:])
		print(curr)
		k = int((required-startend[1])/timedelta(minutes = 30))
		for i in range(0,k):
			print(i)
			newCurr =  pd.Series(curr)
			newCurr["t-2 Enroll"] = curr["t-1 Enroll"]
			newCurr["t-2 Wait"] = curr["t-1 Wait"]
			newCurr["t-2 Emptiness"] = curr["t-1 Emptiness"]
			newCurr["t-2 Prob"] = curr["t-1 Prob"]
			newCurr["t-1 Enroll"] = curr["Enrol"]
			newCurr["t-1 Wait"] = curr["Wait"]
			newCurr["t-1 Emptiness"] = curr["Emptiness"]
			newCurr["t-1 Prob"] = curr["Probability"]
			X = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,2,4)
			newCurr["Enrol"] = enrolM.predict(X, batch_size = 16)
			newCurr["Wait"] = waitM.predict(X, batch_size = 16)
			newCurr["Emptiness"] = emptiM.predict(X, batch_size = 16)
			newCurr["Probability"] = ProbM.predict(X, batch_size = 16)
			print(newCurr)
			curr = newCurr

		return curr["Wait"]
	elif required < startend[0]:
		print("Not enough data")
		return 0
	else:
		curr = pd.Series(prediction.loc[required])
		v = curr.drop(["Wait","Enrol","Emptiness","Probability","Avail","Quota"]).values.reshape(1,2,4)
		ans = waitM.predict(v, batch_size = 16)
		return ans


