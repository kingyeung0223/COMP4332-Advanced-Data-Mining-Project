# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html

import pymongo
from project_crawl.items import course_item
from project_crawl.items import section_item


class ProjectCrawlPipeline(object):
    # This would be called every time a spider is initiated
    # connection to database
    client = pymongo.MongoClient()
    db = "comp4332_Project"
    def open_spider(self, spider):
        try:
            self.client = pymongo.MongoClient("mongodb://localhost:27017")
            self.db = self.client[self.db]
        except pymongo.errors.ConnectionFailure as error:
            print(error)

    # This would be called every time when a spider is close
    # close connection to database
    def close_spider(self, spider):
        try:
            self.client.close()
        except pymongo.errors.ConnectionFailure as error:
            print(error)

    def process_item(self, item, spider):
        # calling different functions to handle course_item and section_item
        self.insert_course(item)
        for i in range(len(item['SectionList'])):
            self.insert_section(item['SectionList'][i])
        return item

    def insert_course(self, item):
        # removing last blank-space from CID
        print("\n INSERTING A COURSE\n")
        item['CID'] = item['CID'][:-1]

        # check if the course already exists in db
        course_in_db = self.db.course.find({"CID": item['CID']}).count()

        if int(course_in_db) == 0:
            # Extracting CName from header
            item['CName'] = item['CName'][1:-10]
            # Extracting Credit No from header
            if type(item['Credit']) != int:
                # credit number position shifts because of "units" and unit
                if item['Credit'][-2] == "s":
                    item['Credit'] = int(item['Credit'][-8])
                elif item['Credit'][-2] == "t":
                    item['Credit'] = int(item['Credit'][-7])
                # insert into database
                self.db.course.insert(
                    {
                        "CID": item['CID'],
                        "CName": item['CName'],
                        "CDescription": item['CDescription'],
                        "Credit": item['Credit'],
                        "Pre_requisite": item['Pre_requisite'],
                        "Attributes": item['Attributes'],
                        "CoList": item['CoList'],
                        "Co_requisite": item['Co_requisite'],
                        "Exclusion": item['Exclusion'],
                        "SectionListWithTimeslot": []
                    }
                )
                # CDescription, Pre-requisite, Attributes, CoList, Co_requisite, Exclusion havent been processed yet

    def insert_section(self, item):
        # inserting section_item into database
        # removing last blank-space from CID
        print("\n INSERTING A SECTION\n")
        item['CID'] = item['CID'][:-1]

        item['Quota'] = int(item['Quota'])
        item['Enrol'] = int(item['Enrol'])
        item['Avail'] = int(item['Avail'])
        item['Wait'] = int(item['Wait'])

        # insert into database
        self.db.course.update_one(
            {"CID": item['CID']},
            {
                "$push": {
                    "SectionListWithTimeslot": {
                        "Timeslot": item['Timeslot'],
                        "Section": item['Section'],
                        "Date_and_Time": item['Date_and_Time'],
                        "Instructor": item['Instructor'],
                        "Room": item['Room'],
                        "Quota": item['Quota'],
                        "Enrol": item['Enrol'],
                        "Avail": item['Avail'],
                        "Wait": item['Wait'],
                        "Remark": item['Remark']
                    }
                }
            }
        )
