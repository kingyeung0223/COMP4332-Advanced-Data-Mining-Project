# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy

# Idea, course item would be returned first when scanning department pages
class course_item(scrapy.Item):
    CID = scrapy.Field()
    CName = scrapy.Field()
    CDescription = scrapy.Field()
    Credit = scrapy.Field()
    Pre_requisite = scrapy.Field()
    Attributes = scrapy.Field()
    CoList = scrapy.Field()
    Co_requisite = scrapy.Field()
    Exclusion = scrapy.Field()
    SectionList = scrapy.Field()

# section info of courses in a single timeslot would be returned
# when scanning tables of courses of a department in a single timeslot
class section_item(scrapy.Item):
    # fields that should appear in SectionListWithTimeslot in mongodb
    Timeslot = scrapy.Field()
    Section = scrapy.Field()
    Date_and_Time = scrapy.Field()
    Room = scrapy.Field()
    Instructor = scrapy.Field()
    Quota = scrapy.Field()
    Enrol = scrapy.Field()
    Avail = scrapy.Field()
    Wait = scrapy.Field()
    Remark = scrapy.Field()

    # fields for mapping courses during insertion
    CID = scrapy.Field()

