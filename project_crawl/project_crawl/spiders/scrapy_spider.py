import scrapy, datetime, winsound, time, copy
from project_crawl.items import section_item
from project_crawl.items import course_item

class Webpage_spider(scrapy.Spider):
    name = 'course_spider'
    base_url = ""

    # function to check if info (could be a list or variable) already exist in a list
    # If yes, the original list would be returned without any changes
    # If no, new_info would be appended to the original list
    def is_row_info_same(self, original_list, new_info):
        sample = []
        if isinstance(original_list, type(sample)):
            if not isinstance(new_info, type(sample)):
                if new_info in original_list:
                    return original_list
                original_list.append(new_info)
            else:
                for element in new_info:
                    if element not in original_list:
                        original_list.append(element)
            return original_list

    # define the url for starting requuest
    def start_requests(self):
        if self.urls[-1] == "/":
            self.base_url = self.urls
        else:
            self.base_url = self.urls + "/"
        yield scrapy.Request('%s' % self.urls)

    # sending request to different timeslots
    def parse(self, response):
        url_list = response.xpath('//li/a/@href').extract()
        for i in range(len(url_list)):
            url_list[i] = self.base_url + url_list[i]
            next_step = response.follow(url=url_list[i], callback=self.parse_depart)
            next_step.meta['base_url'] = url_list[i]

            # get and pass timeslot info to next recursion level
            time_info = url_list[i].split("/")
            m = int(time_info[6])
            d = int(time_info[7])
            h = int(time_info[8])
            mi = int(time_info[9])
            next_step.meta['Timeslot'] = datetime.datetime(2018, m, d, h, mi, 0)
            yield next_step

    # sending request to different departments
    def parse_depart(self, response):
        base_url = response.meta['base_url']
        timeslot = response.meta['Timeslot']
        url_list = response.xpath('//a/@href').extract()
        for i in range(len(url_list)):
            url_list[i] = base_url + "/" + url_list[i]
            next_step = response.follow(url=url_list[i], callback=self.parse_course)
            next_step.meta['Timeslot'] = timeslot
            yield next_step

    # scanning info of courses in a single department
    def parse_course(self, response):
        total_course = len(response.xpath("//div[@class = 'course']/@class").extract())
        # accessing different courses on the page
        for i in range(1, total_course+1):
            # Colecting course info with course_item()
            # if necessary, data cleaning are performed in pipelines
            course = course_item()
            header_info = response.xpath("//div[@id='classes']/div[{}]/h2/text()".format(i)).extract_first()
            header_info = header_info.split("-")
            course['CID'] = header_info[0]
            #print("CID: ", course['CID'])

            course['CName'] = header_info[1]
            #print("CName: ", course['CName'])

            course['CDescription'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() = 'DESCRIPTION']/td/text()".format(i)).extract_first()
            #print("CDesription: ", course['CDescription'])

            # Cleaning would be performed in pipelines
            course['Credit'] = response.xpath("//div[@id='classes']/div[{}]/h2/text()".format(i)).extract_first()
            #print("Uncleaned Credit: ", course['Credit'])

            # Note: Multiple prerequisites are written as single row on the page with "AND" "OR"
            # may have to consider splitting with "AND" "OR" later
            course['Pre_requisite'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() = 'PRE-REQUISITE']/td/text()".format(i)).extract_first()
            #print("Pre-re: ", course['Pre_requisite'])

            # Note: Multiple attributes are written in multiple rows
            # Attributes are extracted as a list
            course['Attributes'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() ='ATTRIBUTES']/td/text()".format(i)).extract()
            #print("Attributes: ", course['Attributes'])

            # Note: Have to check if always colist with only 1 course
            # OR multiple courses in different lines
            # OR multiple lines in single line
            course['CoList'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() ='CO-LIST WITH']/td/text()".format(i)).extract_first()
            #print("CoList: ", course['CoList'])

            # Note: Multiple co-requisite are written as single row with "AND", "OR"
            # may have to consider splitting with "AND", "OR" later
            course['Co_requisite'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() ='CO-REQUISITE']/td/text()".format(i)).extract_first()
            #print("Co_re: ", course['Co_requisite'])

            # Note: Multiple exclusion are written as single row, separated by ","
            # may have to consider further splitting in the cleaning process
            course['Exclusion'] = response.xpath("//div[@id='classes']/div[{}]//tr[th/text() ='EXCLUSION']/td/text()".format(i)).extract_first()
            #print("Exclusion: ", course['Exclusion'])

            # Preparation for iterating rows of a table
            # create section_item to store section info
            # A row and its next row could belong to the same section
            # In this case, section dependent field should be kept constant during iterating course table
            # So field dependent variable have to be declared before the loop
            # Remark is now found to be section dependent
            section_list = []
            section = section_item()
            section['CID'] = course['CID']
            section['Timeslot'] = response.meta['Timeslot']
            #print("timeslot: ", section['Timeslot'])

            section['Section'] = section['Date_and_Time'] = section['Room'] = section['Instructor'] = ""
            section['Quota'] = section['Enrol'] = section['Avail'] = section['Wait'] = section['Remark'] = ""

            # total number of rows in each table including header row
            total_rows = len(response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr".format(i)).extract())
            # print("total rows: ", total_rows)

            # storing course index for iteration
            c_num = i

            next_is_same_sect = False
            row_remaining = 0
            # iterating rows of a table
            # the first row is header which is out of our interest, so j starts with 2
            for j in range(2, total_rows+1):
                if not next_is_same_sect:
                    section['Section'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[1]/text()".format(c_num, j)).extract_first()
                    #print("Section: ", section['Section'])
                    # print("\n i am iterating the {}th row\n".format(j-1))
                    # print(section['Section'])
                    # time.sleep(3)

                    section['Quota'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[5]//text()".format(c_num, j)).extract_first()
                    #print("Quota: ", section['Quota'])

                    section['Enrol'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[6]/text()".format(c_num, j)).extract_first()
                    #print("Enrol: ", section['Enrol'])

                    section['Avail'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[7]//text()".format(c_num, j)).extract_first()
                    #print("Avail: ", section['Avail'])

                    section['Wait'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[8]//text()".format(c_num, j)).extract_first()
                    #print("Wait: ", section['Wait'])

                    # For safety, these fields should be a list
                    section['Instructor'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[4]/text()".format(c_num, j)).extract()
                    #print("Instruct: ", section['Instructor'])

                    section['Date_and_Time'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[2]/text()".format(c_num, j)).extract()
                    #print("DateTime: ", section['Date_and_Time'])

                    section['Room'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[3]/text()".format(c_num, j)).extract()
                    #print("Room: ", section['Room'])

                    # There are 2 types of remarks: "consent required", "popup details about each section"
                    section['Remark'] = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]//div[contains(@class, 'consent')]/div/text()".format(c_num, j)).extract()
                    type_2_remark = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]//div[contains(@class, 'classnotes')]/div/text()".format(c_num, j)).extract_first()
                    if type_2_remark is not None:
                        section['Remark'].append(type_2_remark)
                    #print("Remark: ", section['Remark'])

                    test_cond = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[5]/@rowspan".format(c_num, j)).extract_first()
                    if test_cond is None:
                        next_is_same_sect = False
                        section_list.append(copy.deepcopy(section))
                    else:
                        row_remaining = int(response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[5]/@rowspan".format(c_num, j)).extract_first())
                        row_remaining -= 1
                        next_is_same_sect = True
                        #print("Remaining rows: ", row_remaining)
                else:
                    temp_Instructor = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[3]/text()".format(c_num, j)).extract_first()
                    temp_Data_And_Time = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[1]/text()".format(c_num, j)).extract_first()
                    temp_Room = response.xpath("//div[@id='classes']/div[{}]//table[@class = 'sections']//tr[{}]/td[2]/text()".format(i, j)).extract_first()
                    section['Instructor'] = self.is_row_info_same(section['Instructor'], temp_Instructor)
                    section['Date_and_Time'] = self.is_row_info_same(section['Date_and_Time'], temp_Data_And_Time)
                    section['Room'] = self.is_row_info_same(section['Room'], temp_Room)
                    row_remaining -= 1
                    next_is_same_sect = (row_remaining > 0)
                    if not next_is_same_sect:
                        section_list.append(copy.deepcopy(section))
            course['SectionList'] = section_list
            yield course
