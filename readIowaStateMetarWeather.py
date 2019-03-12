# -*- coding: utf-8 -*-

import requests
import csv
from datetime import datetime, timedelta
import time
from urllib.error import URLError
from metar import Metar
from scrapy.selector import Selector
import re
import calendar
import pandas as pd
import stations
import readUtils

pressureSelector=re.compile("^Q[0-9]{4}$")
tempSelector=re.compile("^(M)?[0-9]{2}/(M)?[0-9]{2}$")
timeSelector=re.compile("^[0-9]{6}Z$")
visibilitySelector=re.compile("^[0-9]{4}$")
windSelector=re.compile("^[0-9]{5}(G[0-9][0-9])?KT$")
windSelectorVRB=re.compile("^VRB[0-9]{2}(G[0-9][0-9])?KT$")
windDirectionVariabilitySelector=re.compile("^[0-9]{3}V[0-9]{3}$")

consolidated=[]
allmetars=[]
rowno=0

dateRanges=[
    [2010,1,1,2011,12,31],
    [2012,1,1,2013,12,31],
    [2014,1,1,2015,12,31],
    [2016,1,1,2017,9,30],
]

for dates in dateRanges:
    startYear=dates[0]
    startMonth=dates[1]
    startDay=dates[2]
    endYear=dates[3]
    endMonth=dates[4]
    endDay=dates[5]

    for stationCode in stations.weatherStationLookup:
        print('Reading data for station {}'.format(stationCode))
        url="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={}&data=all&year1={}&month1={}&day1={}&year2={}&month2={}&day2={}&tz=Etc%2FUTC&format=onlycomma&latlon=no&direct=no&report_type=1&report_type=2". \
            format(stationCode, startYear, startMonth, startDay, endYear, endMonth, endDay)

        respData, status_code, responseHeaders = readUtils.fetchWithRetry(url)

        if status_code==200 and respData is not None:
            lineCount=0
            with open('metar-tmp-preprocessed.csv', 'w') as destfile:
                lineBuffer=None
                lines=respData.split('\n')
                for line in lines:
                    if line[0:4]=='    ':
                        if lineBuffer is not None:
                            lineBuffer=lineBuffer.strip()+' '+line.strip()
                    else:
                        if lineBuffer is not None:
                            destfile.write(lineBuffer+"\n")
                            lineCount=lineCount+1
                        lineBuffer=line
                if lineBuffer is not None:
                    destfile.writelines(lineBuffer+"\n")
                    lineCount=lineCount+1

            if lineCount>0:
                with open('metar-tmp-preprocessed.csv', 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        stationCode=row['station']
                        dt=row['valid']
                        metar=row[' metar']

                        rowno=rowno+1

                        print('{} {} {} {}'.format(rowno, dt, stationCode, metar))

                        if stationCode is not None and dt is not None and metar is not None:

                            formattedDT=dt[0:4]+"-"+dt[5:7]+"-"+dt[8:10]+"T"+dt[11:13]+":"+dt[14:16]+":00Z"
                            allmetars.append([stationCode, formattedDT, metar])

                            stationName=None
                            latitude=None
                            longitude=None
                            ele=None
                            if stationCode in stations.weatherStationLookup:
                                sdata=stations.weatherStationLookup[stationCode]
                                stationName=sdata[0]
                                latitude=sdata[2]
                                longitude=sdata[3]
                                ele=sdata[4]

                            temperature = None
                            dewpt = None
                            barometricPressure = None
                            visibilityMetres = None
                            windDirection = None
                            windDirectionFrom = None
                            windDirectionTo = None
                            windSpeed = None
                            windGustSpeed = None

                            windComponent = None

                            split=metar.split()
                            for term in split:
                                if len(term)>0: # ignore any multi-space
                                    if term=='METAR' or term=='AUTO' or term==stationCode or timeSelector.match(term) or term=='NIL':
                                        pass # ignore standard items
                                    else:
                                        if pressureSelector.match(term):
                                            parsed=Metar.Metar(term)
                                            barometricPressure = parsed.press.value("MB") if parsed.press is not None else None

                                        elif tempSelector.match(term):
                                            parsed=Metar.Metar(term)
                                            temperature = parsed.temp.value("C") if parsed.temp is not None else None
                                            dewpt = parsed.dewpt.value("C") if parsed.dewpt is not None else None

                                        elif windSelector.match(term) or windSelectorVRB.match(term):
                                            parsed=Metar.Metar(term)
                                            windComponent=term
                                            windDirection = parsed.wind_dir.value() if parsed.wind_dir is not None else None
                                            windSpeed = parsed.wind_speed.value("KT")*0.514444 if parsed.wind_speed is not None else None
                                            windGustSpeed = parsed.wind_gust.value("KT")*0.514444 if parsed.wind_gust is not None else None

                                        elif windDirectionVariabilitySelector.match(term) and windComponent is not None:
                                            parsed=Metar.Metar(windComponent+" "+term)
                                            windDirectionFrom = parsed.wind_dir_from.value() if parsed.wind_dir_from is not None else None
                                            windDirectionTo = parsed.wind_dir_to.value() if parsed.wind_dir_to is not None else None

                                        elif visibilitySelector.match(term):
                                            visibilityMetres=float(term)

                                        elif term=='CAVOK':
                                            visibilityMetres=10000

                                        # else:
                                            # print('Unrecognised term {}'.format(term))

                            row=[formattedDT, 'WeatherObserved', stationCode,
                                             stationName, barometricPressure, temperature, dewpt,
                                             visibilityMetres, windDirection, windDirectionFrom, windDirectionTo,
                                             windSpeed, windGustSpeed, latitude, longitude, ele]

                            consolidated.append(row)



df=pd.DataFrame(consolidated, columns=['dateObserved', 'type', 'stationCode', 'stationName',
                     'barometricPressure',
                     'temperature', 'dewPointTemperature', 'visibilityMetres',
                     'windDirection', 'windDirectionFrom', 'windDirectionTo', 'windSpeed',
                     'windGustSpeed', 'latitude', 'longitude', 'elevation'])
df.to_csv('metar/metarDecoded-amalgamated.csv')

df1=pd.DataFrame(allmetars, columns=['stationCode', 'dateObserved', 'metar'])
df1.to_csv('metar/metarRaw-amalgamated.csv')

#
# ...         print(row['first_name'], row['last_name'])
#
# def decodeMetar(raw):
# #     station,valid,tmpf, dwpf, relh, drct, sknt, p01i, alti, mslp, vsby, gust, skyc1, skyc2, skyc3, skyc4, skyl1, skyl2, skyl3, skyl4, presentwx, metar
# # EGPK,2010-01-01 00:50,19.40,17.60,92.57,0.00,0.00,M,29.83,M,6.21,M,FEW,M,M,M,3000.00,M,M,M,M,EGPK 010050Z 00000KT 9999 FEW030 M07/M08 Q1010
# # EGKK,2010-01-01 01:20,33.80,30.20,86.49,10.00,9.00,M,29.53,M,6.21,M,FEW,M,M,M,3800.00,M,M,M,M,EGKK 010120Z 01009KT 9999 FEW038 01/M01 Q1000
# # EGLC,2010-01-01 01:20,35.60,30.20,80.51,20.00,8.00,M,29.56,M,6.21,M,SCT,M,M,M,3500.00,M,M,M,M,EGLC 010120Z AUTO 02008KT 9999 SCT035/// 02/M01 Q1001 RESN
#
#     wrappedMetar=raw.strip()
#     dt=wrappedMetar[0:12]
#     metar=wrappedMetar[13:-1]
#             # print('Processing {}'.format(metar))
#             #'METAR EGLC 140820Z AUTO 28012KT 9999 SCT044 03/00 Q1018'
#             try:
#
#                 # wrappedMetar='201701180320 METAR EGLC 180320Z AUTO 09005KT 010V110 8000 NCD 00/M03 Q1039='
#                 # dt=wrappedMetar[0:12]
#                 # metar=wrappedMetar[13:-1]
#
#                 formattedDT=dt[0:4]+"-"+dt[4:6]+"-"+dt[6:8]+"T"+dt[8:10]+":"+dt[10:12]+":00Z"
#                 # 2012-01-13T01:00:00Z
#
#                 temperature = None
#                 dewpt = None
#                 barometricPressure = None
#                 visibilityMetres = None
#                 windDirection = None
#                 windDirectionFrom = None
#                 windDirectionTo = None
#                 windSpeed = None
#                 windGustSpeed = None
#
#                 windComponent = None
#
#                 split=metar.split()
#                 for term in split:
#                     if len(term)>0: # ignore any multi-space
#                         if term=='METAR' or term=='AUTO' or term==stationCode or timeSelector.match(term) or term=='NIL':
#                             pass # ignore standard items
#                         else:
#                             if pressureSelector.match(term):
#                                 parsed=Metar.Metar(term)
#                                 barometricPressure = parsed.press.value("MB") if parsed.press is not None else None
#
#                             elif tempSelector.match(term):
#                                 parsed=Metar.Metar(term)
#                                 temperature = parsed.temp.value("C") if parsed.temp is not None else None
#                                 dewpt = parsed.dewpt.value("C") if parsed.dewpt is not None else None
#
#                             elif windSelector.match(term) or windSelectorVRB.match(term):
#                                 parsed=Metar.Metar(term)
#                                 windComponent=term
#                                 windDirection = parsed.wind_dir.value() if parsed.wind_dir is not None else None
#                                 windSpeed = parsed.wind_speed.value("KT")*0.514444 if parsed.wind_speed is not None else None
#                                 windGustSpeed = parsed.wind_gust.value("KT")*0.514444 if parsed.wind_gust is not None else None
#
#                             elif windDirectionVariabilitySelector.match(term) and windComponent is not None:
#                                 parsed=Metar.Metar(windComponent+" "+term)
#                                 windDirectionFrom = parsed.wind_dir_from.value() if parsed.wind_dir_from is not None else None
#                                 windDirectionTo = parsed.wind_dir_to.value() if parsed.wind_dir_to is not None else None
#
#                             elif visibilitySelector.match(term):
#                                 visibilityMetres=float(term)
#
#                             elif term=='CAVOK':
#                                 visibilityMetres=10000
#
#                             # else:
#                                 # print('Unrecognised term {}'.format(term))
#
#                 row=[formattedDT, 'WeatherObserved', stationCode,
#                                  stationName, barometricPressure, temperature, dewpt,
#                                  visibilityMetres, windDirection, windDirectionFrom, windDirectionTo,
#                                  windSpeed, windGustSpeed, latitude, longitude, ele]
#
#
# # This function will return the content sent back by the HTML request made. The request is made to obtain weather data in a csv
# # file for a distinct time and a particular station.
# def retrieveMetarData(siteID, fromDate, toDate):
#     content=[]
# #    print('Retrieving '+str(siteID)+" "+date+" "+timeDate)
#
#     start=re.compile("^[0-9]{12} METAR")
#     retryCount=5
#
#     reading=True
#     while reading:
#         try:
#             reading=False
#
#             url=str("http://www.ogimet.com/display_metars2.php?lang=en&lugar={}&tipo=SA&ord=DIR&nil=SI&fmt=txt&ano={}&mes={}&day={}&hora=00&anof={}&mesf={}&dayf={}&horaf=23&minf=59&send=send"
#                 .format(siteID, fromDate.year, fromDate.month, fromDate.day, toDate.year, toDate.month, toDate.day))
#
#             #            url=str("http://www.ogimet.com/display_metars2.php?lang=en&lugar=EGLC&tipo=SA&ord=DIR&nil=SI&fmt=txt&ano=2012&mes=5&day=1&hora=00&anof=2012&mesf=5&dayf=31&horaf=23&minf=59&send=send"
#             #     .format(siteID, fromDate.year, fromDate.month, fromDate.day, toDate.year, toDate.month, toDate.day))
#
#             # payload= {'Type':'Observation', 'ObservationSiteID': siteID, 'Date': date, 'PredictionTime':timeDate}
#             # headers = {'Content-Type': 'application/x-www-form-urlencoded',
#             #              'Accept': 'application/x-www-form-urlencoded'}
#             print('Requesting {}'.format(url))
#             r = requests.get(url)
#             quotaLimited=False
#             if r.status_code==200:
#                 responseText=r.text
#                 preContent=Selector(text=responseText).xpath('//pre/text()').extract()
#                 if preContent is not None and isinstance(preContent, list) and len(preContent)>0:
#                     for section in preContent:
#                         lines=section.splitlines()
#                         inmetar=False
#                         curmetar=None
#                         for line in lines:
#                             sline=line.rstrip()
#                             if len(sline)==0:
#                                 inmetar=False
#                                 curmetar=None
#
#                             elif sline[0]=='#':
#                                 inmetar=False
#                                 curmetar=None
#                                 if sline=='#Sorry, Your quota limit for slow queries rate has been reached':
#                                     reading=True
#                                     print('Retry - quota limited')
#                                     time.sleep(15)
#                                     quotaLimited=True
#                                     break
#
#                             else:
#
#                                 handled=False
#                                 if start.match(sline):
#                                     if sline[-1]=='=':
#                                         # print('Full metar {}'.format(sline))
#                                         handled=True
#                                         content.append(sline)
#                                         inmetar=False
#                                     else:
#                                         curmetar=sline
#                                         inmetar=True
#                                         # print('Initial metar {}'.format(sline))
#                                         handled=True
#
#                                 if inmetar and not handled:
#                                     curmetar=curmetar+' '+sline.strip()
#                                     # print('Appended metar {}'.format(sline.strip()))
#                                     handled=True
#                                     if sline[-1]=='=':
#                                         # print('Multiline metar {}'.format(curmetar))
#                                         content.append(curmetar)
#                                         inmetar=False
#                                         curmetar=None
#                                     handled=True
#
#                                 if not handled:
#                                     print(sline)
#
#             else:
#                 print('Status code {}'.format(r.status_code))
#
#             if not quotaLimited and retryCount>0 and len(content)==0:
#                 retryCount=retryCount-1
#                 time.sleep(60)
#                 reading=True
#
#         except URLError:
#             time.sleep(60)
#             reading=True
#             continue
#         except TimeoutError:
#             time.sleep(120)
#             reading=True
#             continue
#         except ConnectionResetError:
#             time.sleep(60)
#             reading=True
#             continue
#         except ValueError as e:
#             time.sleep(60)
#             reading=True
#             continue
#
#     return content
#
#
# # This function will take in an integer value for visibility and converts it into an appropriate string.
# # def visibilityCheck(visibilityNo):
# #     visibilityText = ""
# #     if not visibilityNo:
# #         visibilityText = "unknown"
# #     if visibilityNo > 0 and visibilityNo < 1000:
# #         visibilityText = "veryPoor"
# #     if visibilityNo > 1000 and visibilityNo <= 4000:
# #         visibilityText = "poor"
# #     if visibilityNo > 4000 and visibilityNo <= 10000:
# #         visibilityText = "moderate"
# #     if visibilityNo > 10000 and visibilityNo <= 20000:
# #         visibilityText = "good"
# #     if visibilityNo > 20000 and visibilityNo <= 40000:
# #         visibilityText = "veryGood"
# #     if visibilityNo > 40000:
# #         visibilityText = "excellent"
# #
# #     return visibilityText
#
#
# # This function will take in a stationID and a timeframe to retrieve all the weather data during that period and harmonise it.
# def readMetarHistory(siteAlias, siteID, startDate, endDate):
#     dataSet   = []
#
#     print('Processing {} ({})'.format(siteAlias, siteID))
#     # startDateParsed = datetime.strptime(startDate, '%Y-%m-%d')
#     endDateParsed = datetime.strptime(endDate, '%Y-%m-%d')
#
#     # normalInterval = timedelta(days=30)
#
#     dateIterator = datetime.strptime(startDate, '%Y-%m-%d')
#
#     # i = 0
#     finished=False
#     while not finished:
#
#         if dateIterator>endDateParsed:
#             finished=True
#         else:
#             # interval=normalInterval
#             # outstandingDays=endDateParsed-dateIterator+timedelta(days=1)
#             # if outstandingDays<interval:
#             #     interval=outstandingDays
#             startWeekday,monthLength = calendar.monthrange(dateIterator.year, dateIterator.month)
#             toDate=datetime(dateIterator.year, dateIterator.month, monthLength)
#             if toDate>endDateParsed:
#                 toDate=endDateParsed
#
#             print('Run between {} - {}'.format(dateIterator, toDate))
#
#             metarContent = None
#             metarContent = retrieveMetarData(siteID, dateIterator, toDate)
#             print('Returned {} records'.format(len(metarContent)))
#
#             if metarContent!=None:
#                 dataSet.extend(metarContent)
#
#             dateIterator=toDate+timedelta(days=1)
#
#     return dataSet
#
# # This function will take in the following inputs: list of harmonised weather data, station code, start and end dates. It will then write data to a
# # csv file according to the inputs of the function.
# def writeToRawText(metarData, stationCode, start, end):
#
#     raw='metar/metarData-'+str(stationCode)+'-'+start+'-'+end+'.txt'
#     with open(raw, 'w') as datafile:
#         datafile.writelines("%s\n" % l for l in metarData)
#         # for line in metarData:
#         #     datafile.write(line)
#
#         datafile.close()
#
# def readFromRawText(stationCode, start, end):
#     metarData=None
#
#     raw='metar/metarData-'+str(stationCode)+'-'+start+'-'+end+'.txt'
#     with open(raw, 'r') as datafile:
#         metarData=datafile.readlines()
#         datafile.close()
#
#     return metarData
#
# def writeToCsv(metarData, stationName, stationCode, start, end, latitude, longitude, ele):
#
#     rows=[]
#
#     filename='metar/metarData-'+str(stationCode)+'-'+start+'-'+end+'.csv'
#
#     pressureSelector=re.compile("^Q[0-9]{4}$")
#     tempSelector=re.compile("^(M)?[0-9]{2}/(M)?[0-9]{2}$")
#     timeSelector=re.compile("^[0-9]{6}Z$")
#     visibilitySelector=re.compile("^[0-9]{4}$")
#     windSelector=re.compile("^[0-9]{5}(G[0-9][0-9])?KT$")
#     windSelectorVRB=re.compile("^VRB[0-9]{2}(G[0-9][0-9])?KT$")
#     windDirectionVariabilitySelector=re.compile("^[0-9]{3}V[0-9]{3}$")
#
#     with open(filename, 'w') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['dateObserved', 'type', 'stationCode', 'siteName',
#                          'barometricPressure',
#                          'temperature', 'dewPointTemperature', 'visibilityMetres',
#                          'windDirection', 'windDirectionFrom', 'windDirectionTo', 'windSpeed',
#                          'windGustSpeed', 'latitude', 'longitude', 'elevation'])
#         i= 0
#         while i < len(metarData):
#             #'201701140820 METAR EGLC 140820Z AUTO 28012KT 9999 SCT044 03/00 Q1018=
#             wrappedMetar=metarData[i].rstrip()
#             dt=wrappedMetar[0:12]
#             metar=wrappedMetar[13:-1]
#             # print('Processing {}'.format(metar))
#             #'METAR EGLC 140820Z AUTO 28012KT 9999 SCT044 03/00 Q1018'
#             try:
#
#                 # wrappedMetar='201701180320 METAR EGLC 180320Z AUTO 09005KT 010V110 8000 NCD 00/M03 Q1039='
#                 # dt=wrappedMetar[0:12]
#                 # metar=wrappedMetar[13:-1]
#
#                 formattedDT=dt[0:4]+"-"+dt[4:6]+"-"+dt[6:8]+"T"+dt[8:10]+":"+dt[10:12]+":00Z"
#                 # 2012-01-13T01:00:00Z
#
#                 temperature = None
#                 dewpt = None
#                 barometricPressure = None
#                 visibilityMetres = None
#                 windDirection = None
#                 windDirectionFrom = None
#                 windDirectionTo = None
#                 windSpeed = None
#                 windGustSpeed = None
#
#                 windComponent = None
#
#                 split=metar.split()
#                 for term in split:
#                     if len(term)>0: # ignore any multi-space
#                         if term=='METAR' or term=='AUTO' or term==stationCode or timeSelector.match(term) or term=='NIL':
#                             pass # ignore standard items
#                         else:
#                             if pressureSelector.match(term):
#                                 parsed=Metar.Metar(term)
#                                 barometricPressure = parsed.press.value("MB") if parsed.press is not None else None
#
#                             elif tempSelector.match(term):
#                                 parsed=Metar.Metar(term)
#                                 temperature = parsed.temp.value("C") if parsed.temp is not None else None
#                                 dewpt = parsed.dewpt.value("C") if parsed.dewpt is not None else None
#
#                             elif windSelector.match(term) or windSelectorVRB.match(term):
#                                 parsed=Metar.Metar(term)
#                                 windComponent=term
#                                 windDirection = parsed.wind_dir.value() if parsed.wind_dir is not None else None
#                                 windSpeed = parsed.wind_speed.value("KT")*0.514444 if parsed.wind_speed is not None else None
#                                 windGustSpeed = parsed.wind_gust.value("KT")*0.514444 if parsed.wind_gust is not None else None
#
#                             elif windDirectionVariabilitySelector.match(term) and windComponent is not None:
#                                 parsed=Metar.Metar(windComponent+" "+term)
#                                 windDirectionFrom = parsed.wind_dir_from.value() if parsed.wind_dir_from is not None else None
#                                 windDirectionTo = parsed.wind_dir_to.value() if parsed.wind_dir_to is not None else None
#
#                             elif visibilitySelector.match(term):
#                                 visibilityMetres=float(term)
#
#                             elif term=='CAVOK':
#                                 visibilityMetres=10000
#
#                             # else:
#                                 # print('Unrecognised term {}'.format(term))
#
#                 row=[formattedDT, 'WeatherObserved', stationCode,
#                                  stationName, barometricPressure, temperature, dewpt,
#                                  visibilityMetres, windDirection, windDirectionFrom, windDirectionTo,
#                                  windSpeed, windGustSpeed, latitude, longitude, ele]
#                 writer.writerow(row)
#                 rows.append(row)
#
#             except:
#                 print('Exception decoding {}'.format(metar))
#
#             i +=1
#     return rows
#
# # The main function
# def main():
#     start='2012-01-01'
#     end='2012-01-31'
#     # start='2017-01-01'
#     # end='2017-09-30'
#
#     amalgamated=[]
#     for item in weatherStationDict:
#         alias=item[0]
#         icao=item[1]
#         lat=item[2]
#         lng=item[3]
#         ele=item[4]
#
#         weatherData = readMetarHistory(alias, icao, start, end)
#         if weatherData is not None and len(weatherData)>0:
#             # weatherData=readFromRawText(icao, start, end)
#             writeToRawText(weatherData, icao, start, end)
#             rows=writeToCsv(weatherData, alias, icao, start, end, lat, lng, ele)
#             amalgamated.extend(rows)
#
#     df=pd.DataFrame(amalgamated, columns=['dateObserved', 'type', 'stationCode', 'siteName',
#                          'barometricPressure',
#                          'temperature', 'dewPointTemperature', 'visibilityMetres',
#                          'windDirection', 'windDirectionFrom', 'windDirectionTo', 'windSpeed',
#                          'windGustSpeed', 'latitude', 'longitude', 'elevation'])
#
#     filename='metar/metarData-amalgamated-'+start+'-'+end+'.csv'
#     df.to_csv(filename)
#
#
# main()
