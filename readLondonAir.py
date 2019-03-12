# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:18:22 2017

@author: shauntomy
"""

import requests
import json
import uuid
import pprint
import time
import datetime
import csv
import math
import sys

import readUtils
import getopt

blacklist=['BQ1', 'BQ2', 'BQ3', 'BQ4', 'ME5', 'ME6']

# This function retrieves the stations with air quality sensors within a region (input). 
def retrieveAirQualityDataGroupInfo(groupName, debug):
    url = 'http://api.erg.kcl.ac.uk/AirQuality/Information/MonitoringSites/GroupName=' + groupName +'/json'
    dicts=readUtils.fetchJsonWithRetry(url, retryDelay=60, debug=debug)
    return dicts
        
# This function uses retrieveAirQualityDataGroupInfo() function to store the station codes within a specific region that are
# used to monitor the air quality.
def retrieveStationCodes(groupName, debug):
    stationCodes = []    
    resp = retrieveAirQualityDataGroupInfo(groupName, debug)
    status=resp[1]
    if status==200:
        dicts=resp[0]
        if dicts is not None:
            size = len(dicts['Sites']['Site'])

            i = 0
            while i < size:
                code=dicts['Sites']['Site'][i]['@SiteCode']
                if code not in blacklist:
                    stationCodes.append(code)
                i += 1

            stationCodes.sort()
    
    return stationCodes

# This function will retrieve all the air quality data for all the stations within a specified region and in specified timeframe (input)
def retrieveAllAirQualityData(zone, start_date, end_date, debug):
    stationCodes = retrieveStationCodes(zone, debug)
    listForHarmonisation = []
    for item in stationCodes:
        listForHarmonisation.append(retrieveAirQualityDataWithStationCode(item, start_date, end_date, debug))
    
    return listForHarmonisation

# This function will return the air quality data for a particular station (input) in aspecific time frame (input). 
# Extra delays have been added to the requests to avoid making bulk requests in a short time period.
def retrieveAirQualityDataWithStationCode(stationCode, start_date, end_date, debug):
    consolidatedData = {"AirQualityData": {
        "SiteCode": stationCode,
        "Data":[]}}

    print('Processing {0}'.format(stationCode))

    numDays=15
    
    customDays = datetime.timedelta(days=numDays)
    
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    second_date = start_date + customDays

    dateDifference = int((end_date - start_date).days)

    i = 0
    # printProgressBar(i, dateDifference, prefix = 'Progress:', suffix = 'Complete', length = 50)
    numRecords = 0


    while start_date <= end_date:
        if second_date > end_date:
            second_date = end_date

        start_date_string = start_date.strftime('%Y-%m-%d')
        if start_date==second_date:
            adjusted=start_date-datetime.timedelta(days=1)
            start_date_string = adjusted.strftime('%Y-%m-%d')

        second_date_string = second_date.strftime('%Y-%m-%d')

        url = 'http://api.erg.kcl.ac.uk/AirQuality/Data/Site/SiteCode={0}/StartDate={1}/EndDate={2}/Json'.format(stationCode, start_date_string, second_date_string)
        resp=readUtils.fetchJsonWithRetry(url, debug=debug)
        if resp is not None and resp[1]==200:
            snapshotData=resp[0]
            if snapshotData is not None and any(snapshotData):
                for item in snapshotData['AirQualityData']['Data']:
                    measurementDate=item['@MeasurementDateGMT']
                    parsedMeasurementDate=datetime.datetime.strptime(measurementDate.split(None, 1)[0], '%Y-%m-%d')
                    if parsedMeasurementDate>=start_date and parsedMeasurementDate<=end_date:
                        consolidatedData['AirQualityData']['Data'].append(item)
                        numRecords+=1

        time.sleep(5)

        start_date = start_date + customDays
        second_date = second_date + customDays
        
        if (dateDifference - i) < numDays:
            i = i + (dateDifference - i)
        else:
            i += numDays

    print('Read '+str(numRecords)+' records for site '+stationCode)

    return consolidatedData
    

# This function takes in a single data entity from London Air website and converts the data into harmonised entiy
# according the Data Model document
def consolidateOriginalData(originalData, zone, debug):
    print('Consolidating records for '+zone)
    resp = retrieveAirQualityDataGroupInfo(zone, debug)

    if resp[1]==200:
        zoneData=resp[0]

        siteMeta={}
        for site in zoneData['Sites']['Site']:
            siteCode=site['@SiteCode']
            siteMeta[siteCode]=site

        consolidated={}
        for record in originalData:
            siteCode=record['AirQualityData']['SiteCode']
            dataRecords=record['AirQualityData']['Data']
            #Create storage by date
            if siteCode not in consolidated:
                meta=siteMeta[siteCode]
                consolidated[siteCode]={'Meta':{'LocalAuthorityName':meta['@LocalAuthorityName'],
                                         'SiteCode':siteCode,
                                         'SiteName':meta['@SiteName'],
                                         'SiteType':meta['@SiteType'],
                                         'Latitude':readUtils.safeParseFloat(meta['@Latitude']),
                                         'Longitude':readUtils.safeParseFloat(meta['@Longitude'])},
                                        'Measurements':{}}
            for data in dataRecords:
                measurementDate=data['@MeasurementDateGMT']
                if measurementDate not in consolidated[siteCode]['Measurements']:
                    consolidated[siteCode]['Measurements'][measurementDate]={}
                species=data['@SpeciesCode']
                value=data['@Value']
                consolidated[siteCode]['Measurements'][measurementDate][species]=readUtils.safeParseFloat(value)

    return consolidated

# This function writes the data to a csv file
def writeToCsv(airQualData, outputfile, debug):
    if debug:
        print('Writing '+outputfile)
     
    with open(outputfile, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['MeasurementDate', 'SiteCode', 'SiteName', 'SiteType', 'LocalAuthorityName', 'Latitude', 'Longitude', 'NO2', 'PM10', 'PM2.5', 'CO', 'O3', 'SO2'])

        for siteCode, siteData in airQualData.items():
            siteMeta=siteData['Meta']
            siteMeasurements=siteData['Measurements']

            localAuthorityName=readUtils.extractValueFromDict(siteMeta, 'LocalAuthorityName')
            siteCode=readUtils.extractValueFromDict(siteMeta, 'SiteCode')
            siteName=readUtils.extractValueFromDict(siteMeta, 'SiteName')
            siteType=readUtils.extractValueFromDict(siteMeta, 'SiteType')
            latitude=readUtils.extractValueFromDict(siteMeta, 'Latitude')
            longitide=readUtils.extractValueFromDict(siteMeta, 'Longitude')

            for measurementDate, measurementValues in siteMeasurements.items():
                NO2 = readUtils.extractValueFromDict(measurementValues, "NO2")
                PM10 = readUtils.extractValueFromDict(measurementValues, "PM10")
                PM25 = readUtils.extractValueFromDict(measurementValues, "PM25")
                CO = readUtils.extractValueFromDict(measurementValues, "CO")
                O3 = readUtils.extractValueFromDict(measurementValues, "O3")
                SO2 = readUtils.extractValueFromDict(measurementValues, "SO2")

                writer.writerow([measurementDate, siteCode, siteName, siteType, localAuthorityName, latitude, longitide, NO2, PM10, PM25, CO, O3, SO2])

def saveToOrion(consolidatedData, host, port, schema, debug, baseUrl, username, password):
    pass
    #TODO

def saveToPostgres(consolidatedData, host, port, schema, debug, username, password, table):
    pass
    #TODO

def main(argv):
    optlist, args = getopt.getopt(argv, '', ['start=', 'end=', 'zone=', 'host=', 'port=', 'mode=', 'output-file=', 'schema=', 'debug=', 'username=', 'password=', 'table=', 'baseurl=',])

    now=datetime.datetime.now()
    yesterday=now-datetime.timedelta(days=1)
    yearago=yesterday-datetime.timedelta(days=365)

    format='%Y-%m-%d'

    host='localhost'
    port=1026
    schema='orion'
    start=yearago.strftime(format)
    end=yesterday.strftime(format)
    zone='greenwich' #London
    mode=1 #default 1=csv, 2=fiware, 3=postgres
    filename=None
    modeName=None
    debug=False
    username=None
    password=None
    baseUrl=None
    table='AirQuality'


    for o, a in optlist:
        # print('Option '+o+' = '+a)
        if o=='--start':
            start=a
        elif o=='--end':
            end=a
        elif o=='--zone':
            zone=a
        elif o=='--username':
            username=a
        elif o=='--password':
            password=a
        elif o=='--table':
            table=a
        elif o=='--baseurl':
            baseUrl=a
        elif o=='--host':
            host=a
            mode=2
        elif o=='--port':
            port=a
            mode=2
        elif o=='--output-file':
            filename=a
            mode=1
        elif o=='--schema':
            schema=a
            mode=2
        elif o=='--mode':
            modeName=a
        elif o=='--debug':
            debug=readUtils.parseBoolean(a)

    if modeName=='csv':
        mode=1
    elif modeName=='orion':
        mode=2
    elif modeName=='postgresql':
        mode=3

    if filename is None:
        filename='londonAirQualityData-'+zone+'-'+start+'-'+end+'.csv'

    if debug==True:
        print('Startup Options:')
        print('    modeName = '+(modeName if modeName is not None else 'n/a'))
        print('    mode     = '+str(mode))
        print('    start    = '+start)
        print('    end      = '+end)
        print('    zone     = '+zone)

        if mode==1:
            print('    filename = '+filename)
        elif mode==2:
            print('    host     = '+host)
            print('    port     = '+str(port))
            print('    schema   = '+schema)
            print('    username = '+username)
            print('    baseurl  = '+baseUrl)
        elif mode==3:
            print('    host     = '+host)
            print('    port     = '+str(port))
            print('    schema   = '+schema)
            print('    username = '+username)
            print('    table    = '+table)

    originalData = retrieveAllAirQualityData(zone, start, end, debug)
    consolidatedData = consolidateOriginalData(originalData, zone, debug)
    if mode==1:
        writeToCsv(consolidatedData, filename, debug)
    elif mode==2:
        saveToOrion(consolidatedData, host, port, schema, debug, baseUrl, username, password)
    elif mode==3:
        saveToPostgres(consolidatedData, host, port, schema, debug, username, password, table)


if __name__ == "__main__":
    # args=['--start=2017-07-01', '--end=2017-07-31', '--zone=London']
    # main(args)
    main(sys.argv[1:])
