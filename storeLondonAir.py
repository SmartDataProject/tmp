
import uuid
import requests
import getopt
from pandas import DataFrame
import sys
import math
import time
import readUtils

batchLimit=500

def formatAsNGSIDate(dtStr):
    r=None
    if dtStr is not None and len(dtStr)==19:
        r=dtStr[0:10]+'T'+dtStr[11:19]+'Z'
    return r

def formatAsStoredNGSIDate(dtStr):
    r=None
    if dtStr is not None and len(dtStr)==19:
        r=dtStr[0:10]+'T'+dtStr[11:19]+'.00Z'
    return r

def clean(str):
    r=str
    if str is not None:
        r=str.replace('(','[').replace(')',']')
    return r

def qv(value, units):
    r={}
    if value is not None and isinstance(value, float) or isinstance(value, int):
        r={
            "type": "ExtQuantitativeValue",
            "value" : {
                "value": value,
                "units": units
            }
        }
    return r

def authorise(authaddr, domain, user, password):
    print('Sending authorization request for user='+user+' ,domain='+domain)
    token=None
    authbody={ "auth": {
      "identity": {
        "methods": ["password"],
        "password": {
          "user": {
            "name": user,
            "domain": { "name": domain},
            "password": password
          }
        }
      }
    } }

    resp, status_code, headers = readUtils.postAndfetchJsonWithRetry(authaddr, jsonData=authbody)

    if status_code==201:
        token=headers['X-Subject-Token']
        print('Authorized with token '+token)

    return token

def listExistingEntities(fiwareAuthBaseURL, fiwareUser, fiwarePassword, srvBaseAddr, service, servicePath, stationCodes, startDT, endDT):
    cache={}

    pstart=time.time()
    for id in stationCodes:
        lastDo=None
        finished=False
        cache[id]={}
        start=startDT[id]
        end=endDT[id]

        token=None
        if fiwareAuthBaseURL is not None:
            token=authorise(fiwareAuthBaseURL, service, fiwareUser, fiwarePassword)

        while not finished:

            headers={
                 'Fiware-Service' : service,
                 'Fiware-ServicePath' : servicePath,
                 'Accept': 'application/json'}
            if token is not None:
                headers['x-auth-token']=token

            url=srvBaseAddr+"entities?type=AirQualityObserved&q=stationCode=="+id
            if lastDo is not None:
                url=url+";dateObserved>"+lastDo
            elif startDT is not None:
                url=url+";dateObserved>="+start

            if endDT is not None:
                url=url+";dateObserved<="+end

            url=url+"&orderBy=dateObserved&attrs=dateObserved&limit="+str(batchLimit)

            print('Requesting '+url)
            start_time = time.time()
            jsonData, status_code, responseHeaders=readUtils.fetchJsonWithRetry(url, extraheaders=headers)

            print('List completion status = {} ({}s)'.format(status_code, (time.time() - start_time)))

            if status_code==200:
                if not any(jsonData) or (isinstance(jsonData, list) and len(jsonData)==0):
                    finished=True
                else:
                    for row in jsonData:
                        observed=row['dateObserved']
                        if observed is not None and 'value' in observed:
                            dateObserved=observed['value']
                            cache[id][dateObserved]=True
                            lastDo=dateObserved
                    if len(jsonData)<batchLimit:
                        finished=True


    print('Took {} to fetch cache'.format(time.time()-pstart))

    return cache

def postEntity(srvBaseAddr, token, service, servicePath, data):
    print('Posting entity')
    headers={'Fiware-Service' : service,
         'Fiware-ServicePath' : servicePath,
         'Accept': 'application/json'}
    if token is not None:
        headers['x-auth-token']=token

    url=srvBaseAddr+"entities"
    r2, status_code, headers=readUtils.postAndfetchJsonWithRetry(url, jsonData=data, extraheaders=headers)

    print('Create completion status = {} message = {}'.format(status_code, r2))
    return status_code


def constructNewRecord(row):

    lat=row['Latitude']
    lng=row['Longitude']

    measured=row['MeasurementDate']

    r = {'id': str(uuid.uuid4()),
        'stationCode': {
            'value': row['SiteCode'],
            'type': 'Text'
        },
        'stationName': {
            'value': clean(row['SiteName']),
            'type': 'Text'
        },
        'localAuthorityName': {
            'value': clean(row['LocalAuthorityName']),
            'type': 'Text'
        },
        'siteType': {
            'value': clean(row['SiteType']),
            'type': 'Text'
        },
         'type': 'AirQualityObserved',
         "dateObserved": {
             "value": formatAsNGSIDate(row['MeasurementDate']),
             "type": "DateTime"
         },
         "source": {
             "value": "http://www.londonair.org.uk",
             "type": "URL"
         },
         "dataProvider": {
             "value": "GSM Association",
             "type": "Text"
         },
         "schemaVersion": {
             "value": "1.0",
             "type": "Text"
         },
         "location": {
             "value": {
                 "type": "Point",
                 "coordinates": [
                     lng,
                     lat
                 ]
             },
             "type": "geo:json"
         },
    }

    NO2=row['NO2']
    PM10=row['PM10']
    PM25=row['PM2.5']
    CO=row['CO']
    O3=row['O3']
    SO2=row['SO2']

    if NO2 is not None and (isinstance(NO2, int) or isinstance(NO2, float)) and not math.isnan(NO2):
        r['NO2']=qv(NO2, 'GQ')
    if PM10 is not None and (isinstance(PM10, int) or isinstance(PM10, float)) and not math.isnan(PM10):
        r['PM10']=qv(PM10, 'GQ')
    if PM25 is not None and (isinstance(PM25, int) or isinstance(PM25, float)) and not math.isnan(PM25):
        r['PM2.5']=qv(PM25, 'GQ')
    if CO is not None and (isinstance(CO, int) or isinstance(CO, float)) and not math.isnan(CO):
        r['CO']=qv(CO, 'GQ')
    if O3 is not None and (isinstance(O3, int) or isinstance(O3, float)) and not math.isnan(O3):
        r['O3']=qv(O3, 'GQ')
    if SO2 is not None and (isinstance(SO2, int) or isinstance(SO2, float)) and not math.isnan(SO2):
        r['SO2']=qv(SO2, 'GQ')
    return r

def storeRecord(fiwareBaseURL, fiwareService, fiwareServicePath, token, row, index, cache):

    observationDate=row['MeasurementDate']
    stationCode=row['SiteCode']

    print('Processing site {} date {} index {}'.format(stationCode, observationDate, index))

    exists=False
    if stationCode in cache:
        if formatAsStoredNGSIDate(observationDate) in cache[stationCode]:
            exists=True

    if not exists:
        data=constructNewRecord(row)
        storeStatus=postEntity(fiwareBaseURL, token, fiwareService, fiwareServicePath, data)
        if storeStatus==201:
            print('Stored entry')
        else:
            print('Error storing entry {}'.format(storeStatus))

def extractStationCodes(source):
    stationCodes=[]
    startDT={}
    endDT={}

    print('Checking for site codes & date ranges')

    for i in source.index:
        datarow=source.loc[i,:]
        site=datarow['SiteCode']
        dt=datarow['MeasurementDate']
        fdt=formatAsNGSIDate(dt)

        if site is not None:
            if site not in stationCodes:
                stationCodes.append(site)
                startDT[site]=fdt
                endDT[site]=fdt
            else:
                if fdt<startDT[site]:
                    startDT[site]=fdt
                if fdt>endDT[site]:
                    endDT[site]=fdt

    return stationCodes, startDT, endDT

def mainLoop(fiwareAuthBaseURL, fiwareBaseURL, fiwareService, fiwareServicePath,
             fiwareUser, fiwarePassword, source):

    useCount=0
    token=None

    stationCodes, startDT, endDT=extractStationCodes(source)

    cache=listExistingEntities(fiwareAuthBaseURL, fiwareUser, fiwarePassword, fiwareBaseURL, fiwareService, fiwareServicePath, stationCodes, startDT, endDT)

    for i in source.index:
        datarow=source.loc[i,:]

        if useCount==0:
            if fiwareAuthBaseURL is not None:
                token=authorise(fiwareAuthBaseURL, fiwareService, fiwareUser, fiwarePassword)
            useCount+=1
        else:
            useCount+=1
            if useCount>500:
                useCount=0

        try:
            storeRecord(fiwareBaseURL, fiwareService, fiwareServicePath, token, datarow, i, cache)
        except Exception as e:
            print('Trapped exception '+str(e))

def main(argv):
    optlist, args = getopt.getopt(argv, '', ['authBaseURL=', 'baseURL=', 'service=', 'servicePath=', 'user=', 'password='])

    sourceFile=None

    if args is not None and len(args)>=1:
        sourceFile=args[0]

    fiwareAuthBaseURL= None
    fiwareBaseURL= 'http://localhost:1026/v2/'
    fiwareService= 'sandboxdata'
    fiwareServicePath= '/LondonAirQuality'
    fiwareUser= None
    fiwarePassword= None

    for o, a in optlist:
        if o=='--authBaseURL':
            fiwareAuthBaseURL=a
        elif o=='--baseURL':
            fiwareBaseURL=a
        elif o=='--service':
            fiwareService=a
        elif o=='--servicePath':
            fiwareServicePath=a
        elif o=='--user':
            fiwareUser=a
        elif o=='--password':
            fiwarePassword=a

    if sourceFile is not None:
        source=DataFrame.from_csv(sourceFile, index_col=None)
        filtered=source

        # filtered=source[source['SiteCode']=='GN2']

        if source is not None:
            mainLoop(fiwareAuthBaseURL, fiwareBaseURL, fiwareService, fiwareServicePath,
                     fiwareUser, fiwarePassword, filtered)

if __name__ == "__main__":
    main(sys.argv[1:])
    # main(['londonAirQualityData-greenwich-2013-01-01-2013-01-16.csv'])
