
import requests
import time
import math

def fetchJsonWithRetry(url, nRetries=10, retryDelay=15, debug=False, extraheaders=None):
    retries=nRetries
    respData=None
    finished=False
    status_code=None
    responseHeaders=None

    headers = {'Accept': 'application/json'}
    if extraheaders is not None and isinstance(extraheaders, dict) and len(extraheaders)>0:
        for k in extraheaders.keys():
            headers[k]=extraheaders[k]

    while not finished:
        if debug:
            print('Requesting '+url)
        try:
            r = requests.get(url, headers=headers)

            respData = r.json()
            status_code=r.status_code
            responseHeaders=r.headers
            finished=True

        except ValueError as e:
            pass
        except ConnectionResetError as e:
            pass
        except ConnectionError as e:
            pass
        except Exception as e:
            if debug:
                print('Other Exception: ',e)
            pass
        if not finished:
            time.sleep(retryDelay)
            print('Retrying '+url)
            retries-=1
            if retries<0:
                finished=True
                if debug:
                    print('Failed to retrieve '+url)
    return respData, status_code, responseHeaders

def fetchWithRetry(url, nRetries=10, retryDelay=15, debug=False, extraheaders=None):
    retries=nRetries
    respData=None
    finished=False
    status_code=None
    responseHeaders=None

    headers = {}
    if extraheaders is not None and isinstance(extraheaders, dict) and len(extraheaders)>0:
        for k in extraheaders.keys():
            headers[k]=extraheaders[k]

    while not finished:
        if debug:
            print('Requesting '+url)
        try:
            r = requests.get(url, headers=headers)

            respData = r.content
            if respData is not None:
                respData=respData.decode('UTF-8')
            status_code=r.status_code
            responseHeaders=r.headers
            finished=True

        except ValueError as e:
            pass
        except ConnectionResetError as e:
            pass
        except ConnectionError as e:
            pass
        except Exception as e:
            if debug:
                print('Other Exception: ',e)
            pass
        if not finished:
            time.sleep(retryDelay)
            print('Retrying '+url)
            retries-=1
            if retries<0:
                finished=True
                if debug:
                    print('Failed to retrieve '+url)
    return respData, status_code, responseHeaders

def postAndfetchJsonWithRetry(url, jsonData, nRetries=10, retryDelay=15, debug=False, extraheaders=None):
    retries=nRetries
    respData=None
    finished=False
    status_code=None
    responseHeaders=None

    headers = {'Content-Type': 'application/json',
             'Accept': 'application/json'}

    if extraheaders is not None and isinstance(extraheaders, dict) and len(extraheaders)>0:
        for k in extraheaders.keys():
            headers[k]=extraheaders[k]

    while not finished:
        if debug:
            print('Requesting '+url)
        try:
            r = requests.post(url, json=jsonData, headers=headers)

            respData = r.content
            status_code=r.status_code
            responseHeaders=r.headers
            finished=True
            print('Stored with response {}'.format(status_code))

        except ValueError as e:
            pass
        except ConnectionResetError as e:
            pass
        except ConnectionError as e:
            pass
        except Exception as e:
            if debug:
                print('Other Exception: ',e)
            pass
        if not finished:
            time.sleep(retryDelay)
            print('Retrying '+url)
            retries-=1
            if retries<0:
                finished=True
                if debug:
                    print('Failed to retrieve '+url)
    return respData, status_code, responseHeaders

def safeParseFloat(source):
    v=None
    if source is not None:
        if isinstance(source, str) and len(source.strip())>0:
            try:
                v=float(source)
            except Exception:
                print("Exception parsing ", source)
                pass
        elif isinstance(source,float) and not math.isnan(source):
            v=source
    return v

def extractValueFromDict(dd, field):
    v=None
    if field in dd:
        v=dd[field]
    return v

def parseBoolean(source):
    v=False
    if isinstance(source, bool):
        v=source
    elif isinstance(source, str):
        l=source.strip().lower()
        v=l in ['true', 't', 'y', '1']
    elif isinstance(source, int):
        v=source!=0
    return v
