
import requests
import uuid

authbody={ "auth": {
  "identity": {
    "methods": ["password"],
    "password": {
      "user": {
        "name": "test_admin",
        "domain": { "name": "testdata" },
        "password": "953dcb2c3edc"
      }
    }
  }
} }

headers = {'Content-Type': 'application/json',
             'Accept': 'application/json'}

r = requests.post("http://113.196.74.71:5001/v3/auth/tokens", json=authbody, headers=headers)

if r.status_code==201:
    token=r.headers['X-Subject-Token']

    headers={'x-auth-token': token,
             'Fiware-Service' : 'testdata',
             'Fiware-ServicePath' : '/LondonAirQuality',
             'Accept': 'application/json'}

    r2=requests.get("http://113.196.74.71:1027/v2/entities", headers=headers)

    if r2.status_code==200:
        print('Number of already stored entities = {}'.format(len(r2.json())))

    #add a new entity

    id=uuid.uuid4()

    testdata = {
        "id": str(id),
        "type": "AirQualityObserved",
        "CO": {
          "type": "Number",
          "value": 1,
          "metadata": {}
        },
        "NO": {
          "type": "Number",
          "value": 190,
          "metadata": {}
        },
        "NO2": {
          "type": "Number",
          "value": 68,
          "metadata": {}
        },
        "NOx": {
          "type": "Number",
          "value": 360,
          "metadata": {}
        },
        "SO2": {
          "type": "Number",
          "value": 8,
          "metadata": {}
        },
        "dateObserved": {
          "type": "DateTime",
          "value": "2016-11-11T23:00:00.00Z",
          "metadata": {}
        },
        "location": {
          "type": "geo:json",
          "value": {
            "type": "Point",
            "coordinates": [
              0.0,
              0.0
            ]
          },
          "metadata": {}
        },
        "source": {
          "type": "URL",
          "value": "http://example.com",
          "metadata": {}
        },
      }

    r3=requests.post("http://113.196.74.71:1027/v2/entities", headers=headers, json=testdata)

    if r3.status_code==201:
        print('New entity created')

    r4=requests.get("http://113.196.74.71:1027/v2/entities", headers=headers)

    if r4.status_code==200:
        print('Number of retrieved entities = {}'.format(len(r4.json())))
        for entity in r4.json():
            print('Entity id={} type={}'.format(entity['id'], entity['type']))


