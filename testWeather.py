import requests
import uuid
import pprint

authbody={ "auth": {
  "identity": {
    "methods": ["password"],
    "password": {
      "user": {
        "name": "developer",
        "domain": { "name": "weatherdata" },
        "password": "80780f922wsx"
      }
    }
  }
} }

headers = {'Content-Type': 'application/json',
             'Accept': 'application/json'}

r = requests.post("http://113.196.74.71:5001/v3/auth/tokens", json=authbody, headers=headers)
print(r.status_code)

if r.status_code==201:
    token=r.headers['X-Subject-Token']

    headers={'x-auth-token': token,
             'Fiware-Service' : 'weatherdata',
             'Fiware-ServicePath' : '/Taiwan',
             'Accept': 'application/json'}
    print(headers)


    r4=requests.get("http://113.196.74.71:1027/v2/entities", headers=headers)
    print(r4.status_code)
    
    if r4.status_code==200:
        print('Number of retrieved entities = {}'.format(len(r4.json())))
        for point in r4.json():
            if point['type']=='WeatherObserved':
                pprint.pprint(point)

            #print('Entity id={} type={}'.format(entity['id'], entity['type']))





