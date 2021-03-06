
def toGeo(deg, min, sec=0):
    return float(deg)+(float(min)/float(60))+(float(sec)/float(3600))

weatherStationDict= [
    #Name, code, lat, long, alt
    ['Aberdeen', 'EGPD', toGeo(57,12), -toGeo(2,13), 65],
    ['Alderney', 'EGJA', toGeo(49,43), -toGeo(2,12), 71],
    ['RAF Barkston Heath', 'EGYE', toGeo(52,58), -toGeo(0,34), 112],
    ['Belfast International', 'EGAA', toGeo(54,39), -toGeo(6,13), 81],
    ['George Best Belfast City', 'EGAC', toGeo(54,36), -toGeo(5,53), 5],
    ['Bembridge', 'EGHJ', toGeo(50,40,40.96), -toGeo(1,6,34.51), 17],
    ['Benbecula', 'EGPL', toGeo(57,28,53.58), -toGeo(7,21,45.62), 6],
    ['RAF Benson', 'EGUB', toGeo(51,37), -toGeo(1,5), 63],
    ['Birmingham', 'EGBB', toGeo(52,27), -toGeo(1,44), 99],
    ['Blackpool', 'EGNH', toGeo(53,46), -toGeo(3,2), 10],
    ['Boscombe Down', 'EGDM', toGeo(51,10), -toGeo(1,45), 124],
    ['RAF Boulmer', 'EGQM', toGeo(55,25,19), -toGeo(1,36,12), 24],
    ['Bournemouth', 'EGHH', toGeo(50,47), -toGeo(1,50), 11],
    ['Shoreham', 'EGKA', toGeo(50,50), -toGeo(0,17), 2],
    ['Bristol', 'EGGD', toGeo(51,23), -toGeo(2,43), 189],
    ['Bristol Filton', 'EGTG', toGeo(51,31), -toGeo(2,35), 69],
    ['RAF Brize Norton', 'EGVN', toGeo(51,45), -toGeo(1,35), 88],
    ['Caernarfon', 'EGCK', toGeo(53,6,15.22), -toGeo(4,20,13.48), 4],
    ['Cambridge', 'EGSC', toGeo(52,12), toGeo(0,11), 15],
    ['Campbeltown', 'EGEC', toGeo(55,26,14), -toGeo(5,41,11), 13],
    ['Cardiff', 'EGFF', toGeo(51,24), -toGeo(3,21), 67],
    ['Carlisle Lake District', 'EGNC', toGeo(54,56,15), -toGeo(2,48,33), 57],
    ['RAF Church Fenton', 'EGXG', toGeo(53,50,12.81), -toGeo(1,11,44.19), 9],
    ['RAF Colerne', 'EGUO', toGeo(51,26,33.55), -toGeo(2,16,48), 181],
    ['RAF Coningsby', 'EGXC', toGeo(53,5), -toGeo(0,10), 7],
    ['Cork', 'EICK', toGeo(51,51), -toGeo(8,29), 153],
    ['RAF Cosford', 'EGWC', toGeo(52,38,25.95), -toGeo(2,18,18.99), 83],
    ['RAF Cottesmore', 'EGXJ', toGeo(52,44,8.81), -toGeo(0,38,55.39), 141],
    ['Coventry', 'EGBE', toGeo(52,22), -toGeo(1,29), 82],
    ['Cranfield', 'EGTC', toGeo(52,4), -toGeo(0,37), 111],
    ['RAF Cranwell', 'EGYD', toGeo(53,2), -toGeo(0,30), 67],
    ['RNAS Culdrose', 'EGDR', toGeo(50,5), -toGeo(5,15), 78],
    ['City of Derry', 'EGAE', toGeo(55,3), -toGeo(7,9), 9],
    ['RAF Dishforth', 'EGXD', toGeo(54,8,12.39), -toGeo(1,25,3.09), 36],
    ['Doncaster/Sheffield', 'EGCN', toGeo(53,29), -toGeo(1,0), 17],
    ['Donegal', 'EIDL', toGeo(55,2,39), -toGeo(8,20,28), 9],
    ['Dublin', 'EIDW', toGeo(53,26), -toGeo(6,15), 68],
    ['Dublin (Casement)', 'EIME', toGeo(53,18,6), -toGeo(6,27,4), 97],
    ['Dundee', 'EGPN', toGeo(56,27), -toGeo(3,1), 4],
    ['Durham Tees Valley', 'EGNV', toGeo(54,31), -toGeo(1,25), 37],
    ['Edinburgh', 'EGPH', toGeo(55,57), -toGeo(3,21), 41],
    ['Exeter', 'EGTE', toGeo(50,44), -toGeo(3,25), 30],
    ['RAF Fairford', 'EGVA', toGeo(51,41), -toGeo(1,47), 87],
    ['Farnborough', 'EGLF', toGeo(51,17), -toGeo(0,46), 65],
    ['Glasgow', 'EGPF', toGeo(55,52), -toGeo(4,26), 8],
    ['Glasgow Prestwick', 'EGPK', toGeo(55,30), -toGeo(4,35), 20],
    ['Gloucestershire', 'EGBJ', toGeo(51,54), -toGeo(2,10), 29],
    ['Great Yarmouth', 'EGSD', toGeo(52,38,7.66), toGeo(1,43,21.6), 0],
    ['Guernsey', 'EGJB', toGeo(49,26), -toGeo(2,36), 102],
    ['Hawarden', 'EGNR', toGeo(53,10), -toGeo(2,59), 10],
    ['RAF Holbeach', 'EGYH', toGeo(52,52), toGeo(0,9), 3],
    ['Humberside', 'EGNJ', toGeo(53,35), toGeo(0,21), 31],
    ['Inverness', 'EGPE', toGeo(57,32), -toGeo(4,3), 9],
    ['Islay', 'EGPI', toGeo(55,40,55), -toGeo(6,15,24), 16],
    ['Isle of Man', 'EGNS', toGeo(54,5), -toGeo(4,38), 17],
    ['Isles of Scilly', 'EGHE', toGeo(49,55), -toGeo(6,18), 31],
    ['Jersey', 'EGJJ', toGeo(49,13), -toGeo(2,12), 84],
    ['Kerry', 'EIKY', toGeo(52,10,51), -toGeo(9,31,26), 34],
    ['RAF Kinloss', 'EGQK', toGeo(57,39), -toGeo(3,34), 7],
    ['Kirkwall', 'EGPA', toGeo(58,57), -toGeo(2,54), 21],
    ['Knock', 'EIKN', toGeo(53,54,37), -toGeo(8,49,5), 203],
    ['RAF Lakenheath', 'EGUL', toGeo(52,25), toGeo(0,34), 10],
    ['Lands End', 'EGHC', toGeo(50,6,10.09), -toGeo(5,40,24.7), 122],
    ['Leconfield', 'EGXV', toGeo(53,52,37), -toGeo(0,26,15), 7],
    ['Leeds Bradford', 'EGNM', toGeo(53,52), -toGeo(1,39), 208],
    ['RAF Leeming', 'EGXE', toGeo(54,18), -toGeo(1,32), 40],
    ['RAF Leuchars', 'EGQL', toGeo(56,22,30.75), -toGeo(2,51,39.53), 12],
    ['RAF Linton', 'EGXU', toGeo(54,3), -toGeo(1,15), 16],
    ['Liverpool', 'EGGP', toGeo(53,20), -toGeo(2,51), 26],
    ["London City", "EGLC", toGeo(51,30), toGeo(0,3), 5 ], #latitude: 51-30N, longitude: 000-03E,
    ["London Biggin Hill", "EGKB", toGeo(51,19), toGeo(0,2), 183 ], #latitude: 51-19N, longitude: 000-02E,
    ["London Gatwick", "EGKK", toGeo(51,9), -toGeo(0,11), 62 ], #latitude: 51-09N, longitude: 000-11W,
    ["London Heathrow", "EGLL", toGeo(51,29), -toGeo(0,27), 24 ],#latitude: 51-29N, longitude: 000-27W,
    ["London Luton", "EGGW",toGeo(51,52), -toGeo(0,22), 160 ], #latitude: 51-52N, longitude: 000-22W,
    ["London Southend", "EGMC",toGeo(51,34), toGeo(0,42), 15 ], #latitude: 51-34N, longitude: 000-42E,
    ["London Stansted", "EGSS",toGeo(51,53), toGeo(0,14), 106 ], #latitude: 51-53N, longitude: 000-14E,
    ['RAF Lossiemouth', 'EGQS', toGeo(57,43), -toGeo(3,19), 13],
    ['Lydd', 'EGMD', toGeo(50,57), toGeo(0,56), 3],
    ['RAF Lyneham', 'EGDL', toGeo(51,30,1.28), -toGeo(1,59,45.78), 156],
    ['Manchester', 'EGCC', toGeo(53,21), -toGeo(2,17), 69],
    ['Manston', 'EGMH', toGeo(51,20,31.7), toGeo(1,20,46.3), 54],
    ['RAF Marham', 'EGYM', toGeo(52,39), toGeo(0,34), 23],
    ['Middle Wallop', 'EGVP', toGeo(51,9), -toGeo(1,34), 91],
    ['RAF Mildenhall', 'EGUN', toGeo(52,22), toGeo(0,29), 10],
    ['Newcastle', 'EGNT', toGeo(55,2), -toGeo(1,42), 81],
    ['Newquay', 'EGHQ', toGeo(50,26), -toGeo(5,0), 119],
    ["RAF Northolt", "EGWU", toGeo(51,33), -toGeo(0,25), 38 ], #latitude: 51-33N, longitude: 000-25W,
    ['Norwich', 'EGSH', toGeo(52,38), toGeo(1,18), 14],
    ['East Midlands', 'EGNX', toGeo(52,50), -toGeo(1,20), 94 ],
    ['Oban', 'EGEO', toGeo(56,27,49), -toGeo(5,24), 7],
    ['RAF Odiham', 'EGVO', toGeo(51,14), -toGeo(0,57), 123],
    ['Oxford', 'EGTK', toGeo(51,50,13), -toGeo(1,19,12), 82],
    ['Pembrey', 'EGOP', toGeo(51,43), -toGeo(4,22), 3],
    ['Plymouth', 'EGHD', toGeo(50,25,22.12), -toGeo(4,6,21.2), 145],
    ['Preston', 'EGNO', toGeo(53,44,42), -toGeo(2,53,2), 17],
    ['RAF Scampton', 'EGXP', toGeo(53,18,27), -toGeo(0,33,3), 62],
    ['Scatsta', 'EGPM', toGeo(60,26), -toGeo(1,18), 22],
    ['Shannon', 'EINN', toGeo(52,42), -toGeo(8,55), 14],
    ['RAF Shawbury', 'EGOS', toGeo(52,48), -toGeo(2,40), 76],
    ['Sligo', 'EISG', toGeo(54,16,49), -toGeo(8,35,57), 3],
    ['Southampton', 'EGHI', toGeo(50,54), -toGeo(1,24), 9],
    ['RAF Spadeadam', 'EGOM', toGeo(55,3), -toGeo(2,33), 325],
    ['St Athan', 'EGDX', toGeo(51,24), -toGeo(3,26), 50],
    ['Stornoway', 'EGPO', toGeo(58,13), -toGeo(6,19), 9],
    ['Sumburgh', 'EGPB', toGeo(59,53), -toGeo(1,18), 5],
    ['RAF Tain', 'EGQA', toGeo(57,49), -toGeo(3,58), 4],
    ['Tiree', 'EGPU', toGeo(56,30), -toGeo(6,53), 12],
    ['RAF Topcliffe', 'EGXZ', toGeo(54,12), -toGeo(1,23), 28],
    ['RAF Valley', 'EGOV', toGeo(53,15), -toGeo(4,32), 1],
    ['RAF Waddington', 'EGXW', toGeo(53,10), -toGeo(0,31), 68],
    ['Warton', 'EGNO', toGeo(53,44,41.78), -toGeo(2,53,1.82), 17],
    ['Waterford', 'EIWF', toGeo(52,11,14), -toGeo(7,5,13), 36],
    ['RAF Wattisham', 'EGUW', toGeo(52,7), toGeo(0,58), 87],
    ['Wick', 'EGPC', toGeo(58,27), -toGeo(3,5), 39],
    ['RAF Wittering', 'EGXT', toGeo(52,37), -toGeo(0,28), 84],
    ['RAF Woodvale', 'EGOW', toGeo(53,34,54), -toGeo(3,3,20), 11],
    ['RAF Wyton', 'EGUY', toGeo(53,34,54), -toGeo(3,3,20), 11],
    ['RNAS Yeovilton', 'EGDY', toGeo(52,21,26.16), -toGeo(0,6,22.47), 41],
    ]

weatherStationLookup={}

for row in weatherStationDict:
    weatherStationLookup[row[1]]=row
