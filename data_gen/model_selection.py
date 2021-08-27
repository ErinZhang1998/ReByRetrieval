import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--scene_num_to_shapenet_info", dest="scene_num_to_shapenet_info")
# '/raid/xiaoyuz1/preselect_august/selected_object.pkl'
parser.add_argument("--selected_object_file", dest="selected_object_file")


all_mugs = [{
    'name' : 'mugs',
    'canonical' : 794,
    'similar' : [
        792,796,797,798,801,805,812,817,819,820, \
        822,824,826,830,832,833,834,836,844,845, \
        849,853,855,856,861,863,869,870,872,878, \
        879,881,889,892,906,913,921,928,930,938, \
        940,942,943,955,956,957,958,960,961,962, \
        966,970,975,982,986,992,1000,
    ],
    'not_sure' : [
        800,810,821,827,829,839,840,841,842,843,848, \
        850,854,857,860,866,871,873,875,884,886,887, \
        888,899,900,902,908,909,910,912,916,919,920, \
        924,925,933,935,944,948,950,952,963,969,972, \
        978,983,990,1002,1005,
    ]
},
{
    'name': 'mugs_square_handle',
    'canonical' : 802,
    'similar' : [
        804,811,825,847,852,865,867,894,895,896, \
        897,901,915,917,918,922,929,941,985,987,995, \
        998,999,1001,1003,1004,
    ],
    'not_sure' : [
        851,859,866,876,882,884,885,891,908,924,936, \
        951,953,973,974,976,977,984,989,994,1005,
    ]
},
{
    'name' : 'mug_tappered',
    'canonical' : 807,
    'similar' : [
        813,815,823,831,874,904,905,954,959,968,988,
    ],
    'not_sure' : [
        800,821,827,829,835,838,839,841,842,862,864,868, \
        883,887,924,926,931,932,936,937,939,949,1002,
    ]
},
# {
#     'name' : 'not_very_cylinder_body',
#     'canonical' : None,
#     'similar' : [
#         795,803,880,893,907,991
#     ],
#     'not_sure' : [ 
#     ]
# }
]

#######################################################
all_cameras = [
{
    'name' : 'camera_box',
    'canonical' : 1021,
    'similar' : [
        1011,1016,1020,1021,1029,1051,1086,1088, \
        1096,1105,
    ],
    'not_sure' : [
        1045,1052,1054,1055,1064,1069,1072,1081, \
        1082,1087
    ]
},
{
    'name' : 'camera_box_protrude',
    'canonical' : 1017,
    'similar' : [
        1006,1009,1017,1024,1030,1034,1036,1041,1044, \
        1046,1047,1048,1049,1058,1059,1060,1075,1084, \
        1085,1102,1107,1113,1114,1117,
    ],
    'not_sure' : [
        1039,1045,1052,1054,1055,1056,1064,1069,1072, \
        1073,1076,1077,1078,1081,1082,1091,1094,1095, \
        1115,
    ]
}
]

####################################################### 
all_bags = [{
    'name' : 'triangular_bag',
    'canonical' : None,
    'similar' : [
        2830,2835,2836,2842,2845,2848,2849,2852,2862, \
        2864,2865,2871,2881,2884,2895,2900,2903,2904, \
        2907,
    ],
    'not_sure' : [
        2855,2867,2868,
    ]
},
{
    'name' : 'rectangular_bag',
    'canonical' : None,
    'similar' : [
        2826,2827,2829,2837,2840,2844,2847,2856,2860, \
        2873,2875,2879,2880,2882,2883,2886,2890, \
        2892,2896,2905,2908,2872,
    ],
    'not_sure' : [
        2831,2833,2834,2838,2853,2855,2867,2868,
    ]
}]
## laptop 
all_laptops = [
{
    'name' : 'straight_laptop',
    'canonical' : None,
    'similar' : [
        1119,1124,1126,1128,1131,1133, \
        1138,1143,1145,1146,1150,1151,1152, \
        1155,1156,1157,1158,1160,1161,1164, \
        1166,1167,1168,1169,1172,1173,1175, \
        1178,1179,1182,1184,1185,1187,1190, \
        1193,1198,1200,1201,1202,1204,1205, \
        1211,1212,1214,1215,1216,1217,1218, \
        1206,1220,1221,1222,1224,1225,1228, \
        1230,1231,1232,1239,1241,1243,1245, \
        1251,1252,1253,1255,1259,1262,1264, \
        1267,1269,1270,1271,1276,1277,1279, \
        1279,1277,1276,1577,1576,1574,1569, \
        1568,1552,1554,1555,1560,1559,1563, \
        1536,1538,1540,1544,1546,1550,1551, \
        1525,1526,1527,1528,1530,1531,
    ],
    'not_sure' : [],
},
{
    'name': 'slightly_slanted',
    'canonical' : None,
    'similar' : [
        1127,1129,1134,1136,1137,1139,1140,1142, \
        1147,1149,1153,1154,1159,1163,1174, \
        1176,1180,1181,1183,1186,1188,1191, \
        1192,1195,1197,1199,1210,1213,1223, \
        1234,1236,1237,1238,1240,1242,1244, \
        1249,1250,1254,1256,1257,1258,1260, \
        1261,1265,1266,1272,1273,1274,1275, \
        1278,1278,1275,1229,1572,1573,1571, \
        1561,1553,1557,1562,1537,1539,1542, \
        1545,1549,1548,1520,1524,1532,1533, \
        1535,
    ],
    'not_sure' : [],
},
{
    'name' : 'slanted_laptop',
    'canonical' : None,
    'similar' : [
        1121,1125,1132,1135,1144,1148,1162,1170, \
        1171,1194,1207,1208,1219,1227,1233,1263, \
        1268,1578,1556,1541,1543,1522,1523,
    ],
    'not_sure' : [
    ]
}
]

## jar
all_jars = [{
    'name' : 'jar',
    'canonical' : 1920,
    'similar' : [
       1927,1935,1940,1941,1999,2060,2085,2139,1911, \
       1879,1875,1887,1858,1857,1848,1800,1777,1731, \
       1728,1727,1671,1675,1624,1627,
    ],
    'not_sure' : [
        1936,1942,1944,1947,1948,1950,1959,1974, \
        1978,1982,1985,2015,2050,2080,2083,2084, \
        2098,2108,2167,
    ]
},
{
    'name' : 'jar_short_neck',
    'canonical' : 1968,
    'similar' : [
        1973,1975,1992,2003,2006,2007,2012,2028, \
        2029,2030,2042,2052,2065,2066,2069,2122, \
        2136,2137,2140,2141,2142,2150,2152,2153, \
        2170,1824,1832,1749,1729,1732,1703,1586, \
        
    ],
    'not_sure' : [
        2103,2107,2155,2161,2166,
    ]
}]
#########################################################################
all_cans = [
{
    'name' : 'can',
    'canonical' : None,
    'similar' : list(
        filter(lambda x: x not in [709,747], list(np.arange(685,792)))
    ),
    'not_sure' : [],
}
]
# all_bottles = [
# {
#     'name' : 'wine_bottle',
#     'canonical' : 34,
#     'similar' : [
#         1,4,9,12,14,15,19,31,34,35,36,38,40,48,50,53,55,61,62,72,76,79,83,84,85,86,88,89,90,91, \
#         94,96,97,100,102,106,107,109,111,116,119,120,131,132,135,137,139,143,145,150,161,162,176, \
#         178,179,182,196,201,203,207,212,215,219,220,221,222,226,231,232,233,237,238,239,243,249, \
#         252,254,256,257,260,268,274,275,280,288,297,301,302,304,305,306,307,310,313,318,321,325, \
#         326,337,338,341,353,358,361,362,364,366,368,369,374,379,380,385,390,399,412,416,420,421, \
#         422,427,432,438,439,441,447,453,457,458,460,465,468,470,472,473,475,487,489,491,493,496,
#     ],
#     'not_sure' : [157,112,110,258,263,265,191],
# }
# ]

all_bottles = [{'name': 'wine bottle',
  'canonical': 34,
  'similar': [1,
   4,
   9,
   12,
   14,
   15,
   19,
   31,
   34,
   35,
   36,
   38,
   40,
   48,
   50,
   53,
   55,
   61,
   62,
   72,
   76,
   79,
   83,
   84,
   85,
   86,
   88,
   89,
   90,
   91,
   94,
   96,
   97,
   100,
   102,
   106,
   107,
   109,
   111,
   116,
   119,
   120,
   131,
   132,
   135,
   137,
   139,
   143,
   145,
   150,
   161,
   162,
   176,
   178,
   179,
   182,
   196,
   201,
   203,
   207,
   212,
   215,
   219,
   220,
   221,
   222,
   226,
   231,
   232,
   233,
   237,
   238,
   239,
   243,
   249,
   252,
   254,
   256,
   257,
   260,
   268,
   274,
   275,
   280,
   288,
   297,
   301,
   302,
   304,
   305,
   306,
   307,
   310,
   313,
   318,
   321,
   325,
   326,
   337,
   338,
   341,
   353,
   358,
   361,
   362,
   364,
   366,
   368,
   369,
   374,
   379,
   380,
   385,
   390,
   399,
   412,
   416,
   420,
   421,
   422,
   427,
   432,
   438,
   439,
   441,
   447,
   453,
   457,
   458,
   460,
   465,
   468,
   470,
   472,
   473,
   475,
   487,
   489,
   491,
   493,
   496,
   242,
   244,
   282,
   285,
   292,
   295,
   309,
   316,
   324,
   343,
   345,
   346,
   54,
   60,
   63,
   169,
   193,
   402,
   415,
   433,
   445],
  'not_sure': [157,
   112,
   110,
   258,
   263,
   265,
   191,
   210,
   227,
   230,
   241,
   323,
   359,
   56,
   200,
   437,
   469,
   471,
   485,
   497]
},
{'name': 'beer bottle',
  'similar': [149, 153, 17, 25, 27, 28, 188, 277, 308],
  'not_sure': []
},
{
    'name': 'flask',
    'similar': [32, 124, 259, 20, 213, 224, 74, 174],
    'not_sure': []
},
{
    'name': 'water bottle',
    'similar': [37,
    41,
    44,
    46,
    148,
    154,
    158,
    159,
    142,
    123,
    127,
    98,
    99,
    104,
    264,
    266,
    267,
    269,
    271,
    22,
    24,
    26,
    30,
    177,
    189,
    108,
    211,
    214,
    216,
    225,
    228,
    234,
    245,
    246,
    251,
    291,
    298,
    311,
    312,
    314,
    315,
    334,
    335,
    6,
    11,
    65,
    68,
    71,
    80,
    87,
    95,
    165,
    167,
    170,
    173,
    192,
    194,
    198,
    199,
    202,
    205,
    206,
    208,
    363,
    365,
    370,
    372,
    373,
    375,
    381,
    386,
    387,
    388,
    389,
    392,
    393,
    395,
    409,
    413,
    425,
    431,
    440,
    442,
    443,
    448,
    454,
    461,
    480,
    490,
    494,
    495],
    'not_sure': [290, 467, 477]},
    {
        'name': 'water bottle2',
    'similar': [218,
    128,
    133,
    130,
    113,
    115,
    125,
    126,
    155,
    147,
    140,
    101,
    105,
    261,
    262,
    18,
    29,
    181,
    183,
    223,
    229,
    247,
    250,
    273,
    278,
    279,
    283,
    303,
    322,
    330,
    332,
    339,
    342,
    352,
    13,
    67,
    163,
    168,
    195,
    383,
    384,
    396,
    411,
    464,
    478,
    479
],
'not_sure': []},
{'name': 'cylinder',
'similar': [47,
16,
184,
287,
7,
10,
405,
406,
434,
449,
455,
462,
483,
492],
'not_sure': []},
{
    'name': 'jar_bottle',
'similar': [144, 180, 240, 253, 317, 57, 59, 424],
'not_sure': []
},
 {'name': 'round body bottle',
  'similar': [138, 121, 186, 190, 255, 329, 331, 474, 484],
  'not_sure': []},
 {'name': 'square bottle',
  'similar': [122,
   289,
   293,
   294,
   300,
   2,
   8,
   23,
   43,
   49,
   82,
   164,
   166,
   171,
   197,
   382,
   435,
   446,
   451,
   463],
  'not_sure': []},
#  {'name': 'weird',
#   'similar': [235,
#    236,
#    248,
#    276,
#    281,
#    284,
#    286,
#    296,
#    299,
#    319,
#    327,
#    328,
#    333,
#    336,
#    340,
#    344,
#    347,
#    348,
#    349,
#    350,
#    351,
#    354,
#    355,
#    356,
#    357,
#    0,
#    3,
#    5,
#    33,
#    39,
#    42,
#    45,
#    51,
#    52,
#    58,
#    64,
#    66,
#    69,
#    70,
#    73,
#    75,
#    77,
#    78,
#    81,
#    92,
#    93,
#    160,
#    172,
#    175,
#    204,
#    367,
#    371,
#    377,
#    391,
#    394,
#    397,
#    398,
#    400,
#    401,
#    403,
#    404,
#    407,
#    408,
#    410,
#    414,
#    417,
#    418,
#    419,
#    423,
#    426,
#    428,
#    429,
#    430,
#    436,
#    444,
#    450,
#    452,
#    456,
#    459,
#    466,
#    476,
#    481,
#    482,
#    486,
#    488],
#   'not_sure': [146,
#    151,
#    152,
#    156,
#    129,
#    134,
#    136,
#    141,
#    114,
#    117,
#    118,
#    103,
#    108,
#    270,
#    21,
#    185,
#    187,
#    209,
#    217,
#    218,
#    320]}
]
#####################################################################################3
all_bowls = [
{
    'name' : 'plate_like_bowls',
    'canonical' : 503,
    'similar' : [
        513,515,517,518,520,521,648,653,526,529,661,537,669,672,545,547,548, \
        678,561,563,565,575,576,577,578,580,599,600,607,608,613,616,619,623, \
        625,630,632,506,509,511,
    ],
    'not_sure' : [],
},
{
    'name' : 'deep_bowls',
    'canonical' : 605,
    'similar' : [
        512,516,519,522,523,524,525,527,531,532,533,534,535,536,538,544,546, \
        549,550,551,552,553,554,555,557,559,562,564,566,567,569,570,571,572, \
        574,579,582,583,588,589,590,591,592,593,595,597,598,601,603,604,606, \
        610,611,612,614,615,617,618,620,621,622,626,628,633,637,638,643,644, \
        646,649,651,654,656,658,659,660,662,663,664,665,667,668,670,671,673, \
        674,675,676,677,679,680,683,498,500,501,504,505,507,510,
    ],
    'not_sure' : [],
}
]

all_data_dict = {
    '03797390' : (all_mugs,'mug'),
    '02773838' : (all_bags,'bag,traveling bag,travelling bag,grip,suitcase'),
    '03642806' : (all_laptops,'laptop,laptop computer'),
    '03593526' : (all_jars,'jar'),
    '02946921' : (all_cans,'can,tin,tin can'),
    '02942699' : (all_cameras,'camera,photographic camera'),
    '02880940' : (all_bowls,'bowl'),
    '02876657' : (all_bottles,'bottle'),
}

def create_category_list(original_anno, scene_num_to_shapenet_info):
    data_list = []
    for D in original_anno:
        # canonical_list = [D['canonical']] if D['canonical'] is not None else []
        if not 'canonical' in D or D['canonical'] is None:
            canonical_list = []
        else:
            canonical_list = [D['canonical']]
        
        similar_list = D['similar'] + canonical_list
        similar_list = list(set(similar_list))
        
        similar_list = list(set(similar_list + D['not_sure']))

        same_object_category_list = []
        for scene_num in similar_list:
            try:
                model_id = scene_num_to_shapenet_info[scene_num]['model_id']
            except:
                print(scene_num_to_shapenet_info[scene_num])
                exit(0)
            same_object_category_list += [model_id]
        # import pdb; pdb.set_trace()
        data_list += [(D['name'], same_object_category_list)]
    return data_list

def get_all_data(scene_num_to_shapenet_info):
    
    train_data = []
    for k,v in all_data_dict.items():
        train_data += [(v[1], k, create_category_list(v[0], scene_num_to_shapenet_info))]
    # train_data += [('mug', '03797390', create_category_list(all_mugs))]
    # train_data += [('bag,traveling bag,travelling bag,grip,suitcase', '02773838', create_category_list(all_bags))]
    # train_data += [('laptop,laptop computer', '03642806', create_category_list(all_laptops))]
    # train_data += [('jar', '03593526', create_category_list(all_jars))]
    # train_data += [('can,tin,tin can', '02946921', create_category_list(all_cans))]
    # train_data += [('camera,photographic camera', '02942699', create_category_list(all_cameras))]
    # train_data += [('bowl', '02880940', create_category_list(all_bowls))]
    # train_data += [('bottle', '02876657', create_category_list(all_bottles))]

    return train_data

if __name__ == '__main__':
    args = parser.parse_args()
    scene_num_to_shapenet_info = pickle.load(open(args.scene_num_to_shapenet_info, 'rb'))
    all_data = get_all_data(scene_num_to_shapenet_info)
    all_data_fh = open(args.selected_object_file, 'wb+')
    pickle.dump(all_data, all_data_fh)