import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--csv_file_dir", dest="csv_file_dir")

(args, argss) = parser.parse_args()

#bag
bag1 = ['88d375bc79ef32532e50abc7a1305908', '2022610a5d1a8455abc49cae1a831a9e', \
 'adfe9029a1ca723eb8966aeece708f87']
bag2 = ['f297885d4918000ec8724d5673a063a6']
bag3 = ['774fcae246e7ad25c8724d5673a063a6']

bag = ['bag,traveling bag,travelling bag,grip,suitcase', '02773838', [bag1, bag2,bag3]]

#bottle
bottle1 = ['726a6ce68bcb4b2d14513156cf2b8d0d', '957f897018c8c387b79156a61ad4c01', \
'1ffd7113492d375593202bf99dddc268'] + ['81bbf3134d1ca27a58449bd132e3a3fe', '940b9e91a4a32a4130612f5c0ef21eb8', \
'edfcffbdd585d00ec41b4a535d52e063']
bottle2 = ['fa44223c6f785c60e71da2487cb2ee5b', '9f2bb4a157164af19a7c9976093a710d', \
 '452c562f86da1ca7bdcda0bf7e7b4744']+ ['8309e710832c07f91082f2ea630bf69e', 'dc0926ce09d6ce78eb8e919b102c6c08', \
'3f91158956ad7db0322747720d7d37e8']
bottle = ['bottle', '02876657', [bottle1, bottle2]]

#bowl
bowl1=['454fa7fd637177cf2bea4b6e7618432', 'bbf4b10b538c7d03bcbbc78f3e874841', \
'c25fd49b75c12ef86bbb74f0f607cdd', 'c6be3b333b1f7ec9d42a2a5a47e9ed5'] +['468b9b34d07eb9211c75d484f9069623', 'cfac22c8ca3339b83ce5cb00b21d9584', \
'429a622eac559887bbe43d356df0e955', '4530e6df2747b643f6415fd62314b5ed']
bowl = ['bowl', '02880940', [bowl1]]

# bowl_like_but_weird = ["9dcfab1fab6232003bec56bff764ba78",
#                       "4967063fc3673caa47fe6b02985fbd0",
#                       ]
# bowl_exclude = ["d28f7a7a8fbc5fc925b5a13384fa548b", 
#                "baa4d6ca9521a55b51d6f7c8e810987e",
#                 "e16e27d3670853b12d4e6b2984840098",   
#                 "a9ba34614bfd8ca9938afc5c0b5b182", 
#                 "c399bfee7f25f0ff95aab58c1db71c10",
#                 "e4c871d1d5e3c49844b2fa2cac0778f5",
#                 "3bedfd655c45c6747bc26b16f550876f",
#                 "1f910faf81555f8e664b3b9b23ddfcbc",
#                 "aad3cd6a40a15cb8664b3b9b23ddfcbc",
#                 "9a6cec1cfca7e8b5ebc583f22bd58b85",
#                 "e3095ecacc36080cb398a1cfd1079875",
#                 "bb7586ebee0dc2be4e346ee2650d150",
#                 "bec41cc7f631e00b1bf40e92cbe8569f",
#                 "d78860e8318fb75a12a72bbc636a1f9d",
#                 "2ffe06ee50ec1420adbe0813683fcfd0",
#                 "4967063fc3673caa47fe6b02985fbd0",
#                 "5f2ef71aa9b94edbb84959963148f2a2",
#                 "571f294b5ab7e47a5517442d05421cb",
#                 "3f56833e91054d2518e800f0d88f9019",
#                 "4cf18594e8851a3d3a5e6305a3a7adee",
#                 "38d2122b2aa05f92e1b9b7b3765e2bea",
#                 "b007af6eabaa290dd42b9650f19dd425",
#                 "5019f979a6a360963a5e6305a3a7adee",
#                 "709778e5ebb67e32b396efafc0d0d8ac",
#                 "b65cbede8f483d51f7de4d28691515e1",
#                 "524eb99c8d45eb01664b3b9b23ddfcbc",
#                 "ee3b4a98683feab4633d74df68189e22",
#                 "12ddb18397a816c8948bef6886fb4ac",
#                 "eff9864bfb9920b521374fbf1ea544c",
#                 "2a1e9b5c0cead676b8183a4a81361b94",
#                 "b941ca9770b59c3918a27ff49f2f297f",
#                 "5aad71b5e6cb3967674684c50f1db165",
#                 "7995c6a5838e12ed447eea2e92abe28f",
#                 "4017528ab5816338664b3b9b23ddfcbc",
#                 "faa200741fa93abb47ec7417da5d353d",
#                 "446583d7bf857dced5cb6d178687b980",
#                 "d162f269bc5fb7da85df81f999944b5d",
#                 "e3d4d57aea714a88669ff09d7001bab6",
                
#                ]

#can,tin,tin can
can1=['7bd05573c09fb0c2af39ccaab9314b14', '17ef524ca4e382dd9d2ad28276314523', \
'd3e24e7712e1e82dece466fd8a3f2b40', '990a058fbb51c655d773a8448a79e14c', \
'bf974687b678c66e93fb5c975e8de2b7'] + ['7e9ea6ccb5a1689a3299d37e620d2f2', 'a7e0a5111234031fbd685fccc028124d', \
     'baaa4b9538caa7f06e20028ed3cb196e']
can = ['can,tin,tin can', '02946921', [can1]]

#clock
# clock1=['247ca61022a4f47e8a94168388287ad5', '3521751471b748ff2846fa729d90e125', \
# '772f1ed7779459e8d835a15bbfa33167']
clock1=['3521751471b748ff2846fa729d90e125']
clock2=['1d5a354ee3e977d7ce57d3de4658a486', 'e8c8090792a48c08b045cbdf51c133cd', \
'37a995cd9a8a125743dbb6421d614c0d']
clock3 = ['253156f6fea2d869ff59f04994ef1f0c', \
    '57c8fe2fb023b648ae29bc118c70aa10']
clock4 = ['6d12c792767c7d46bf3c901830f323db']
clock = ['clock', '03046257', [clock1, clock2, clock3, clock4]]

#jar
jar1=['1da5c02a928b8889dfd1be983f4bd279', 'a1fae2bdac896ab83a75e6d000e08290', \
'8e5595181e9eef7d82ec48ff3a4fe07c', 'ee10db30b91b9683f9215f842248bc25']
jar2=['c2bd95766b5cade2621a1668752723db']
jar3=['a18343c4b0a8026faca4186c3b7dd23d', '57693fd44e597bd8fed792cc021b4e66', \
'cb451e92ce5a422c9095fe1213108032']
jar = ['jar', '03593526', [jar1,jar2,jar3]]

#mug
mug1=['b6f30c63c946c286cf6897d8875cfd5e', 'f7d776fd68b126f23b67070c4a034f08', \
'403fb4eb4fc6235adf0c7dbe7f8f4c8e', '46ed9dad0440c043d33646b0990bb4a', \
'896f1d494bac0ebcdec712af445786fe', 'edaf960fb6afdadc4cebc4b5998de5d0']
mug2=['5582a89be131867846ebf4f1147c3f0f', 'c39fb75015184c2a0c7f097b1a1f7a5', \
'8b1dca1414ba88cb91986c63a4d7a99a', '599e604a8265cc0a98765d8aa3638e70']
mug = ['mug', '03797390', [mug1, mug2]]

preselect = [bag, bottle, bowl, can, clock, jar, mug]

csv_columns = ['synsetId', 'name', 'ShapeNetModelId', 'objId']
dict_data = []
obj_id = 0
for cat_name, cat_id, cat_objects in preselect:
    for objs in cat_objects:
        for obj in objs:
            row = {'synsetId': cat_id, 'name': cat_name, \
                  'ShapeNetModelId': obj, 'objId': obj_id}
            dict_data.append(row)
        obj_id += 1

def write_to_csv(csv_file, dict_data, csv_columns):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")

csv_file = os.path.join(args.csv_file_dir, "preselect_table_top.csv")
write_to_csv(csv_file, dict_data, csv_columns)


df = pd.read_csv(csv_file)
train,test = train_test_split(dict_data, test_size=0.3)
train_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_train.csv")
test_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_test.csv")

write_to_csv(train_csv_file_path, train, csv_columns)
write_to_csv(test_csv_file_path, test, csv_columns)
df_train = pd.read_csv(train_csv_file_path)
df_test = pd.read_csv(test_csv_file_path)