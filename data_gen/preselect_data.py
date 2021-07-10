import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import os 
import trimesh
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option("--csv_file_dir", dest="csv_file_dir")
parser.add_option("--shapenet_filepath", dest="shapenet_filepath")

(args, argss) = parser.parse_args()
'''
# def output_selected(csv_file, selected_l):
#     csv_df = pd.read_csv(csv_file)
#     res = []
#     for idx in selected_l:
#         res.append(csv_df.iloc[idx]["fullId"].split(".")[-1])
#     return res
# triangle_body_strap_selected = [1,2,4,6,15,17,24,38,45,57,58,61,62,63,67,68,77,78,80]
# square_body_strap_selected = [11,28,37,69]
# triangle_body_straps = output_selected('/raid/xiaoyuz1/bag.csv', triangle_body_strap_selected)
# square_body_straps = output_selected('/raid/xiaoyuz1/bag.csv', square_body_strap_selected)
# bag = ['bag,traveling bag,travelling bag,grip,suitcase', '02773838', [bag1, bag2,bag3]]
# square_basket_selected = [0,3,7,21,25,29,30,31,39,42,49,54,66,71,74,76,78,79,80,81,87]
# round_basket_selected = [15,18,28,37,44,48,51,94]
# square_baskets =  output_selected('/raid/xiaoyuz1/basket.csv', square_basket_selected)
# round_baskets = output_selected('/raid/xiaoyuz1/basket.csv', round_basket_selected)
# basket = ['basket,handbasket', '02801938', [square_baskets, round_baskets]]
# bowl_df = pd.read_csv("/raid/xiaoyuz1/bowl.csv")
# bowl_selected = [8,19,22,29,33,34,37,61,64,66,75,78,80,90,94, \
#                  102,104,123,130,131,135,140,147,153,183]
# bowls = output_selected("/raid/xiaoyuz1/bowl.csv", bowl_selected)

# bowl = ['bowl', '02880940', [bowls]]
# print(len(bowls))
# wine_bottle_df = pd.read_csv("/raid/xiaoyuz1/wine_bottle.csv")
# wine_bottle_selected= [0,1,2,4,10,11,17,22,23,24,35,37,38,48, \
#                        51,54,60,64,67,69,74,77,83,84,95,99,112,115,137,156]
# wine_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",wine_bottle_selected)


# sprayers = ["cbc1cbc9cf65e9c2fd1d6016d24cc8d", 
#             "9b9a4bb5550f00ea586350d6e78ecc7", 
#            "d45bf1487b41d2f630612f5c0ef21eb8"]


# round_bottles_selected = [18,31,42,58,72,75,87,93,102,113,135,142,158, \
#                           159,183,188,196,212,223,229,233,244,247,256, \
#                           128,143,206,218,220]
# round_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",round_bottles_selected)


# square_water_bottle_selected=[63,65,106,109,170]
# square_water_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",square_water_bottle_selected)


# strap_bottles_selected = [42,43,114]
# strap_bottles = output_selected("/raid/xiaoyuz1/wine_bottle.csv",strap_bottles_selected)



# bottle = ['bottle', '02876657', [wine_bottles, sprayers,round_bottles,square_water_bottles, \
#                                 strap_bottles]]
# print(np.sum([len(l) for l in bottle[2]]))
# can_selected = list(np.arange(108))
# can_selected.remove(10)
# can_selected.remove(32)
# can_selected.remove(71)
# can_selected.remove(98)
# cans = output_selected("/raid/xiaoyuz1/can.csv",can_selected)
# can = ['can,tin,tin can', '02946921', [cans]]
# print(np.sum([len(l) for l in can[2]]))
# clock1=['3521751471b748ff2846fa729d90e125']
# clock2=['1d5a354ee3e977d7ce57d3de4658a486', 'e8c8090792a48c08b045cbdf51c133cd', \
# '37a995cd9a8a125743dbb6421d614c0d']
# clock3 = ['253156f6fea2d869ff59f04994ef1f0c', \
#     '57c8fe2fb023b648ae29bc118c70aa10']
# clock4 = ['6d12c792767c7d46bf3c901830f323db']
# clock = ['clock', '03046257', [clock1, clock2, clock3, clock4]]
# jar1=['c1be3d580b4088bf4cc80585c0d3d970', 
#       '1da5c02a928b8889dfd1be983f4bd279', 
#       'a1fae2bdac896ab83a75e6d000e08290', 
#       '8e5595181e9eef7d82ec48ff3a4fe07c', 
#       'ee10db30b91b9683f9215f842248bc25',
#       '386a6dffaef80143fa0d49b618d792ba']
# jar2=['a18343c4b0a8026faca4186c3b7dd23d', '57693fd44e597bd8fed792cc021b4e66', \
# 'cb451e92ce5a422c9095fe1213108032']
# jar = ['jar', '03593526', [jar1,jar2]]
# laptop_csv = pd.read_csv("/raid/xiaoyuz1/laptop.csv")
# laptop_selected = [  1,   2,   3,   4,   6,   7,   8,   9,  11,  13,  15,  16,  19,
#         20,  22,  25,  26,  30,  31,  32,  34,  35,  37,  38,  39,  41,
#         43,  44,  45,  48,  49,  53,  55,  57,  58,  59,  60,  61,  63,
#         66,  67,  68,  69,  70,  71,  73,  74,  75,  76,  77,  80,  81,
#         83,  86,  88,  90,  92,  93,  97,  98,  99, 100, 101, 104, 105,
#        110, 112, 113, 114, 115, 116, 117, 118, 120, 123, 124, 125, 127,
#        128, 129, 130, 131, 134, 135, 136, 137, 138, 139, 140, 141, 142,
#        145, 146, 148, 150, 158, 159, 161, 163, 165]
# laptops = output_selected("/raid/xiaoyuz1/laptop.csv",laptop_selected)
# cameras_selected = [1,4,5,6,13,14,17,21,22,27,31,37,40,50,61,77,81,93,95,101,103,104,109,111]
# cameras = output_selected("/raid/xiaoyuz1/camera.csv",cameras_selected)
# square_handle_mug_selected = [1,21,23,38,41,53,67,69,82,121,166,168,171,177,211]
# round_hanle_mug_selected = [3,10,11,13,15,17,18,25,27,30,31,35,44,46,51,54,55,57, \
#                             63,65,72,74,75,83,84,87,88,96,98,99,108,110,112,120,122, \
#                             123,130,137,140,152,159,160,164,169,170,176,184,206]
# square_handle_mugs = output_selected("/raid/xiaoyuz1/mug.csv",square_handle_mug_selected)
# round_hanle_mugs = output_selected("/raid/xiaoyuz1/mug.csv",round_hanle_mug_selected)
# cup_like_mug_selected = [61,86,139,151]
# cup_like_mugs = output_selected("/raid/xiaoyuz1/mug.csv",cup_like_mug_selected)
# mug = ['mug', '03797390', [square_handle_mugs, round_hanle_mugs,cup_like_mugs]]

'''

def output_selected(csv_file, selected_l):
    csv_df = pd.read_csv(csv_file)
    res = []
    for idx in selected_l:
        res.append(csv_df.iloc[idx]["fullId"].split(".")[-1])
    return res

######################################################################vvbag
triangle_body_straps = ['adfe9029a1ca723eb8966aeece708f87',
 'd5881d42567baaf5dc19a9901b7e9a4f',
 '83ab18386e87bf4efef598dabc93c115',
 '45f56ad8c0f4059323166544c0deb60f',
 '157fa29bcf6b890d76bb31db0358e9c6',
 'e90e27c3020d25dd76bb31db0358e9c6',
 '74c548ef3ca7b1987515e7bb7dba4019',
 'c3a5009c4867f7293c8d0fdfb1cc2535',
 '8bc53bae41a8105f5c7506815f553527',
 '4e4fcfffec161ecaed13f430b2941481',
 'e49f6ae8fa76e90a285e5a1f74237618',
 'cbc1512b28e9ed382bef451de0ed6949',
 '581af450a3a8009abc49cae1a831a9e',
 '6881468dd51c214922022e93ae2b2b5b',
 '88d375bc79ef32532e50abc7a1305908',
 '2022610a5d1a8455abc49cae1a831a9e',
 'e5fea6c1dacf3ed4cd99ccc7ff441abf',
 '5d4f774fdda6fa1dcd99ccc7ff441abf',
 '2ca6df7a5377825cfee773c7de26c274']
square_body_straps = ['774fcae246e7ad25c8724d5673a063a6',
 'd3bd250ca3cb8e29976855a35549333',
 '7565e6f425dd6d376d987ae9a225629c',
 '1342fc4f613687f92569b963af585e33']
bag = ['bag,traveling bag,travelling bag,grip,suitcase', '02773838', [triangle_body_straps, square_body_straps]]

######################################################################bottle
wine_bottles = ['d851cbc873de1c4d3b6eb309177a6753',
 '546111e6869f41aca577e3e5353dd356',
 'e101cc44ead036294bc79c881a0e818b',
 '9f2bb4a157164af19a7c9976093a710d',
 '8cd9b10f611ac28e866a1445c8fba9da',
 'fa44223c6f785c60e71da2487cb2ee5b',
 'd297d1b0e4f0c244f61150ce90be197a',
 'af3dda1cfe61d0fc9403b0d0536a04af',
 '3f91158956ad7db0322747720d7d37e8',
 'dc0926ce09d6ce78eb8e919b102c6c08',
 'a34966853ab2272ab2047d3072d5e051',
 '8309e710832c07f91082f2ea630bf69e',
 '799397068de1ae1c4587d6a85176d7a0',
 '114509277e76e413c8724d5673a063a6',
 'f83c3b75f637241aebe67d9b32c3ddf8',
 'f853ac62bc288e48e56a63d21fb60ae9',
 'b742bd7f675191c24ad6a0c67b7f7a5b',
 '831918158307c1eef4757ae525403621',
 'ab6792cddc7c4c83afbf338b16b43f53',
 'abe3a232d973941d49a3c1009fa79820',
 'd85f1862dfe799cbf78b6c51ab8f145e',
 '7980922e83b5461febe67d9b32c3ddf8',
 '158634b1d7d010eeebe67d9b32c3ddf8',
 '9012b03ddb6d9a3dfbe67b89c7bdca4f',
 '684ff2b770c26616d3dfba73f54d35bb',
 '452c562f86da1ca7bdcda0bf7e7b4744',
 '22d18e34097ec57a80b49bbcfa357c86',
 'a87fc2164d5bb73b9a6e43b878d5b335',
 '11fc9827d6b467467d3aa3bae1f7b494',
 '7b1fc86844257f8fa54fd40ef3a8dfd0']
sprayers = ["cbc1cbc9cf65e9c2fd1d6016d24cc8d", 
            "9b9a4bb5550f00ea586350d6e78ecc7", 
           "d45bf1487b41d2f630612f5c0ef21eb8"]
round_bottles = ['1d4480abe9aa45ce51a99c0e19a8a54',
 'd9aee510fd5e8afb93fb5c975e8de2b7',
 '9eccbc942fc8c0011ee059e8e1a2ee9',
 '46b5318e39afe48a30eaaf40a8a562c1',
 'c5e425b9b1f4f42b6d7d15cb5e1928e',
 'defc45107217afb846564a8a219239b',
 '1d4093ad2dfad9df24be2e4f911ee4af',
 '6b884893938943f70b5ecdc69e74dca',
 '15787789482f045d8add95bf56d3d2fa',
 '632a0bd7869cb763780bbc8616cb15f8',
 '957f897018c8c387b79156a61ad4c01',
 '1ffd7113492d375593202bf99dddc268',
 '371b604f78300e02d76ab6ff59fe7e10',
 '726a6ce68bcb4b2d14513156cf2b8d0d',
 'bf7ecd80f7c419feca972daa503b3095',
 '6ebe74793197919e93f2361527e0abe5',
 'b1271f402910cf05cfdfe3f21f42a111',
 '3336ff069b249afcaea6c6fc97ee6184',
 '44dae93d7b7701e1eb986aac871fa4e5',
 'cec14f91bf10f86e8291825d073a05e1',
 '4185c4eb651bd7e03c752b66cc923fdb',
 '72d49a11c34a3b6880dd154b5d9c087',
 '618d55a791b8280cf256a8c3e3396495',
 '8adc0ce79962ac2021072d05c97a5e0a',
 'd3b53f56b4a7b3b3c9f016d57db96408',
 'bcacdde81063a5df30612f5c0ef21eb8',
 '940b9e91a4a32a4130612f5c0ef21eb8',
 'd8021dc9fc9109b130612f5c0ef21eb8',
 '81bbf3134d1ca27a58449bd132e3a3fe']
square_water_bottles = ['dacc6638cd62d82f42ebc0504c999b',
 'f4851a2835228377e101b7546e3ee8a7',
 '8f2b8d281413a8bd5b326e4735ab9003',
 'a1275bd03ab15100f6dbe3dc17d6cdf7',
 'cf7a79435eb5b1bdb0be98650cd7fb6f']
strap_bottles = ['9eccbc942fc8c0011ee059e8e1a2ee9',
 '7984d4980d5b07bceba393d429f71de3',
 '681e91bbadebeac529471183b63392dc']
bottle = ['bottle', '02876657', [wine_bottles, sprayers,round_bottles,square_water_bottles, strap_bottles]]

######################################################################bowl
bowls=['8bb057d18e2fcc4779368d1198f406e7',
 '899af991203577f019790c8746d79a6f',
 '7c43116dbe35797aea5000d9d3be7992',
 'c82e28d7f713f07a5a15f0bff2482ab8',
 '429a622eac559887bbe43d356df0e955',
 'a1393437aac09108d627bfab5d10d45d',
 '9a52843cc89cd208362be90aaa182ec6',
 'c25fd49b75c12ef86bbb74f0f607cdd',
 '960c5c5bff2d3a4bbced73c51e99f8b2',
 'ab2fd38fc4f37cce86bbb74f0f607cdd',
 '3152c7a0e8ee4356314eed4e88b74a21',
#  '6930c4d2e7e880b2e20e92c5b8147e4a',
 'e30e5cbc54a62b023c143af07c12991a',
 '8d1f575e9223b28b8183a4a81361b94',
 '708fce7ba7d911f3d5b5e7c77f0efc2',
 'e816066ac8281e2ecf70f9641eb97702',
 '4fdb0bd89c490108b8c8761d8f1966ba',
 'b4c43b75d951401631f299e87625dbae',
 '9024177b7ed352f45126f17934e17803',
 'ce905d4381d4daf65287b12a83c64b85',
 '4530e6df2747b643f6415fd62314b5ed',
 'afb6bf20c56e86f3d8fdbcba78c84028',
 '817221e45b63cef62f74bdafe5239fba',
 '3a7737c7bb4a6194f60bf3def77dca62',
 'a042621b3378bc18a2c59a4d90e63212']
bowl = ['bowl', '02880940', [bowls]]


######################################################################can,tin,tin can
can1 = ['baaa4b9538caa7f06e20028ed3cb196e',
 'bf974687b678c66e93fb5c975e8de2b7',
 '3a7d8f866de1890bab97e834e9ba876c',
 '343287cd508a798d38df439574e01b2',
 '38dd2a8d2c984e2b6c1cd53dbc9f7b8e',
 '5bd768cde93ec1acabe235874aea9b9b',
 '3c8af6b0aeaf13c2abf4b6b757f4f768',
 '3703ada8cc31df4337b00c4c2fbe82aa',
 '4a6ba57aa2b47dfade1831cbcbd278d4',
 'b36902ae19ac77f2a89caedb1243a99',
 'c4bce3dc44c66630282f9cd3f45eaa2a',
 '7bd05573c09fb0c2af39ccaab9314b14',
 '2eeefdfc9b70b89eeb153e9a37e99fa5',
 '4cc3601af4a09418b459058f42771eff',
 '90d40359197c648b23e7e4bd2944793',
 'd052c17866cf5cf8387e8ce4aad01a52',
 'd511899945a400b183b4ef314c5735aa',
 'ace554442d7d530830612f5c0ef21eb8',
 'd801d5b05f7d872341d8650f1ad40eb1',
 '10c9a321485711a88051229d056d81db',
 '129880fda38f3f2ba1ab68e159bfb347',
 '11c785813efc4b8630eaaf40a8a562c1',
 'd44cec47dbdead7ca46192d8b30882',
 '7883b684806946276f056d414894e46d',
 'f4108f92f3f12f99e3ecb6fd6ed1dd90',
 'f4ad0b7f82c36051f51f77a6d7299806',
 '295be2a01cb9f29b716714dd1fd945b7',
 '17ef524ca4e382dd9d2ad28276314523',
 '85fa7911905e932bf485d100eb31d589',
 'ebcbb82d158d68441f4c1c50f6e9b74e',
 '540cb2a72840ec1130612f5c0ef21eb8',
 'd3e24e7712e1e82dece466fd8a3f2b40',
 'd53fd8769ff53b9a2daf5369a15791ca',
 '19fa6044dd31aa8e9487fa707cec1558',
 'a7e0a5111234031fbd685fccc028124d',
 '990a058fbb51c655d773a8448a79e14c',
 '91483776d1930de7515bc9246d80fdcc',
 '637006720b7886e0c7a50f701fe65efe',
 'f755800334fcb49b450911b585bf4df8',
 'fe6be0860c63aa1d8b2bf9f4ef8234',
 '4d4fc73864844dad1ceb7b8cc3792fd',
 '9effd38015b7e5ecc34b900bb2492e',
 '1beb16ca4f1fd6e42150a45ec52bcbd7',
 '788094fbf1a523f768104c9df46104ca',
 '5505ddb926a77c0f171374ea58140325',
 'a70947df1f1490c2a81ec39fd9664e9b',
 'b45716cd72f1e9172fbee880b9f634b4',
 '59dc6215c5eeb1ee749d52b3e269018',
 '4961c651fdb6b45fa57e04ecfc2d7abd',
 '3fd8dae962fa3cc726df885e47f82f16',
 '51f0b5ce9711f63bb15a1cf05bc3d210',
 '7e9ea6ccb5a1689a3299d37e620d2f2',
 'e928ebedb376404f8e5478c8425d418a',
 '70446b4dc6649d0173c0d206b70af93c',
 'b6c4d78363d965617cb2a55fa21392b7',
 '6a703fd2b09f50f347df6165146d5bbd',
 'fcd14e7ad72bfa70471c65fdb52b88b2',
 '60f4012b5336902b30612f5c0ef21eb8',
 '7b643c8136a720d9db4a36333be9155',
 '9b1f0ddd23357e01a81ec39fd9664e9b',
 'fd40fa8939f5f832ae1aa888dd691e79',
 'e8c446c352b84189bc8d62498ee8e85f',
 'e532d6bd597d4e91e3e4737f6033f0f8',
 '29bc4b2e86b91d392e06d87a0fadf00',
 '25c253b2e40b6f4ea61649b05d63e9bb',
 'dc815e056c71e2ed7c8ed5da8582ce91',
 'bb487f4b77abbef0a8d976d1bb073663',
 'adaaccc7f642dee1288ef234853f8b4d',
 '297b1ba8efcbb6a86128527957a7bb1',
 'eac30c41aad2ff27c0ca8d7a07be3be',
 '5c326273d61272ad4b6e06cda31f9bc6',
 '669033f9b748c292d18ceeb5427760e8',
 '6f2c06bb129e52be63ed57e35c972b4b',
 'be67418a10003cc9eae3efbc9dbeea',
 '56dfb6a30f498643bbf0c65ae96423ae',
 'ac66fb0ff0d50368ced499bff9a86355',
 'a5bab9546d6a1baa33ff264b2ec3aaa9',
 '408028e3bdd8d05b2d6c8e51365a5a87',
 '203c5e929d588d07c6754428123c8a7b',
 'b1980d6743b7a98c12a47018402419a2',
 'f8fd565d00a7a6f9fef1ca8c5a3d2e08',
 '3a9041d7aa0b4b9ad9802f6365035049',
 '52e295024593705fb00c487926b62c9',
 '147901ede668deb7d8d848cc867b0bc8',
 'fac6341f9e5bfddaf5aaab5ed17143d6',
 '8cf26f6912f4a9e34a045a96d74810ea',
 '70172e6afe6aff7847f90c1ac631b97f',
 'a7059f3bf782790654976319206e3c9c',
 'af444e72a44e6db153c22afefaf6f2a4',
 '3fd196c22459cc66c8687ff9b0b4e4ac',
 '96387095255f7080b7886d94372e3c76',
 'a087f6b5ea424ccc785f06f424b9d06',
 'efa4c58192cc5cf89e7b86262150f',
 'f390f6d6418135b69859d120d9976364',
 '926d45845c2be919f8cd8f969312dc1',
 'fd73199d9e01927fffc14964b2380c14',
 '28c17225887339bd6193d9e76bb15876',
 'f6316c6702c49126193d9e76bb15876',
 '100c5aee62f1c9b9f54f8416555967',
 '6b2c6961ad0891936193d9e76bb15876',
 '97ca02ee1e7b8efb6193d9e76bb15876',
 'e706cf452c4c124d77335fb90343dc9e',
 'bea7315d4410d0ce83b1cdcee646c9a4']
can = ['can,tin,tin can', '02946921', [can1]]

######################################################################clock
clock1=['3521751471b748ff2846fa729d90e125']
clock2=['1d5a354ee3e977d7ce57d3de4658a486', 'e8c8090792a48c08b045cbdf51c133cd', \
'37a995cd9a8a125743dbb6421d614c0d']
clock3 = ['253156f6fea2d869ff59f04994ef1f0c', \
    '57c8fe2fb023b648ae29bc118c70aa10']
clock4 = ['6d12c792767c7d46bf3c901830f323db']
clock = ['clock', '03046257', [clock1, clock2, clock3, clock4]]

######################################################################jar
jar1=['c1be3d580b4088bf4cc80585c0d3d970', 
      '1da5c02a928b8889dfd1be983f4bd279', 
      'a1fae2bdac896ab83a75e6d000e08290', 
      '8e5595181e9eef7d82ec48ff3a4fe07c', 
      'ee10db30b91b9683f9215f842248bc25',
      '386a6dffaef80143fa0d49b618d792ba']
jar2=['a18343c4b0a8026faca4186c3b7dd23d', '57693fd44e597bd8fed792cc021b4e66', 'cb451e92ce5a422c9095fe1213108032']
jar = ['jar', '03593526', [jar1,jar2]]

######################################################################laptop
laptops = ['97e94d800fd6dc07dbaa6d42a4980930',
       '82edd31783edc77018a5de3a5f9a5881',
       '5fb5b09b324dc153ed883f1f11a51185',
       '9bfb4fcfe4fc903ca1fa797bebd1cbce',
       'b806daf849a5dba289c212008d2a390e',
       '3088048452f40a8965932bedd33dbd98',
       '76005568c6a76385c8f56abbf37ac61c',
       'f5fc954736b06be15fd06491ae919ea3',
       'e55ececde88255b93e73f3893a7337bb',
       'c8309234b360aa2c747803756378b292',
       '1ce087aba42caa73b8152979f6537fad',
       'b95ca4fa91a57394e4b68d3b17c43658',
       '11448f34681c545439f3410d5f76299b',
       '10f18b49ae496b0109eaabd919821b8',
       '5eefc8f2d755c843614c1d5b48350fb',
       '894e47a6adb76680d4eb7e68e898dc44',
       '53bdcb3c0b28a51b580ee4476b0b0ff',
       '4d3dde22f529195bc887d5d9a11f3155',
       '1dc28ae9dcfc3d773487ed70d4534caf',
       'f53ea19f871a80d420685b5a7e34b501',
       '34715b469341fd4ce4b68d3b17c43658',
       'fd2f7a1c6eb7a0d4803dd502eefd8dc3',
       '24721e62300a3c4f98be7382d7a678c3',
       '67e882442eb4c03255e8ddeaf1791474',
       '1ce688d90a2010a69718283200011d2a',
       '34913d1dcde913848bd55eee82fc09d6',
       '9c6176af3ee3918d6140d56bf601ecf2',
       '63e5805f8bd216313bba289a9fdd2a7d',
       'f496fa98fb41f6b1ea5768960d4a805c',
       '39778c495baf4bd9ac41b162b25b4656',
       'e9e28a11f71337fc201115f39f20d1ff',
       '9eb06745445806576e14170ade57410',
       'ebc59a9b4291ce274c3c121820623e4e',
       'b51683c6285fa0f69067ac5c9d4ee692',
       '9785a579ea791039c639c533e8b5aec1',
       '621882a4afd2a126369873c1090720a1',
       'bad11c851d356f6363920080bd2c2ed3',
       'bd879abd9b5c15d75d66db82418edc83',
       '2416c2fcbae368a7b95c83f902f3aac0',
       '61972112749c9beeb95c80bb1ee18b0e',
       'bb33d26f324d19d94845e0946708405d',
       '69ca190b89d622d6def5bf46a0f0ff11',
       '4f3575df3821e08c466909b3e9553909',
       '3b2bddb0e8ff57c85831a0624cf5a945',
       'b9eb4471432dbbc3e4b68d3b17c43658',
       '1f507b26c31ae69be42930af58a36dce',
       'cb090bd99ed76f9689661310be87a70d',
       'f14056ee8bbebeecc1b05209f08e5ec6',
       '9cb54ee28aec14013cb02989e2da5a2a',
       'ce13a85fb71694fcb611830890d7aa97',
       'eda651202abbb90c94daa4565dd30535',
       'cc691d9e8e189ce47a381a112bfd785',
       '9a0cafe6dfafe0503fe4aa36ea0cc020',
       '59bf155f2c4d655894c7c2dce500aa02',
       '6c6a96e4486cc02cda66ecbb2c411f37',
       '342150823878f1ec9aeacfc2a1a52243',
       'fdec2b8af5dd988cef56c22fd326c67',
       '3934cb69fe3584ef8f6cc6fefa15515a',
       'b28638f8c153b9333639bdeac6c4cb9a',
       '92702a4de72aac3e20685b5a7e34b501',
       'dd205f0f8bc78bae359d7b39cfebb287',
       '92e6341ab62ce4875c0be177939e290',
       '6b78948484df58cdc664c3d4e2d59341',
       '56517e22979b563460ac9d6174947ab2',
       'cbcb79f534518dfbcfe78be5b7b99c8d',
       '87ffe7aaa7304b1b775a6b1e21d79260',
       '1b67b4bfed6688ba5b22feddf58c05e1',
       'a2346fd2b76d58bb2aacdb6e0b5b6c83',
       'dc264eab83ca12b3da4c0d8596dff972',
       '17069b6604fc28bfa2f5beb253216d5b',
       '7da4f6dcda3fc40db1ef4670cf6c2a91',
       'dda1832a36858f06ea791b47ef8b531a',
       'a4b410734514306ac401e233323032d6',
       '39e80a6570002a431181170a86d04637',
       '7f6bd9a933f6cbd33585ebacb5c964c2',
       '62036dabbd9ffa71549c63d8891393c6',
       '1f8338b068279743f8267dd94d223348',
       '19117e74b575cdca82555f4c45537277',
       'f4c6dec2587420aaf92e5f8fe21ceb0',
       'f9f484e4364663d61bb5a64b9a0f552b',
       '37b5e63e9f80e2402052a3b22ea3f616',
       '81ba52b908e4e1bc8ca8637757ac3f67',
       'a4eb5dea7eda6ab58e5480b52f7861ca',
       '62818ccc8dd6a3ecb300959c6f62c5f9',
       '28fbfd8b8c9c6f16e1e44e2fc05361d9',
       '40e0263822860cc5d69e26904fa68e7f',
       '6ba2cdd89db562a08329485eab7078c4',
       '6f6c8762f55f94e24145b5f47fac09a5',
       '5be7aff08cd8dc37253cd18ba2e1c61e',
       '1497a7a1871af20162360e5e854659a',
       'cc0535a34cdc7d676bf98d15712168f',
       '774f062adc4130eca309fc846e6b0c18',
       '593d32126cea601d5a952e55d06611ce',
       'e466c2c86a439c1faebff3b001eb4a27',
       '26ad81059039dfcca3e30e62a8e6f77f',
       'd37f5f4d76bb5aee6d0e9ec8b698ce7a',
       '91b857a748f0ccf2ab0b172d4dea80cd',
       '247012532a6342f424afdcb79b0329d8',
       '72133503ddaf54483e725ee552a0026',
       'd88e4093a73064b6b44d9812f259e403']
laptop = ['laptop,laptop computer', '03642806', [laptops]]

######################################################################camera
cameras = ['97690c4db20227d248e23e2c398d8046',
 'd6721b4ee3d004b8c7e03242f1bf8d19',
 'e9e22de9e4c3c3c92a60bd875e075589',
 '509017601d92a7d1db286a46dfc37518',
 'fb3b5fae94f7b02a3b269928487f8a4c',
 'db663e3f7ee9869f5c351e299b24e355',
 '235a6dd25c0a6f7b66f19f26ac490096',
 '46c09085e451de8fc3c192db90697d8c',
 '6d036fd1c70e5a5849493d905c02fa86',
 '2693df58698a2ca29c723bc28575d785',
 '6c14c6f6cca53a6710d0920f7087353b',
 'a4b0c73d0f12bc75533388d244d29c5c',
 '4f2a9bf0d8eb00e0a570c6c691c987a8',
 '9726bf2b38d817eab169d2793795b997',
 '5d42d432ec71bfa1d5004b533b242ce6',
 '82819e1201d2dc583a3e53900c6cbba',
 'b92acfcd92408529d863a5ae2bdfd29',
 'e67273eff31fabce656c3a28e34d04c4',
 'f1540b3d6da38fbf1d908355fc20d631',
 'e9f2c58d90e723f7cc57882dfaef8a57',
 'a3c9dcaada9e04d09061da204e7c463c',
 '51176ec8f251800165a1ced01089a2d6',
 'cda4fc24b2a602b5b5328fde615e4a0c',
 '9e91f482b829c4d1e9fff7dfdebc774b']
camera = ['camera,photographic camera', '02942699', [cameras]]

######################################################################mug
square_handle_mugs = ['b88bcf33f25c6cb15b4f129f868dedb',
 'ec846432f3ebedf0a6f32a8797e3b9e9',
 '639a1f7d09d23ea37d70172a29ade99a',
 '7d6baadd51d0703455da767dfc5b748e',
 'e6dedae946ff5265a95fb60c110b25aa',
 'a0c78f254b037f88933dc172307a6bb9',
 'f1c5b9bb744afd96d6e1954365b10b52',
 '57f73714cbc425e44ae022a8f6e258a7',
 '9d8c711750a73b06ad1d789f3b2120d0',
 '10f6e09036350e92b3f21f1137c3c347',
 '5d72df6bc7e93e6dd0cd466c08863ebd',
 'c0c130c04edabc657c2b66248f91b3d8',
 '3d1754b7cb46c0ce5c8081810641ef6',
 '4b7888feea81219ab5f4a9188bfa0ef6',
 '34869e23f9fdee027528ae0782b54aae']
round_hanle_mugs = ['b6f30c63c946c286cf6897d8875cfd5e',
 'b7e705de46ebdcc14af54ba5738cb1c5',
 '1ea9ea99ac8ed233bf355ac8109b9988',
 '1be6b2c84cdab826c043c2d07bb83fc8',
 'b811555ccf5ef6c4948fa2daa427fe1f',
 'cf777e14ca2c7a19b4aad3cc5ce7ee8',
 '1eaf8db2dd2b710c7d5b1b70ae595e60',
 '9c930a8a3411f069e7f67f334aa9295c',
 'c6bc2c9770a59b5ddd195661813efe58',
 '187859d3c3a2fd23f54e1b6f41fdd78a',
 '2852b888abae54b0e3523e99fd841f4',
 'f7d776fd68b126f23b67070c4a034f08',
 '1a1c0a8d4bad82169f0594e65f756cf5',
 '46ed9dad0440c043d33646b0990bb4a',
 'd46b98f63a017578ea456f4bbbc96af9',
 '83827973c79ca7631c9ec1e03e401f54',
 '46955fddcc83a50f79b586547e543694',
 'e94e46bc5833f2f5e57b873e4f3ef3a4',
 '336122c3105440d193e42e2720468bf0',
 'ea33ad442b032208d778b73d04298f62',
 'bed29baf625ce9145b68309557f3a78c',
 '85a2511c375b5b32f72755048bac3f96',
 '3093367916fb5216823323ed0e090a6f',
 'bea77759a3e5f9037ae0031c221d81a4',
 '4f9f31db3c3873692a6f53dd95fd4468',
 '24b17537bce40695b3207096ecd79542',
 '1305b9266d38eb4d9f818dd0aa1a251',
 '83b41d719ea5af3f4dcd1df0d0a62a93',
 '9af98540f45411467246665d3d3724c',
 '4815b8a6406494662a96924bce6ef687',
 'ff1a44e1c1785d618bca309f2c51966a',
 'e9bd4ee553eb35c1d5ccc40b510e4bd',
 '73b8b6456221f4ea20d3c05c08e26f',
 'dcec634f18e12427c2c72e575af174cd',
 'fad118b32085f3f2c2c72e575af174cd',
 'b4ae56d6638d5338de671f28c83d2dcb',
 '44f9c4e1ea3532b8d7b20fded0142d7a',
 '3143a4accdc23349cac584186c95ce9b',
 'ca198dc3f7dc0cacec6338171298c66b',
 '8aed972ea2b4a9019c3814eae0b8d399',
 'd0a3fdd33c7e1eb040bc4e38b9ba163e',
 'c2e411ed6061a25ef06800d5696e457f',
 '604fcae9d93201d9d7f470ee20dce9e0',
 '1d18255a04d22794e521eeb8bb14c5b3',
 '586e67c53f181dc22adf8abaa25e0215',
 '43f94ba24d2f075c4d32a65fb7bf4ebc',
 '962883677a586bd84a60c1a189046dd1',
 '43e1cabc5dd2fa91fffc97a61124b1a9']
cup_like_mugs = ['9426e7aa67c83a4c3b51ab46b2f98f30',
 '214dbcace712e49de195a69ef7c885a4',
 '6e884701bfddd1f71e1138649f4c219',
 'b9004dcda66abf95b99d2a3bbaea842a']
mug = ['mug', '03797390', [square_handle_mugs, round_hanle_mugs,cup_like_mugs]]

############## ONLY TRAIN

############## ONLY TEST

######################################################################basket
square_baskets = ['2294583b27469851da06470a491566ee',
 '91b15dd98a6320afc26651d9d35b77ca',
 'c1abb91a5a9e2ea36d0c80f3bebb5ec0',
 'e9b6ef7375650b54ad2fb8cd0793fa9a',
 '9394df08489ae97eba5342d638d0c267',
 'd224635923b9ec4637dc91749a7c4915',
 '33a623a68ac3a6541c8c7b57a94dbb2e',
 'd9fb327b0e19a9ddc735651f0fb19093',
 '284c23b810fc7823466f97f37dccbde',
 '434194622dc684f81de9d8208aaa2d25',
 'bd11c268ebb14e2d7ae6f33544c233fe',
 'f870bedc47fdbd287ae6f33544c233fe',
 '5208bc4450a16d0e4b3c42e318f3affc',
 '3e88bbe0b7f7ab5a36b0f2a1430e993a',
 '242e6c21d53890a236b0f2a1430e993a',
 '80765e61d1305e5936b0f2a1430e993a',
 'b3b341fb9a2e406e36b0f2a1430e993a',
 '95385afe0390705a36b0f2a1430e993a',
 '4fcb0ea751f75df936b0f2a1430e993a',
 'a2038457c94b4a5a36b0f2a1430e993a',
 '5dbf477ba7765febb3a8888e78d004b3']
round_baskets = ['dc4a523b039e39bda843bb865a04c01a',
 '6d21c75bbe308322513c73fd461b230a',
 '9e2a1751b9129b1132fd0b3f17b0d6c',
 '7b407dedd933e3812f59b845f0db2ab3',
 '44ed0531eedc723dfaaa4370fa1ccec',
 '526a9f51fe93659c3c140c326fab0b5b',
 '9e4a936285f32194e1a03d0bf111d109',
 '35bc440973661b91259e0fe12d9ec13d']
basket = ['basket,handbasket', '02801938', [square_baskets, round_baskets]]

preselect = [bag, bottle, bowl, can, clock, jar, laptop, camera, mug, basket]
test_only_ids = [2801938]

csv_columns = ['synsetId', 'catId', 'name', 'ShapeNetModelId', 'objId']
dict_data = []
obj_id = 0
cat_id = 0
for cat_name, cat_synset_id, cat_objects in preselect:
    for objs in cat_objects:
        for obj in objs:
            row = {
                'synsetId': cat_synset_id,
                'catId' : cat_id,
                'name': cat_name,
                'ShapeNetModelId': obj,
                'objId': obj_id,
            }
            dict_data.append(row)
        obj_id += 1
    cat_id += 1

def write_to_csv(csv_file, dict_data, csv_columns):
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in dict_data:
                writer.writerow(data)
    except IOError:
        print("I/O error")



if not os.path.exists(args.csv_file_dir):
    os.mkdir(args.csv_file_dir)
csv_file = os.path.join(args.csv_file_dir, "preselect_table_top.csv")
write_to_csv(csv_file, dict_data, csv_columns)


# new_dict = dict()
# for row in dict_data:
#     obj_cat = int(row["synsetId"])
#     obj_id = row["ShapeNetModelId"]
#     obj_mesh_filename = os.path.join(args.shapenet_filepath,'0{}/{}/models/model_normalized.obj'.format(obj_cat, \
#                                                                                                    obj_id))
#     object_mesh = trimesh.load(obj_mesh_filename, force='mesh')
#     new_dict[(obj_cat, obj_id)] = object_mesh.bounds 
# with open(os.path.join(args.csv_file_dir, "object_bounds.pkl"), 'wb+') as handle:
#     pickle.dump(new_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

df_1 = pd.read_csv(csv_file)
for test_id in test_only_ids:
    df_1 = df_1[(df_1['synsetId'] != test_id)]
train_test_data = df_1.to_dict('records')

df_2 = pd.read_csv(csv_file)
for test_id in test_only_ids:
    df_2 = df_2[(df_2['synsetId'] == test_id)]
test_data = df_2.to_dict('records')

train,test = train_test_split(train_test_data, test_size=0.3)
train_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_train.csv")
test_csv_file_path = os.path.join(args.csv_file_dir, "preselect_table_top_test.csv")

write_to_csv(train_csv_file_path, train, csv_columns)
write_to_csv(test_csv_file_path, test+test_data, csv_columns)



# cat_ids = set()
# object_ids = set()
# cat_names = set()
# for idx in range(len(df)):
#     sample = df.iloc[idx]
#     cat_ids.add(sample['synsetId'])
#     cat_names.add(sample['name'])
#     object_ids.add(sample['objId']) 
# cat_ids = list(cat_ids)
# object_ids = list(object_ids)
# cat_names = list(cat_names)

# self.cat_ids = cat_ids
# self.cat_id_to_label = dict(zip(self.cat_ids, range(len(self.cat_ids))))
# self.label_to_cat_id = dict(zip(range(len(self.cat_ids)), self.cat_ids))

# self.object_ids = object_ids
# self.object_id_to_label = dict(zip(self.object_ids, range(len(self.object_ids))))
# self.object_label_to_id = dict(zip(range(len(self.object_ids)), self.object_ids))

# self.cat_names = cat_names
# self.cat_names_to_cat_id = dict(zip(self.cat_names, self.cat_ids))
# self.cat_id_to_cat_names = dict(zip(self.cat_ids, self.cat_names))


def check_too_many_faces(csv_fname, shapenet_dir):
    df = pd.read_csv(csv_fname)
    for idx in range(len(df)):
        row = df.iloc[idx]
        synset_category, shapenet_model_id = row['synsetId'], row['ShapeNetModelId']
        mesh_fname = os.path.join(
                shapenet_dir,
                '0{}/{}/models/model_normalized.obj'.format(synset_category, shapenet_model_id),
            )
        mesh = trimesh.load(mesh_fname, force='mesh')
        print(mesh.faces.shape[0])