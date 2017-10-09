import sys
import os
import subprocess
import argparse

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')

parser = argparse.ArgumentParser()
parser.add_argument('--recompute', action='store_true')
args = parser.parse_args()

exec_path = '../../cpp/cmake-build-relwithdebinfo/imu_localization/IMULocalization_cli'
model_path = '../../../models/svr_cascade1004_c1e001'
data_root1 = '../../../data2/'
data_root2 = '../../../data/phab_body/'
data_list1 = ['hang_handheld_normal2', 'hang_handheld_speed2', 'hang_handheld_side2', 'hang_leg_normal2', 'hang_leg_speed2', 'hang_leg_side2',
              'hang_bag_normal2', 'hang_bag_speed2', 'hang_bag_side2', 'chen_handheld2', 'chen_body2', 'chen_bag2', 'yajie_handheld1', 'yajie_leg1',
              'yajie_bag1', 'yajie_body1', 'huayi_handheld2', 'huayi_leg2']
data_list2 = ['lopata2', 'library1', 'cse1', 'huayi_cse1', 'huayi_cse3']

weight_list = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
suffix_list = ['001', '01', '1', '10', '100', '1000', '10000']
# sanity check
all_good = True
for data in data_list1:
    if not os.path.isdir(data_root1 + data):
        all_good = False
        print(data_root1 + data + " doesn't exist")
for data in data_list2:
    if not os.path.isdir(data_root2 + data):
        all_good = False
        print(data_root2 + data + " doesn't exist")

if not all_good:
    print('Please fix the list before proceeding.')
else:
    print('Sanity check passed')

for i in range(len(weight_list)):
    for data in data_list1:
        dir_path = data_root1 + data + '/result_{}'.format(suffix_list[i])
        if not os.path.isdir(dir_path) or args.recompute:
            command = "%s %s --model_path %s --weight_ls %f --suffix %s" % (exec_path, data_root1 + data, model_path, weight_list[i], suffix_list[i])
            subprocess.call(command, shell=True)
    for data in data_list2:
        dir_path = data_root2 + data + '/result_{}'.format(suffix_list[i])
        if not os.path.isdir(dir_path) or args.recompute:
            command = "%s %s --model_path %s --weight_ls %f --suffix %s" % (exec_path, data_root2 + data, model_path, weight_list[i], suffix_list[i])
            subprocess.call(command, shell=True)

