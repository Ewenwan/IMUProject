import sys
import os
import subprocess
import argparse
import warnings

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')


parser = argparse.ArgumentParser()
parser.add_argument('list', type=str)
parser.add_argument('--recompute', action='store_true')
args = parser.parse_args()

exec_path = '../../cpp/cmake-build-relwithdebinfo/imu_localization/IMULocalization_cli'
model_path = '../../../models/svr_cascade0309'
preset_list = ['mag_only', 'ori_only', 'full', 'raw']
# preset_list = ['full']

root_dir = os.path.dirname(args.list)
data_list = []
with open(args.list) as f:
    for line in f.readlines():
        if line[0] == '#':
            continue
        info = line.split(',')
        if len(info) > 0:
            data_list.append(info[0].strip('\n'))

# Sanity check
all_good = True
for data in data_list:
    data_path = root_dir + '/' + data
    if not os.path.isdir(data_path):
        print(data_path + 'does not exist')
        all_good = False

assert all_good, 'Some datasets do not exist. Please fix the data list.'
print('Sanity check passed')

for data in data_list:
    data_path = root_dir + '/' + data
    if not os.path.isdir(data_path):
        warnings.warn(data_path + ' does not exist. Skip.')
        continue
    for preset in preset_list:
        command = "%s %s --model_path %s --preset %s" % (exec_path, data_path, model_path, preset)
        print(command)
        subprocess.call(command, shell=True)
    # Step counting
    # command = 'python3 ../speed_regression/step_counting.py %s' % data_path
    # print(command)
    # subprocess.call(command, shell=True)
