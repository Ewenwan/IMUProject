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
parser.add_argument('--start_length', type=int, default=-1)
parser.add_argument('--stride', type=float, default=0.67)
args = parser.parse_args()

exec_path = '../../cpp/cmake-build-relwithdebinfo/imu_localization/IMULocalization_cli'
model_path = '../../../models/svr_cascade0309'
# preset_list = ['mag_only', 'ori_only', 'full', 'raw']
preset_list = ['full']

root_dir = os.path.dirname(args.list)
data_list = []
placement_list = []
with open(args.list) as f:
    for line in f.readlines():
        if line[0] == '#':
            continue
        data_list.append(line.strip().split(','))

# Sanity check
classes = {'handheld', 'leg', 'bag', 'body'}
all_good = True
for data in data_list:
    data_path = root_dir + '/' + data[0]
    # if len(data) < 2:
    #     print(data_path + ' does not contain placement information.')
    #     all_good = False
    # if data[1] not in classes:
    #     print(data_path + ' has unknown placement: ' + data[1])
    #     all_good = False
    if not os.path.isdir(data_path):
        print(data_path + ' does not exist')
        all_good = False

assert all_good, 'Some datasets do not exist. Please fix the data list.'
print('Sanity check passed')

for data in data_list:
    data_path = root_dir + '/' + data[0]
    if not os.path.isdir(data_path):
        warnings.warn(data_path + ' does not exist. Skip.')
        continue

    for preset in preset_list:
        command = "%s %s --model_path %s --preset %s --start_portion_length %d" % (exec_path, data_path, model_path,
                                                                                preset, args.start_length)
        print(command)
        subprocess.call(command, shell=True)

    # Step counting
    # command = 'python3 ../step_counting/enhanced_step_counting.py %s --placement %s --stride %f --start_length %d'\
    #           % (data_path, data[1], args.stride, args.start_length)
    # print(command)
    # subprocess.call(command, shell=True)

    # command = 'python3 ../step_counting/frequency_step_counting.py %s --stride %f --start_length %d'\
    #           % (data_path, args.stride, args.start_length)
    # print(command)
    # subprocess.call(command, shell=True)
