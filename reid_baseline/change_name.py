'''
@author:  wubochuan
@contact: 1670306646@qq.com
change dataset name to dataloader
'''

from shutil import copyfile
import os

dir_path = './dataset/bounding_box_train'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
file_path ='./dataset/train_list.txt'

for line in open(file_path):
    result = line.split()
    path = result[0]
    label =result[1]
    name = path.split('/')[1]
    copyfile(path,dir_path+'/'+label+'_'+name)

print('finished!')

