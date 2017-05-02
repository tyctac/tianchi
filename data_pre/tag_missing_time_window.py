#encoding=utf8
import math
import json
from datetime import datetime,timedelta
from utils import config
import numpy
import os,sys
import xgboost
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pylab

## 使用的是原始的该路段的信息，比如A2.csv
##　该模块用来生成一个ｊｓｏｎ文件，里面是一个字典，ｋｅｙ为日期，值为一个字典（ｋｅｙ为时间窗口，值为ｂｏｏｌ值，表示这天的这个值在原始数据中有没有

route_set = ['B3','B1','A3','A2','C3','C1']
time_windowset = ['08-00-00',
    '08-20-00',
    '08-40-00',
    '09-00-00',
    '09-20-00',
    '09-40-00',
    '17-00-00',
    '17-20-00',
    '17-40-00',
    '18-00-00',
    '18-20-00',
    '18-40-00',
    ]
path = config.get_home_dir() + 'files/dataSets/training/task1/'


def generate_missing_value_dic(sourcefolder): ## sourcefile 由sourcefolder产生
    '''
    有些日期的某些时间窗口的值不存在,所以需要将其补齐,使用相邻星期的这天的值取均值,如果相邻的时间在假期或者特殊时间,则换
    一个相邻时间
    :param sourcefolder:
    :param sourcefile:
    :return:
    '''
    sourcefile = sourcefolder[:2] + '.csv'
    in_file_name = path + sourcefolder + sourcefile
    out_file_name = path + sourcefolder + 'exist_dic.json'
    fr = open(in_file_name, 'r')
    record = fr.readlines()
    new_records = []
    for rec in record:
        rearray = rec.split(',')
        rearray = rearray[2:]
        rearray[0] = rearray[0][2:]
        rearray[1] = rearray[1][:-2]
        rearray[-1] = rearray[-1].strip()
        new_records.append(rearray)

    # get date_dic_dic
    old_datestr = '2016-07-19'
    count = 0
    date_dic = {}
    date_dic_dic = {}
    date_array = [old_datestr]  ## 用于存放所有日期
    for rec in new_records:
        current_date = rec[0].split(' ')[0]
        current_time = rec[0].split(' ')[1].replace(':','-')
        if current_date == old_datestr:
            date_dic[current_time] = rec[-1]
            count += 1
        else:
            ## 上一轮日期结束
            date_array.append(current_date)
            date_dic_dic[old_datestr] = date_dic
            ##　上一轮日期结束
            count = 0
            old_datestr = current_date
            date_dic = {}
            date_dic[current_time] = rec[-1]
    date_dic_dic[old_datestr] = date_dic
    ## <--- get date_dic_dic
    ## <<--- write to new file
    # rec_head = 'B,3,"['
    rec_head = sourcefolder[0] + ',' + sourcefolder[1] + ',"['  ## 每个记录的共同前缀
    ret_array = []  ## to write in a file
    restr = ''

    ##　<<　ｇｅｎｅｒａｔｅ　ｎｅｗ  date_array
    date1 = datetime.strptime('2016-07-19', '%Y-%m-%d')
    date2 = datetime.strptime('2016-10-17', '%Y-%m-%d')
    d1 = date1
    date_array = []
    while d1 <= date2:  ## 注意 十月十号没有天气信息
        date_array.append(d1.strftime('%Y-%m-%d'))
        d1 = d1 + timedelta(days=1)
    ## -->> generate new date_array

    for d in date_array:
        if d == '2016-08-26' and sourcefolder == 'B1/':
            print 'i am here!! '
        if d not in date_dic_dic:
            tmp_dic = {}
            for tm in time_windowset:
                tmp_dic[tm] = 0
            date_dic_dic[d] = tmp_dic
        else:
            tmp_dic = date_dic_dic[d]
            for tm in time_windowset:
                if tm not in tmp_dic:
                    tmp_dic[tm] = 0
            date_dic_dic[d] = tmp_dic
    retstr = json.dumps(date_dic_dic)
    f = open(out_file_name, 'w')
    f.write(retstr)
    print 'ok!! '

def main():
    for rs in route_set:
        foder = rs + '/'
        print foder
        generate_missing_value_dic(foder)

if __name__ == '__main__':
    main()