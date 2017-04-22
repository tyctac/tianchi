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
path = config.get_home_dir() + '/files/dataSets/training/task1/'

def complete_train_data(sourcefolder,sourcefile):
    '''
    有些日期的某些时间窗口的值不存在,所以需要将其补齐,使用相邻星期的这天的值取均值,如果相邻的时间在假期或者特殊时间,则换
    一个相邻时间
    :param sourcefolder:
    :param sourcefile:
    :return:
    '''
    in_file_name = path + sourcefolder + sourcefile
    out_file_name = path + sourcefolder + 'sourcefile_complete.csv'
    fr = open(in_file_name,'r')
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
    date_array = [old_datestr] ## 用于存放所有日期
    for rec in new_records:
        current_date = rec[0].split(' ')[0]
        current_time = rec[0].split(' ')[1]
        if current_date == old_datestr:
            date_dic[current_time] = rec[-1]
            count += 1
        else :
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
    rec_head = sourcefolder[0] + ','+sourcefolder[1] + ',"['  ## 每个记录的共同前缀
    ret_array = [] ## to write in a file
    restr = ''
    for d in date_array:
        date_dic = date_dic_dic[d]
        for tm in time_windowset:
            tmp_rec = rec_head
            tm = tm.replace('-',':')
            timestr = d + ' ' + tm
            the_time = datetime.strptime(timestr,"%Y-%m-%d %H:%M:%S") ## 字符串转为时间
            window_end_time = the_time + timedelta(minutes=20)
            if tm in date_dic.keys():
                eta = date_dic[tm]
            else :

                before_time = the_time
                before_date = before_time.date().strftime("%Y-%m-%d")
                be_eta = -2
                while before_date not in date_dic_dic or tm not in date_dic_dic[before_date]:
                    before_time = before_time + timedelta(days=-7)
                    before_date = before_time.date().strftime("%Y-%m-%d")
                    if before_time < datetime.strptime('2016-07-19','%Y-%m-%d'):
                        be_eta = -1
                        break
                if be_eta == -2:
                    be_eta = date_dic_dic[before_date][tm]

                af_eta = -2
                after_time = the_time
                after_date = after_time.date().strftime("%Y-%m-%d")
                while after_date not in date_dic_dic or tm not in date_dic_dic[after_date]:
                    after_time = after_time + timedelta(days=7)
                    after_date = after_time.date().strftime("%Y-%m-%d")
                    if after_time > datetime.strptime('2016-10-17','%Y-%m-%d'):
                        af_eta = -1
                        break
                if af_eta == -2:
                    af_eta = date_dic_dic[after_date][tm]

                if be_eta == -1 and af_eta ==-1:
                    print 'error'
                    return
                elif be_eta == -1 :
                    eta = af_eta
                elif af_eta == -1:
                    eta = be_eta
                else:
                    eta = ((float)(be_eta) + (float)(af_eta))/2.0
            tmp_rec += the_time.strftime('%Y-%m-%d %H:%M:%S') ## 时间转字符串
            tmp_rec += ","
            tmp_rec += window_end_time.strftime('%Y-%m-%d %H:%M:%S')
            tmp_rec += ')",'
            tmp_rec += str(eta)
            # tmp_rec += '\n'
            print tmp_rec
            restr += tmp_rec + '\n'
    f = open(out_file_name,'w')
    f.write(restr)
    print len(date_array)


    ## <---- write to new file


if __name__ == '__main__':
    complete_train_data('C3/','C3.csv')
