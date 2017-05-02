#encoding=utf8
import math
import json
from datetime import datetime,timedelta
from utils import config,often_use
import numpy
import os,sys
import xgboost
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pylab
import add_deficiency_value

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
folder_set = ['A2','A3','B1','B3','C1','C3']
path = config.get_home_dir() + '/files/dataSets/training/task1/'

def normalize_weather_info():
    '''
    归一化天气信息,并对相应的天气赋相应的权值
    :return: store to files
    '''
    jstr = open(config.get_home_dir() + 'files/dataSets/training/weather_info_json.txt', 'r').read()
    out_file = 'files/dataSets/training/weather_info_updated_json.txt','r'
    old_weather_dic = json.loads(jstr)
    for dat  in old_weather_dic.keys():
        tmp_info = old_weather_dic[dat]


def seperate_train_data(sourcefolder,sourcefile): ## sourcefolder : A2, sourcefile:sourcefile_complete.csv
    '''
    source file is  :files/dataSets/training/task1/training_20min_avg_travel_time.csv
    :param sourcefile:
    :return:
    '''
    sourcefile = 'sourcefile_complete.csv'
    in_file_name = path + sourcefolder + sourcefile
    fr = open(in_file_name,'r')
    # fr.readline()
    time_data = fr.readlines()
    fr.close()
    print time_data[0]
    records = {}
    origin = []
    # 时间字符串与对应的存储对象的字典
    store_dic = {}

    jstr = open(config.get_home_dir() + 'files/dataSets/training/weather_info_json.txt','r').read()
    file_tmp_path = path + sourcefolder
    weather_dic = json.loads(jstr) ## 天气对象
    eta = []
    valid_re = 1
    date_set = []
    slices = 1
    old_datestr = '2016-07-19'
    for i in range(len(time_data)):
        # print valid_re
        # valid_re += 1
        tmp = []
        the_traj = time_data[i].replace('"','').split(',')
        intersection_id = the_traj[0]
        tollgate_id  = the_traj[1]
        start_time = the_traj[2]
        start_time = start_time[1:]
        tm = datetime.strptime(start_time,"%Y-%m-%d %H:%M:%S") ##　将字符串变为时间
        weekday = tm.weekday()
        wds = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        wds[weekday] = 1.0
        tmp.extend(wds)
        ## 天气值应该都在吧,按照日期应该没有缺失??
        datestr = tm.strftime("%Y-%m-%d")
        timestr = tm.strftime("%H-%M-%S")
        if timestr == '08-20-00':
            print datestr
        if datestr not in weather_dic.keys():
            print '*************************************errorRRRRRRRRRRRRRR!********************'
            print datestr
            continue
        else:
            tmpob = weather_dic[datestr]
            tmp.extend([float(w) for w in tmpob['6']])
            tmp.extend([float(w) for w in tmpob['9']])
            tmp.extend([float(w) for w in tmpob['15']])
            tmp.extend([float(w) for w in tmpob['18']])
        if datestr == old_datestr:
            slices +=1
        else:
            if slices != 12:
                print old_datestr,' slices :',slices
            slices = 1
            old_datestr = datestr
        if timestr in store_dic.keys():
            time_window_attrs = store_dic[timestr]['attrs']
            time_window_etas = store_dic[timestr]['etas']
            time_window_attrs.append(tmp)
            time_window_etas.append(float(the_traj[-1]))
        else:
            time_window_this_record = {}
            attrs = []
            attrs.append(tmp)
            etas = []
            etas.append(float(the_traj[-1]))
            time_window_this_record['attrs'] = attrs
            time_window_this_record['etas'] = etas
            store_dic[timestr] = time_window_this_record
    for k in store_dic.keys():
        # print k
        feature = file_tmp_path + k + '_features'
        label = file_tmp_path + k + '_labels'
        ft = store_dic[k]['attrs']
        ea = store_dic[k]['etas']
        npary = numpy.array(ft,dtype = float)
        numpy.save(feature,npary)
        npary = numpy.array(ea,dtype = float)
        # print len(npary)
        numpy.save(label,npary)
    f = open('date_set.txt','w')
    f.write('\n'.join(date_set))
    return

def store_origin_label(sourcefolder): ## sourcefolder : A2, sourcefile:sourcefile_complete.csv
    '''
    source file is  :files/dataSets/training/task1/training_20min_avg_travel_time.csv
    :param sourcefile:
    :return:
    '''
    sourcefile = 'exist_dic.json'
    f = open(path+sourcefolder + sourcefile,'r')
    print type(f)
    tpstr = f.read()
    date_dic_dic = json.loads(tpstr)
    jstr = open(config.get_home_dir() + 'files/dataSets/training/weather_info_json.txt','r').read()
    file_tmp_path = path + sourcefolder
    weather_dic = json.loads(jstr) ## 天气对象
    date1 = datetime.strptime('2016-07-19', '%Y-%m-%d')
    date2 = datetime.strptime('2016-10-17', '%Y-%m-%d')
    d1 = date1
    dates = []
    real_value_array = []
    tm_window_dic = {}
    for tm in time_windowset:
        tm_window_dic[tm] = []
    while d1 <= date2:
        if d1.month == 10 and d1.day == 10:  ##　TODO 仍然把10月十号去掉了
            d1 = d1 + timedelta(days=1)
            continue
        dates.append(d1.date())
        for tm in time_windowset:
            dstr = d1.strftime('%Y-%m-%d')
            x = date_dic_dic[dstr][tm]
            tm_window_dic[tm].append(date_dic_dic[datetime.strftime(d1,'%Y-%m-%d')][tm])
        d1 = d1 + timedelta(days=1)

    out_filename = path + sourcefolder + 'tm_window_dic.json'
    tm_dic_str = json.dumps(tm_window_dic)
    f2 = open(out_filename,'w')
    f2.write(tm_dic_str)

def predict(sourcefolder):
    jstr = open(config.get_home_dir() + 'files/dataSets/training/weather_info_predict_json.txt', 'r').read()
    file_tmp_path = path + sourcefolder
    out_file_path = path + sourcefolder + 'result.csv'
    weather_dic = json.loads(jstr)  ## 天气对象
    date1 = datetime.strptime('2016-10-18','%Y-%m-%d')
    date2 = datetime.strptime('2016-10-25','%Y-%m-%d')
    d1 = date1
    tollgate_id = sourcefolder[0]
    intersection_id = sourcefolder[1]
    restr = ''
    while d1 < date2:
        ## 如果这天天气不存在
        pass
        time_date = datetime.strftime(d1,"%Y-%m-%d")
        for twindow in time_windowset:
            ## 生成收费站 和路口id
            tmpstr = str(tollgate_id) + ','
            tmpstr += str(intersection_id) + ','
            hms = twindow.split('-')
            hour = int(hms[0])
            minute = int(hms[1])
            second = int(hms[1])
            window_start = time_date + ' ' + hms[0] + ':' + hms[1] + ':' + hms[2]
            window_start_time = datetime.strptime(window_start,'%Y-%m-%d %H:%M:%S')
            window_end_time = window_start_time + timedelta(minutes=20)
            window_end = datetime.strftime(window_end_time,'%Y-%m-%d %H:%M:%S')
            ## 生成时间窗口字符串
            tmpstr += '"[' + window_start +',' + window_end+')",'
            ## 从文件中载入模型
            modelfile = path + sourcefolder + twindow
            bst = xgboost.Booster(model_file=modelfile)
            ##　生成预测特征向量
            feature_vector = []
            wds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            wds[d1.weekday()] = 1.0
            feature_vector.extend(wds)
            tmpob = weather_dic[time_date]
            feature_vector.extend([float(w) for w in tmpob['6']])
            feature_vector.extend([float(w) for w in tmpob['9']])
            feature_vector.extend([float(w) for w in tmpob['15']])
            feature_vector.extend([float(w) for w in tmpob['18']])
            ## predict
            dtest = xgboost.DMatrix(feature_vector)
            eta = bst.predict(dtest)
            tmpstr += str(eta[0]) + '\n'
            restr += tmpstr ## not neccessary
        d1 = d1 + timedelta(days=1)

    f = open(out_file_path,'w')
    f.write(restr)





def store_train_model(source_folder):
    param = {'bst:max_depth': 2, 'bst:eta': 1, 'silent': 1, 'objective': 'reg:linear'}
    param['nthread'] = 4
    plst = param.items()
    plst += [('eval_metric', 'auc')]  # Multiple evals can be handled in this way
    plst += [('eval_metric', 'ams@0')]
    train_file_dic = get_train_matrix_dic('A2/')
    num_round = 10
    for tm in train_file_dic.keys():
        timestr = tm[:8]
        feature_name = path + source_folder + tm
        label_name = path + source_folder + train_file_dic[tm]
        tmpfeature = numpy.load(feature_name)
        print tmpfeature.shape
        tmplabel = numpy.load(label_name)
        print tmplabel.shape
        dtrain = xgboost.DMatrix(tmpfeature,label = tmplabel)
        bst = xgboost.train(plst,dtrain,num_round)
        bst.save_model(path+source_folder+timestr)


def get_train_matrix_dic(source_folder):
    dir = path + source_folder
    file_list = os.listdir(dir)
    ret = {}
    for f in file_list:
        if 'labels.npy' in f:
            ret[f[:8] + '_features.npy'] = f
    return ret


def get_weather_info(fname): ## predict weather file : /home/zw/Documents/project/tianchi/files/dataSets/testing_phase1/weather (table 7)_test1.csv
    f = open(fname,'r')
    head = f.readline()
    head = head.strip().replace('"','').split(',')
    retdic = {}
    hir = config.get_home_dir()
    weather_info = f.readlines()
    weather_info = [ w.strip().replace('"','').split(',') for w in weather_info]
    for wi in weather_info:
        if wi[0] in retdic.keys():
            tmpdic = retdic[wi[0]]
            if wi[1] not in tmpdic.keys():
                tmpdic[wi[1]] = wi[2:]
        else:
            tmpdic = {}
            tmpdic[wi[1]] = wi[2:]
            retdic[wi[0]] = tmpdic
    print '1'
    retstr = json.dumps(retdic)
    f = open(hir + 'files/dataSets/training/weather_info_predict_json.txt','w')
    f.write(retstr)

def plot_origin(source_folder):
    '''
    同plot_data_neighbor_average方法，不过是画出原始与与neighbor_value的对比，
    :param source_folder:
    :return:
    '''
    train_dates = often_use.get_train_date_array()
    tm_wd_dic = json.loads(open(path+source_folder+'tm_window_dic.json','r').read())
    ys_set = []
    for tm in time_windowset:
        filename = tm + '_labels.npy'
        npy_path = path + source_folder + filename
        ys_set.append(list(numpy.load(npy_path)))
    ys_set_ori = []
    for tm in time_windowset:
        ys_set_ori.append(tm_wd_dic[tm])
    for i in range(len(time_windowset)):
        #     pylab.plot(dates,ys_set[i])
        print train_dates
        pylab.plot(train_dates, ys_set[i])
        pylab.plot(train_dates,ys_set_ori[i])
        pylab.xlabel('dates')
        pylab.ylabel('eta')
        pylab.title('eta of ' + time_windowset[i] + ' of ' + source_folder[:2])
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # ys = range(len(dates))
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # # Plot
        # plt.plot(dates, ys)
        # plt.gcf().autofmt_xdate()  # -fill-missing-with-neighbor-average :figure name
        plt.grid(True)
        plt.show()

def plot_data_neighbor_average(source_folder):
    '''
    画出七月十九到十月十七之间的摸个时间段的曲线变化图
    :param source_folder:
    :return:
    '''
    date1 = datetime.strptime('2016-07-19', '%Y-%m-%d')
    date2 = datetime.strptime('2016-10-17', '%Y-%m-%d')
    d1 = date1
    dates = []
    while d1 <= date2:
        if d1.month == 10 and d1.day == 10:  ##　之前因为
            d1 = d1 + timedelta(days=1)
            continue
        dates.append(d1.date())
        d1 = d1 + timedelta(days=1)

    # datearray = add_deficiency_value.complete_train_data('A3/','A3_backup.csv')
    # print 'datearraylen: ',len(datearray)
    i=1
    # for d in dates:
    #     print i
    #     i += 1
    #     print d
    print 'datelen: ',len(dates)
    ## set 12 timewindowset -label
    ys_set = []
    for tm in time_windowset:
        filename = tm+'_labels.npy'
        npy_path = path + source_folder + filename
        ys_set.append(list(numpy.load(npy_path)))
    for i in range(len(time_windowset)):
    #     pylab.plot(dates,ys_set[i])
        print dates
        pylab.plot(dates,ys_set[i])
        pylab.xlabel('dates')
        pylab.ylabel('eta')
        pylab.title('eta of ' + time_windowset[i] + ' of ' + source_folder[:2])
        plt.gcf().autofmt_xdate()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # ys = range(len(dates))
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        # # Plot
        # plt.plot(dates, ys)
        # plt.gcf().autofmt_xdate()  # -fill-missing-with-neighbor-average :figure name
        plt.grid(True)
        plt.show()

def main():
    '''
    步骤：
    １. store_train_model(sourcefolder)
    2. predict(sourcefolder)
    attention weather : train or test is differentf
    :return:
    '''
    for fder in folder_set:
        sourcefolder = fder + '/'
        store_train_model(sourcefolder)
        predict(sourcefolder) ## 开始预测
    # seperate_train_data(sourcefolder,sourcefile)
    # weather_file = config.get_home_dir() + 'files/dataSets/testing_phase1/weather (table 7)_test1.csv'
    # get_weather_info(weather_file)

def back_main():
    dates = ['01/02/1991', '01/03/1991', '01/04/1991']
    xs = [datetime.strptime(d, '%m/%d/%Y').date() for d in dates]
    ys = range(len(xs))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    # Plot
    plt.plot(xs, ys)
    plt.gcf().autofmt_xdate()  #
    plt.show()



if __name__ == '__main__':
    # main()
    # get date array
    # plot_data('C3/')
    # seperate_train_data('C3/','sourcefile_complete.csv')
    # plot_data('A3/')
    # store_origin_label('A2/')  ## wait a minute
    plot_origin('C3/')