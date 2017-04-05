from utils import config
import json
from datetime import datetime
def seperate_train_data_backup(sourcefile):
    '''
    source file is  :files/dataSets/training/task1/training_20min_avg_travel_time.csv
    :param sourcefile:
    :return:
    '''
    in_file_name = config.get_home_dir() + sourcefile
    hir = config.get_home_dir()
    fr = open(in_file_name,'r')
    fr.readline()
    time_data = fr.readlines()
    fr.close()
    print time_data[0]
    f1 = open(hir + 'files/dataSets/training/task1/A2/1.csv')
    f2 = open(hir + 'files/dataSets/training/task1/A2/2.csv')
    f3 = open(hir + 'files/dataSets/training/task1/A2/3.csv')
    f4 = open(hir + 'files/dataSets/training/task1/A2/4.csv')
    f5 = open(hir + 'files/dataSets/training/task1/A2/5.csv')
    f6 = open(hir + 'files/dataSets/training/task1/A2/6.csv')
    f7 = open(hir + 'files/dataSets/training/task1/A2/7.csv')
    f8 = open(hir + 'files/dataSets/training/task1/A2/8.csv')
    f9 = open(hir + 'files/dataSets/training/task1/A2/9.csv')
    f10 = open(hir + 'files/dataSets/training/task1/A2/10.csv')
    f11 = open(hir + 'files/dataSets/training/task1/A2/11.csv')
    f12 = open(hir + 'files/dataSets/training/task1/A2/12.csv')
    records = {}
    origin = []
    jstr = open(config.get_home_dir() + 'files/dataSets/training/weather_info_json.txt','r')
    weather_dic = json.loads(jstr) ## 天气对象
    eta = []
    for i in range(len(time_data)):
        tmp = []
        the_traj = time_data[i].replace('""','').split(',')
        intersection_id = the_traj[0]
        tollgate_id  = the_traj[1]
        start_time = the_traj[2]
        start_time = start_time[2:]
        print start_time
        tm = datetime.strptime(start_time,"%Y-%m-%d %H:%M:%S")
        weekday = tm.weekday()
        wds = [0,0,0,0,0,0,0]
        wds[weekday] = 1
        tmp.extend(wds)
        ## 天气值应该都在吧,按照日期应该没有缺失??
        datestr = tm.strftime("%Y-%m-%d")
        timestr = tm.strftime("%H-%M-%S")
        if datestr not in weather_dic.keys:
            print '*************************************errorRRRRRRRRRRRRRR!********************'
        else:
            tmpob = weather_dic[datestr]
            tmp.extend(tmpob[6])
            tmp.extend(tmpob[10])
            tmp.extend(tmpob[15])
            tmp.extend(tmpob[18])
        origin.append(tmp)
        eta.append(the_traj[-1])
    return