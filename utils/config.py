#encoding=utf8
import yaml
import sys,os

def get_home_dir():
    # f = open('../utils/config.yaml')
    # f = open('../utils/config.yaml')
    # f = open('/home/zw/Documents/project/tianchi/utils/config.yaml')
    # x = yaml.load(f)
    # f.close()
    # return x['HOME_DIR']
    return os.path.dirname(os.getcwd())

def get_title_weight():
    f = open('/home/zw/Documents/project/tianchi/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['TITLE_WEIGHT']

def main():
    x =  get_home_dir()
    print x


if __name__ == '__main__':
    main()