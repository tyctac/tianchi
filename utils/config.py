#encoding=utf8
import yaml
import sys,os

def get_home_dir():
    return os.path.dirname(os.getcwd())

def get_title_weight():
    hir = get_home_dir()
    f = open(hir + '/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['TITLE_WEIGHT']

def get_weather_weight():
    f = open(get_home_dir() + '/utils/config.yaml')
    x = yaml.load(f)
    f.close()
    return x['weather_weight']

def main():
    x =  get_weather_weight()
    print x


if __name__ == '__main__':
    main()