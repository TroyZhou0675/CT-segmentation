import yaml

def get_yaml(file_yaml):
    rf = open(file=file_yaml, mode='r', encoding='utf-8')
    crf = rf.read()
    rf.close()  # 关闭文件
    yaml_data = yaml.load(stream=crf, Loader=yaml.FullLoader)
    return yaml_data