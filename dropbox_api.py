import dropbox
from light_tests import ContrastTest, BrightnessTest, ColorBalanceTest, GammaTest

def _upload(path):
    with open(path, 'rb') as f:
        data = f.read()
    try:
        dbx.files_upload(data, '/'+path)
    except:
        pass


def upload_model(net_name, jitter, dataset):
    _upload(f'models/{dataset}/{net_name}_{jitter}.pth')

def upload_run(net_name, jitter, dataset):
    _upload(f'runs/{dataset}/{net_name}_{jitter}.txt')

def upload_result(net_name, jitter, dataset, transform):
    _upload(f'results/{dataset}/{net_name}_{jitter}_{transform}.txt')


def upload_all(net_name, jitter, dataset):
    upload_model(net_name, jitter, dataset)
    upload_run(net_name, jitter, dataset)
    for Test in [ContrastTest, BrightnessTest, ColorBalanceTest, GammaTest]:
        upload_result(net_name, jitter, dataset, Test.name)

