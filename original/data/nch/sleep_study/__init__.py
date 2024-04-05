import os

from . import info
from . import data
from . import dataset
from . import raw

_data_root = os.getenv(
    "DLHPROJ_DATA_ROOT",
    '/mnt/e/data/physionet.org'
)
data_dir = f'{_data_root}/files/nch-sleep/3.1.0'


def __init__():
    data.study_list = data.init_study_list()
    age_fn = data.init_age_file()
    info.SLEEP_STUDY = info.load_health_info(info.SLEEP_STUDY, False)
