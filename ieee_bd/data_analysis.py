import os
import glob
import pandas as pd
import argparse
from datetime import datetime
import numpy as np


def extract_info(region, root, stage='training'):
    if region in ['R1', 'R2', 'R3', 'R7', 'R8']:
        track = 'ieee-bd-core'
    else:
        track = 'ieee-bd-transfer-learning'

    save_path = os.path.join(root, track, region)
    info_path = os.path.join(save_path, f"{region}_{stage}_info.csv")
    blacklist_path = os.path.join(save_path, f"{region}_{stage}_blacklist.csv")

    data_path = os.path.join(root, track, region, stage)
    # print(f"{os.path.exists(data_path)}  {data_path} ")
    times_per_day = [str(item.time()).replace(':', '') for item in
                         pd.date_range('00:00:00', periods=4 * 24, freq="0h15min")]
    black_list = []
    useful_list = []
    days = os.listdir(data_path)
    days.sort()
    for day in days:
        # print(f"working on {day}")
        for time_code in times_per_day:
            time_filter = f"/{day}/**/*{time_code}Z.nc"
            files = glob.glob(data_path + time_filter)
            date_time = datetime.strptime(f"{day} {time_code}", "%Y%j %H%M%S")
            number = int(date_time.timestamp() // (60 * 15))  # for order of samples
            date = date_time.date().__str__().replace('-', '')
            temp_dict = {'day': day, 'time': time_code, 'date': date, 'number_stamp': number}
            if len(files) < 5:
                black_list.append(temp_dict)
                print(f"\tblacklist at {day}:::{time_code}")
            else:
                useful_list.append(temp_dict)
        # print(f"{day} completed!")
    print(f"{region}|{stage} All Done !!!")

    useful = pd.DataFrame.from_dict(useful_list)
    blacklist = pd.DataFrame.from_dict(black_list)
    useful['used'] = useful.apply(
        lambda row: (row['number_stamp'] + 35) == useful.iloc[(row.name + 35) % useful.shape[0]]['number_stamp'],
        axis=1)
    useful.to_csv(info_path, index=False)
    blacklist.to_csv(blacklist_path, index=False)
    return useful, blacklist


def extract_test(region, root, stage='test'):
    if region in ['R1', 'R2', 'R3', 'R7', 'R8']:
        track = 'ieee-bd-core'
    else:
        track = 'ieee-bd-transfer-learning'

    # stage = 'test'
    if stage == 'heldout':
        # stage = 'test'
        track = 'heldout'

    save_path = os.path.join(root, track, region)
    info_path = os.path.join(save_path, f"{region}_{stage}_info.csv")
    # blacklist_path = os.path.join(save_path, f"{region_id}_{stage}_blacklist.csv")

    data_path = os.path.join(root, track, region, 'test')  # stage)
    # print(f"{os.path.exists(data_path)}  {data_path} ")
    # black_list = []
    useful_list = []
    days = os.listdir(data_path)
    days.sort()
    for day in days:
        # print(f"working on {day}")
        day_filter = f"/{day}/CRR/*Z.nc"
        files = glob.glob(data_path + day_filter)
        time_codes = [x.split('_')[-1].split('Z.')[0] for x in files]
        time_codes.sort(key=lambda x: x.split('_')[-1].split('Z.')[0])
        # useful_list.extend({})
        for time_code in time_codes:
            date_time = datetime.strptime(time_code, "%Y%m%dT%H%M%S")
            number = int(date_time.timestamp() // (60 * 15))  # for order of samples
            date = date_time.date().__str__().replace('-', '')
            temp_dict = {'day': day, 'time': time_code.split('T')[-1], 'date': date, 'number_stamp': number}
            useful_list.append(temp_dict)
        # print(f"{day} completed!")
    # print(f"{region}|{stage} All Done !!!")

    useful = pd.DataFrame.from_dict(useful_list)
    # blacklist = pd.DataFrame.from_dict(black_list)
    useful['used'] = useful.apply(
        lambda row: (row['number_stamp'] + 3) == useful.iloc[(row.name + 3) % useful.shape[0]]['number_stamp'],
        axis=1)
    useful.to_csv(info_path, index=False)
    return useful


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str,
                        default='/home/ai4ex2021/Datasets/Weather4Cast2021')
    args = parser.parse_args()
    root = args.root
    times_per_day = [str(item.time()).replace(':', '') for item in
                         pd.date_range('00:00:00', periods=4 * 24, freq="0h15min")]

    for region_num in range(1, 12):
        region_id = f"R{region_num}"
        for stage in ['test']:  # ['training', 'validation', 'test']:
            try:
                print(f"Working on {region_id}||{stage}")
                if stage == 'test':
                    useful = extract_test(region_id, root)
                else:
                    useful, blacklist = extract_info(region_id, root, stage)
            except:
                print(f"{region_id} is a test set")
