import os
import re
import random
from pathlib import Path
from collections import defaultdict
from itertools import product, combinations

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sn


class MotionDataLoader:
    def __init__(
        self,
        base_path,
        use_four_activities=False,
        window_length=2.56,
        slide_length=10,
        dgp_vae_ratio=0.4,
        dataset_comb_number=0,
    ):
        self.base_path = base_path
        self.folder_path = os.path.join(self.base_path, "A_DeviceMotion_data/")
        self.info_path = os.path.join(self.base_path, "data_subjects_info.csv")
        self.dataset_comb_number = dataset_comb_number
        self.window_length = window_length
        self.slide_length = slide_length
        self.use_four_activities = use_four_activities
        self.sampling_freq = 50
        self.num_features = 12  # attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
        self.slicer_val = int(self.window_length * self.sampling_freq)

        if not use_four_activities:
            self.tgt_labels = [
                "dws",
                "ups",
                "wlk",
                "jog",
                "sit",
                "std",
            ]  # dws, ups, wlk, jog, sit, std
        else:
            self.tgt_labels = ["dws", "ups", "wlk", "jog"]
        self.gender_labels = ["female", "male"]  # male/female

        self.trial_codes = {
            "dws": [1, 2, 11],
            "ups": [3, 4, 12],
            "wlk": [7, 8, 15],
            "jog": [9, 16],
            "sit": [5, 13],
            "std": [6, 14],
        }  # from folder

        gender_id = {
            1: [1, 2, 4, 6, 9, 11, 12, 13, 14, 15, 17, 20, 21, 22],
            0: [3, 5, 7, 8, 10, 16, 18, 19, 23, 24]
            }

        # the following options make the train ratio atleast 0.55
        male_combs = combinations(gender_id[1], 2)
        female_combs = combinations(gender_id[0], 2)

        all_test_combs = list(product(list(male_combs), list(female_combs)))
        self.selected_test_combs = sum(random.choice(all_test_combs), ())

    def get_ds_infos(self):
        ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
        dss = np.genfromtxt(self.info_path, delimiter=",")
        dss = dss[1:]
        print("----> Data subjects information is imported.")
        return dss
    
    def get_data_distribution(self):
        ds_list = self.get_ds_infos()

        self.data_distn = []

        for i, sub_id in tqdm(enumerate(ds_list[:, 0])):
            for j, activity in enumerate(self.tgt_labels):
                for trial in self.trial_codes[activity]:
                    fname = (
                        self.folder_path
                        + activity
                        + "_"
                        + str(trial)
                        + "/sub_"
                        + str(int(sub_id))
                        + ".csv"
                    )
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(["Unnamed: 0"], axis=1)
                    unlabel_data = raw_data.values


                    gender = int(ds_list[i, 4])  # label

                    num_pts = unlabel_data.shape[0]

                    self.data_distn.append(
                        {
                            "activity":activity,
                            "gender":gender,
                            "num_points":num_pts,
                        }
                    )
        
        return pd.DataFrame(self.data_distn)

                    

    def load_data(self):
        self.male_train_data_samples, self.male_test_data_samples = [], []
        self.male_train_tgt_labels, self.male_test_tgt_labels = [], []

        self.female_train_data_samples, self.female_test_data_samples = [], []
        self.female_train_tgt_labels, self.female_test_tgt_labels = [], []

        ds_list = self.get_ds_infos()

        self.sample_per_trial = {}

        for i, sub_id in tqdm(enumerate(ds_list[:, 0])):
            for j, activity in enumerate(self.tgt_labels):
                for trial in self.trial_codes[activity]:
                    fname = (
                        self.folder_path
                        + activity
                        + "_"
                        + str(trial)
                        + "/sub_"
                        + str(int(sub_id))
                        + ".csv"
                    )
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(["Unnamed: 0"], axis=1)
                    unlabel_data = raw_data.values

                    self.sample_per_trial[trial] = len(unlabel_data)

                    gender = int(ds_list[i, 4])  # label

                    num_iters = unlabel_data.shape[0] // self.slicer_val

                    for slice_idx in range(
                        0, unlabel_data.shape[0] - self.slicer_val, self.slide_length
                    ):
                        sliced_data = np.float32(
                            unlabel_data[slice_idx : slice_idx + self.slicer_val]
                        )
                        if sliced_data.shape[0] != self.slicer_val:
                            continue

                        ## We consider long trials as training dataset and short trials as test dataset
                        if sub_id not in self.selected_test_combs:
                            if gender == 0:
                                self.female_train_data_samples.append(sliced_data)
                                self.female_train_tgt_labels.append(activity)
                            else:
                                self.male_train_data_samples.append(sliced_data)
                                self.male_train_tgt_labels.append(activity)
                        else:
                            if gender == 0:
                                self.female_test_data_samples.append(sliced_data)
                                self.female_test_tgt_labels.append(activity)
                            else:
                                self.male_test_data_samples.append(sliced_data)
                                self.male_test_tgt_labels.append(activity)

        self.unique_tgt_labels = np.unique(
            self.female_train_tgt_labels
            + self.female_test_tgt_labels
            + self.male_train_tgt_labels
            + self.male_test_tgt_labels
        )

        self.le_tgt = preprocessing.LabelEncoder()
        self.le_tgt.fit(
            self.female_train_tgt_labels
            + self.female_test_tgt_labels
            + self.male_train_tgt_labels
            + self.male_test_tgt_labels
        )
        self.male_train_tgt_labels = self.le_tgt.transform(self.male_train_tgt_labels)
        self.male_test_tgt_labels = self.le_tgt.transform(self.male_test_tgt_labels)
        self.female_train_tgt_labels = self.le_tgt.transform(
            self.female_train_tgt_labels
        )
        self.female_test_tgt_labels = self.le_tgt.transform(self.female_test_tgt_labels)

    def train_test_val_split(self):
        # Using only training data for vae, all of it (90-10 train/val split)
        # using the entire training data for downstream train and entire test data for downstream test

        # for downstream
        male_down_all_data = [
            self.male_train_data_samples,
            self.male_train_tgt_labels,
        ]

        male_train_samples_count = len(self.male_train_data_samples)
        self.male_down_x_train = self.male_train_data_samples[male_train_samples_count//10:]
        self.male_down_x_val = self.male_train_data_samples[:male_train_samples_count//10]
        self.male_down_y_train = self.male_train_tgt_labels[male_train_samples_count//10:]
        self.male_down_y_val = self.male_train_tgt_labels[:male_train_samples_count//10]

        (self.male_down_x_test, self.male_down_y_test,) = (
            self.male_test_data_samples,
            self.male_test_tgt_labels,
        )

        # Using only training data for vae, all of it (90-10 train/val split)
        # using the entire training data for downstream train and entire test data for downstream test

        # for downstream
        female_down_all_data = [
            self.female_train_data_samples,
            self.female_train_tgt_labels,
        ]

        female_train_samples_count = len(self.female_train_data_samples)
        self.female_down_x_train = self.female_train_data_samples[female_train_samples_count//10:]
        self.female_down_x_val = self.female_train_data_samples[:female_train_samples_count//10]
        self.female_down_y_train = self.female_train_tgt_labels[female_train_samples_count//10:]
        self.female_down_y_val = self.female_train_tgt_labels[:female_train_samples_count//10]

        (self.female_down_x_test, self.female_down_y_test,) = (
            self.female_test_data_samples,
            self.female_test_tgt_labels,
        )

        self.dgp_x_train = np.array(self.male_down_x_train + self.female_down_x_train)
        self.dgp_y_train = np.concatenate(
            [self.male_down_y_train, self.female_down_y_train]
        )
        self.dgp_gender_train = np.concatenate(
            [
                [np.float32(1.0)] * len(self.male_down_y_train),
                [np.float32(0.0)] * len(self.female_down_y_train),
            ]
        )

        self.dgp_x_val = np.array(self.male_down_x_val + self.female_down_x_val)
        self.dgp_y_val = np.concatenate([self.male_down_y_val, self.female_down_y_val])
        self.dgp_gender_val = np.concatenate(
            [
                [np.float32(1.0)] * len(self.male_down_y_val),
                [np.float32(0.0)] * len(self.female_down_y_val),
            ]
        )

        shuffle_idx = list(range(len(self.dgp_x_train)))
        random.shuffle(shuffle_idx)
        self.dgp_x_train = self.dgp_x_train[shuffle_idx]
        self.dgp_y_train = self.dgp_y_train[shuffle_idx]
        self.dgp_gender_train = self.dgp_gender_train[shuffle_idx]
