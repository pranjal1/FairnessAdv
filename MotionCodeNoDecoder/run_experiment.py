import os
import yaml
import pickle
import random

import numpy as np

from src.train_female_together import run_train as run_train_female
from src.train_male_together import run_train as run_train_male
from src.data.data_loader_id import MotionDataLoader
# from src.data.data_loader_id_ratio import MotionDataLoader


def run_experiment(train_conf_path: str):
    train_conf = yaml.safe_load(open(train_conf_path))
    seed = train_conf.get("base_config").get("seed")
    np.random.seed(seed)
    random.seed(seed)

    if train_conf.get("train_for_male", True):
        run_train = run_train_male
    else:
        run_train = run_train_female

    for run_idx in range(20):
        print("Run idx = ", run_idx)
        seed = train_conf.get("base_config").get("seed") #used for train/val selection and model params init
        DL = MotionDataLoader(
            base_path=train_conf.get("base_config").get("data_dir"),
            use_four_activities=train_conf.get("base_config").get("use_four_activities"),
            dataset_comb_number=run_idx, #train and val sets are randomized, test is fixed
        )
        DL.load_data()
        DL.train_test_val_split()

        if not os.path.isdir(os.path.join(train_conf.get("base_config").get("basedir"))):
            os.makedirs(os.path.join(train_conf.get("base_config").get("basedir")))


        data_dump_path = os.path.join(
            train_conf.get("base_config").get("basedir"), "dgp_data.npz"
        )
        downstream_data_dump_path = os.path.join(
            train_conf.get("base_config").get("basedir"), "down_data.npz"
        )

        np.savez(
            data_dump_path,
            x_train=DL.dgp_x_train,
            x_val=DL.dgp_x_val,
            target_label_train=DL.dgp_y_train,
            target_label_val=DL.dgp_y_val,
            sensitive_label_train=DL.dgp_gender_train,
            sensitive_label_val=DL.dgp_gender_val,
        )

        np.savez(
            downstream_data_dump_path,
            male_x_train=DL.male_down_x_train,
            female_x_train=DL.female_down_x_train,
            male_x_test=DL.male_down_x_test,
            female_x_test=DL.female_down_x_test,
            male_x_val=DL.male_down_x_val,
            female_x_val=DL.female_down_x_val,
            male_target_label_train=DL.male_down_y_train,
            female_target_label_train=DL.female_down_y_train,
            male_target_label_test=DL.male_down_y_test,
            female_target_label_test=DL.female_down_y_test,
            male_target_label_val=DL.male_down_y_val,
            female_target_label_val=DL.female_down_y_val,
        )

        FLAGS, analysis = run_train(train_conf, data_dump_path, downstream_data_dump_path)
        result_dump_path = os.path.join(FLAGS.outdir, "results.pickle")
        pickle.dump(analysis, open(result_dump_path, "wb"))
