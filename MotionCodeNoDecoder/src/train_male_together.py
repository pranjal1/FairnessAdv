import os
import time
import json
import random
import warnings
from datetime import datetime
from types import SimpleNamespace


import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

warnings.simplefilter(action="ignore", category=FutureWarning)

from .lib.models_together import *

import pdb


def run_train(train_config, data_dump_path, downstream_data_dump_path):

    all_results = {}

    FLAGS = json.loads(
        json.dumps(train_config), object_hook=lambda item: SimpleNamespace(**item)
    )

    # np.random.seed(FLAGS.base_config.seed)
    # tf.random.set_seed(FLAGS.base_config.seed)

    if not os.path.isdir(FLAGS.base_config.basedir):
        os.makedirs(FLAGS.base_config.basedir)

    FLAGS.encoder_sizes = [int(size) for size in FLAGS.encoder_config.encoder_sizes]

    if 0 in FLAGS.encoder_sizes:
        FLAGS.encoder_sizes.remove(0)

    # Make up full exp name
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    full_exp_name = "{}_{}".format(timestamp, FLAGS.base_config.exp_name)
    outdir = os.path.join(FLAGS.base_config.basedir, full_exp_name, "training")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_prefix = os.path.join(outdir, "ckpt")
    print("Full exp name: ", full_exp_name)

    #############
    # Load data #
    #############

    data = np.load(data_dump_path)
    downstream_data = np.load(downstream_data_dump_path)
    new_length = FLAGS.base_config.time_len

    x_train = data["x_train"]
    target_label_train = data["target_label_train"]
    sensitive_label_train = data["sensitive_label_train"]

    x_train_male = downstream_data["male_x_train"]
    target_label_train_male = downstream_data["male_target_label_train"]
    sensitive_label_train_male = np.array([np.float32(1.0)] * len(x_train_male))


    x_test_male = downstream_data["male_x_test"]
    target_label_test_male = downstream_data["male_target_label_test"]
    sensitive_label_test_male = np.array([np.float32(1.0)] * len(x_test_male))

    x_test_female = downstream_data["female_x_test"]
    target_label_test_female = downstream_data["female_target_label_test"]
    sensitive_label_test_female = np.array([np.float32(0.0)] * len(x_test_female))

    tgt_uq, tgt_counts = np.unique(target_label_train, return_counts=True)
    tgt_distn = {k: v for k, v in zip(tgt_uq, tgt_counts)}

    genders_uq, genders_counts = np.unique(sensitive_label_train, return_counts=True)
    gender_distn = {k: v for k, v in zip(genders_uq, genders_counts)}

    

    ###################################
    # Define data specific parameters #
    ###################################

    data_dim = x_train.shape[-1]
    time_length = FLAGS.base_config.time_len

    print("Data Loaded")

    tf_x_train = (
        tf.data.Dataset.from_tensor_slices(
            (
                x_train,
                target_label_train,
                sensitive_label_train,
            )
        )
        .shuffle(len(x_train))
        .batch(FLAGS.base_config.batch_size)
        .repeat()
    )

    tf_x_train_male = (
        tf.data.Dataset.from_tensor_slices(
            (
                x_train_male,
                target_label_train_male,
                sensitive_label_train_male,
            )
        )
        .shuffle(len(x_train_male))
        .batch(FLAGS.base_config.batch_size)
        .repeat()
    )

    #test data male
    num_split = np.ceil(len(x_test_male) / FLAGS.base_config.batch_size)
    male_x_test_batches = np.array_split(
        x_test_male, num_split, axis=0
    )

    #test data female
    num_split = np.ceil(len(x_test_female) / FLAGS.base_config.batch_size)
    female_x_test_batches = np.array_split(
        x_test_female, num_split, axis=0
    )


    ###############
    # Build model #
    ###############
    model = TrainerModel(
        latent_dim=FLAGS.base_config.latent_dim,
        data_dim=data_dim,
        time_length=time_length,
        encoder_sizes=FLAGS.encoder_config.encoder_sizes,
        encoder=Encoder,
        tgt_distn=tgt_distn,
        gender_distn=gender_distn,
        num_gru_layers_tgt=FLAGS.activity_predictor_config.num_gru_layers,
        gru_units_tgt=FLAGS.activity_predictor_config.gru_units,
        dense_size_tgt=FLAGS.activity_predictor_config.dense_size,
        use_weighted_loss_tgt=FLAGS.activity_predictor_config.use_weighted_loss,
        weighted_loss_multiplier_adv_tgt=FLAGS.activity_predictor_config.weighted_loss_multiplier,
        num_gru_layers_sens=FLAGS.gender_predictor_config.num_gru_layers,
        gru_units_sens=FLAGS.gender_predictor_config.gru_units,
        dense_size_sens=FLAGS.gender_predictor_config.dense_size,
        use_weighted_loss_sens=FLAGS.gender_predictor_config.use_weighted_loss,
        weighted_loss_multiplier_adv_sens=FLAGS.gender_predictor_config.weighted_loss_multiplier,
        batch_size=FLAGS.base_config.batch_size,
    )

    ########################
    # Training preparation #
    ########################
    print("TF Version: ", tf.__version__)
    print("GPU support: ", tf.test.is_gpu_available())

    print("Training...")
    trainable_vars = model.get_trainable_vars()
    print(f"Total trainable variables {len(trainable_vars)}")
    dgp_enc_training_vars = [
        tv for tv in trainable_vars if tv.name.startswith("dgp_enc")
    ]
    print(f"Total dgp-vae encoder trainable variables {len(dgp_enc_training_vars)}")
    tt_classifier_training_vars = [
        tv for tv in trainable_vars if tv.name.startswith("adv_classifier_tt")
    ]
    print(f"Total tt classifier variables {len(tt_classifier_training_vars)}")
    ts_classifier_training_vars = [
        tv for tv in trainable_vars if tv.name.startswith("adv_classifier_ts")
    ]
    print(
        f"Total ts adversarial classifier variables {len(ts_classifier_training_vars)}"
    )
    dgp_enc_tt_training_vars = dgp_enc_training_vars + tt_classifier_training_vars
    print(
        f"Total dgp-vae encoder and tt trainable variables {len(dgp_enc_tt_training_vars)}"
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.base_config.learning_rate)

    saver = tf.train.Checkpoint(
        optimizer=optimizer,
        encoder=model.encoder.net,
        activity_nw=model.adv_classifier_tt.model,
        gender_nw=model.adv_classifier_ts.model,
    )

    summary_writer = tf.summary.create_file_writer(outdir, flush_millis=10000)

    if FLAGS.base_config.num_steps == 0:
        num_steps = FLAGS.base_config.num_epochs * len(x_train) // FLAGS.base_config.batch_size
    else:
        num_steps = FLAGS.base_config.num_steps
    print(f"Number of training steps: {num_steps}")

    if FLAGS.base_config.print_interval == 0:
        FLAGS.base_config.print_interval = num_steps // FLAGS.base_config.num_epochs

    ############
    # Training #
    ############

    losses_train_enc_tgt = []
    losses_train_gender = []

    losses_refine_train_enc_tgt = []
    losses_refine_train_gender = []

    t0_global = time.time()
    t0 = time.time()    

    print("Training encoder-tt network")
    refinement_steps = 4*num_steps
    with summary_writer.as_default():
        for i, (x_seq, tgt_label, sens_label) in enumerate(
            tf_x_train.take(refinement_steps)
        ):
            try:
                with tf.GradientTape() as enc_tt_tape:
                    enc_tt_tape.watch(dgp_enc_tt_training_vars)
                    loss_tt_male, loss_tt_female, loss_ts_male, loss_ts_female, loss_ts_male_opp, loss_ts_female_opp = model.compute_loss(
                        x=x_seq,
                        target_label=tgt_label,
                        target_sensitive=sens_label,
                    )
                    loss_enc_tt = loss_tt_male
                    losses_train_enc_tgt.append(loss_enc_tt.numpy())
                grads_enc_tt = enc_tt_tape.gradient(loss_enc_tt, dgp_enc_tt_training_vars)
                grads_enc_tt = [
                    np.nan_to_num(grad, posinf=1e4, neginf=-1e4) for grad in grads_enc_tt
                ]
                grads_enc_tt, global_norm = tf.clip_by_global_norm(
                    grads_enc_tt, FLAGS.base_config.gradient_clip
                )
                optimizer.apply_gradients(zip(grads_enc_tt, dgp_enc_tt_training_vars))
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                break
    
    # # test acc without discriminator:

    male_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_tt.model(model.encode(x_batch)).numpy() for x_batch in male_x_test_batches]
    )
    ypred = np.argmax(male_downstream_y_test_pred, axis=-1)
    res = np.unique(target_label_test_male == ypred, return_counts=True)
    print("Result from training on male test in step 1 = ", res)
    print("Result from training on male test in step 1 percent = ", res[1]/np.sum(res[1]))

    male_res = {
            "ypred":ypred,
            "ytrue":target_label_test_male,
            "cm":confusion_matrix(target_label_test_male, ypred),
            "acc":accuracy_score(target_label_test_male, ypred),
        }


    female_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_tt.model(model.encode(x_batch)).numpy() for x_batch in female_x_test_batches]
    )
    ypred = np.argmax(female_downstream_y_test_pred, axis=-1)
    res = np.unique(target_label_test_female == ypred, return_counts=True)
    print("Result from training on female test in step 1 = ", res)
    print("Result from training on female test in step 1 percent = ", res[1]/np.sum(res[1]))

    female_res = {
            "ypred":ypred,
            "ytrue":target_label_test_female,
            "cm":confusion_matrix(target_label_test_female, ypred),
            "acc":accuracy_score(target_label_test_female, ypred)
        }

    all_results["male_train_original"] = {
        "male_test":male_res,
        "female_test":female_res,
    }

    print("Training ts network")
    refinement_steps = 2 * num_steps
    with summary_writer.as_default():
        for i, (x_seq, tgt_label, sens_label) in enumerate(
            tf_x_train.take(refinement_steps)
        ):
            try:
                with tf.GradientTape() as gender_tape:
                    gender_tape.watch(ts_classifier_training_vars)
                    loss_tt_male, loss_tt_female, loss_ts_male, loss_ts_female, loss_ts_male_opp, loss_ts_female_opp = model.compute_loss(
                        x=x_seq,
                        target_label=tgt_label,
                        target_sensitive=sens_label,
                    )
                    loss_gender = loss_ts_male + loss_ts_female
                    losses_train_gender.append(loss_gender.numpy())

                grads_ts = gender_tape.gradient(loss_gender, ts_classifier_training_vars)
                grads_ts = [
                    np.nan_to_num(grad, posinf=1e4, neginf=-1e4) for grad in grads_ts
                ]
                grads_ts, global_norm = tf.clip_by_global_norm(
                    grads_ts, FLAGS.base_config.gradient_clip
                )
                optimizer.apply_gradients(zip(grads_ts, ts_classifier_training_vars))
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                break
    
    # training acc at the end of step 2:
    num_split = np.ceil(len(x_train) / FLAGS.base_config.batch_size)
    x_train_batches = np.array_split(
        x_train, num_split, axis=0
    )
    downstream_s_train_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in x_train_batches]
    )
    ypred = np.argmax(downstream_s_train_pred, axis=-1)
    res = np.unique(sensitive_label_train == ypred, return_counts=True)
    print("Result from training in step 2 = ", res)
    print("In percent = ", res[1]/np.sum(res[1]))

    male_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in male_x_test_batches]
    )
    ypred = np.argmax(male_downstream_y_test_pred, axis=-1)
    res = np.unique(sensitive_label_test_male == ypred, return_counts=True)
    print("Result from training on male test in step 2 = ", res)
    print("In percent = ", res[1]/np.sum(res[1]))

    female_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in female_x_test_batches]
    )
    ypred = np.argmax(female_downstream_y_test_pred, axis=-1)
    res = np.unique(sensitive_label_test_female == ypred, return_counts=True)
    print("Result from training on female test in step 2 = ", res)
    print("In percent = ", res[1]/np.sum(res[1]))

    print("Training encoder-tt network, regularization with adv loss")
    refinement_steps = 20 * num_steps
    with summary_writer.as_default():
        for i, (x_seq, tgt_label, sens_label) in enumerate(
            tf_x_train.take(refinement_steps)
        ):
            try:
                with tf.GradientTape() as enc_tt_tape, tf.GradientTape() as gender_tape:
                    enc_tt_tape.watch(dgp_enc_tt_training_vars)
                    gender_tape.watch(ts_classifier_training_vars)
                    loss_tt_male, loss_tt_female, loss_ts_male, loss_ts_female, loss_ts_male_opp, loss_ts_female_opp = model.compute_loss(
                        x=x_seq,
                        target_label=tgt_label,
                        target_sensitive=sens_label,
                    )
                    loss_gender = loss_ts_male + loss_ts_female
                    # loss_enc_tt = 0.7 * loss_tt_male - loss_ts_male
                    # loss_enc_tt = 0.4 * loss_tt_male - loss_ts_male
                    # loss_enc_tt = 1.0 * loss_tt_male - loss_ts_male
                    # loss_enc_tt = 0.0001 * loss_tt_male - loss_ts_male
                    loss_enc_tt = 0.5 * loss_tt_male - loss_ts_male
                    losses_refine_train_enc_tgt.append(loss_enc_tt.numpy())
                    losses_refine_train_gender.append(loss_gender.numpy())
                
                # sample_rand = random.random()
                # if sample_rand < 0.2:
                grads_enc_tt = enc_tt_tape.gradient(loss_enc_tt, dgp_enc_tt_training_vars)
                grads_enc_tt = [
                    np.nan_to_num(grad, posinf=1e4, neginf=-1e4) for grad in grads_enc_tt
                ]
                grads_enc_tt, global_norm = tf.clip_by_global_norm(
                    grads_enc_tt, FLAGS.base_config.gradient_clip
                )
                optimizer.apply_gradients(zip(grads_enc_tt, dgp_enc_tt_training_vars))
                
                # else:
                grads_ts = gender_tape.gradient(loss_gender, ts_classifier_training_vars)
                grads_ts = [
                    np.nan_to_num(grad, posinf=1e4, neginf=-1e4) for grad in grads_ts
                ]
                grads_ts, global_norm = tf.clip_by_global_norm(
                    grads_ts, FLAGS.base_config.gradient_clip
                )
                optimizer.apply_gradients(zip(grads_ts, ts_classifier_training_vars))
            except KeyboardInterrupt:
                saver.save(checkpoint_prefix)
                break
    
    # training acc at the end of step 3:
    num_split = np.ceil(len(x_train_male) / FLAGS.base_config.batch_size)
    male_x_train_batches = np.array_split(
        x_train_male, num_split, axis=0
    )
    male_downstream_y_train_pred = np.vstack(
        [model.adv_classifier_tt.model(model.encode(x_batch)).numpy() for x_batch in male_x_train_batches]
    )
    male_downstream_s_train_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in male_x_train_batches]
    )

    ypred_tgt = np.argmax(male_downstream_y_train_pred, axis=-1)
    ypred_sens = np.argmax(male_downstream_s_train_pred, axis=-1)
    res_tgt = np.unique(target_label_train_male == ypred_tgt, return_counts=True)
    res_sens = np.unique(sensitive_label_train_male == ypred_sens, return_counts=True)
    print("Tgt Result from training in step 3 = ", res_tgt)
    print("Sens Result from training in step 3 = ", res_sens)


    male_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_tt.model(model.encode(x_batch)).numpy() for x_batch in male_x_test_batches]
    )
    male_downstream_s_test_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in male_x_test_batches]
    )
    male_downstream_z_test = np.vstack(
        [model.encode(x_batch).numpy() for x_batch in male_x_test_batches]
    )
    ypred_tgt = np.argmax(male_downstream_y_test_pred, axis=-1)
    ypred_sens = np.argmax(male_downstream_s_test_pred, axis=-1)
    res_tgt = np.unique(target_label_test_male == ypred_tgt, return_counts=True)
    res_sens = np.unique(sensitive_label_test_male == ypred_sens, return_counts=True)
    print("Tgt Result from training on male test in step 3 = ", res_tgt)
    print("Tgt Result from training on male test in step 3 percent = ", res_tgt[1]/np.sum(res_tgt[1]))
    print("Sens Result from training on male test in step 3 = ", res_sens)
    print("Sens Result from training on male test in step 3 percent = ", res_sens[1]/np.sum(res_sens[1]))

    male_res = {
            "ypred":ypred_tgt,
            "ytrue":target_label_test_male,
            "cm":confusion_matrix(target_label_test_male, ypred_tgt),
            "acc":accuracy_score(target_label_test_male, ypred_tgt)
        }


    female_downstream_y_test_pred = np.vstack(
        [model.adv_classifier_tt.model(model.encode(x_batch)).numpy() for x_batch in female_x_test_batches]
    )
    female_downstream_s_test_pred = np.vstack(
        [model.adv_classifier_ts.model(model.encode(x_batch)).numpy() for x_batch in female_x_test_batches]
    )
    female_downstream_z_test = np.vstack(
        [model.encode(x_batch).numpy() for x_batch in female_x_test_batches]
    )
    ypred_tgt = np.argmax(female_downstream_y_test_pred, axis=-1)
    ypred_sens = np.argmax(female_downstream_s_test_pred, axis=-1)
    res_tgt = np.unique(target_label_test_female == ypred_tgt, return_counts=True)
    res_sens = np.unique(sensitive_label_test_female == ypred_sens, return_counts=True)
    print("Tgt Result from training on female test in step 3 = ", res_tgt)
    print("Tgt Result from training on female test in step 3 percent = ", res_tgt[1]/np.sum(res_tgt[1]))
    print("Sens Result from training on female test in step 3 = ", res_sens)
    print("Sens Result from training on female test in step 3 percent = ", res_sens[1]/np.sum(res_sens[1]))

    female_res = {
            "ypred":ypred_tgt,
            "ytrue":target_label_test_female,
            "cm":confusion_matrix(target_label_test_female, ypred_tgt),
            "acc":accuracy_score(target_label_test_female, ypred_tgt)
        }

    all_results["male_train_w_adv"] = {
        "male_test":male_res,
        "female_test":female_res,
    }

    t_train_total = time.time() - t0_global

    print(f"Total training time: {t_train_total}")

    with open(
        os.path.join(outdir, f"training_curve_enc_tt.tsv"), "w"
    ) as outfile:
        outfile.write("\t".join(map(str, losses_train_enc_tgt)))

    with open(
        os.path.join(
            outdir, f"training_curve_gender_classifier.tsv"
        ),
        "w",
    ) as outfile:
        outfile.write("\t".join(map(str, losses_train_gender)))

    with open(
        os.path.join(outdir, f"training_curve_refine_enc_tt.tsv"),
        "w",
    ) as outfile:
        outfile.write("\t".join(map(str, losses_refine_train_enc_tgt)))

    with open(
        os.path.join(
            outdir, f"training_curve_refine_gender_classifiers.tsv"
        ),
        "w",
    ) as outfile:
        outfile.write("\t".join(map(str, losses_refine_train_gender)))

    np.savez(
        os.path.join(outdir, "downstream_z"),
        male_z_test=male_downstream_z_test,
        female_z_test=female_downstream_z_test,
    )

    FLAGS.latent_path = os.path.join(outdir, "downstream_z.npz")

    print("Training finished.")

    FLAGS.outdir = outdir

    return FLAGS, all_results
