import pandas as pd
import tensorflow as tf
from datetime import datetime
import copy
from types import SimpleNamespace
import path
from typing import Dict,List,Tuple,Optional,Any

#  #####################数据ETL##########################
MetaRaw = SimpleNamespace()
MetaRaw.TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
MetaRaw.TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"
MetaRaw.TARGET = 'Species'
MetaRaw.FEATURE_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
MetaRaw.CATEGORICAL_FEATURE_NAMES = []
MetaRaw.NUMERIC_FEATURE_NAMES = MetaRaw.FEATURE_NAMES

MetaRaw.CSV_COLUMN_NAMES = MetaRaw.FEATURE_NAMES + [MetaRaw.TARGET]
MetaRaw.SPECIES = ['Setosa', 'Versicolor', 'Virginica']
MetaRaw.CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
MetaRaw.FEATURE_DEFAULT = [[0.0], [0.0], [0.0], [0.0]]

MetaETL = SimpleNamespace()


# extract
def maybe_download():
    train_path = tf.keras.utils.get_file(MetaRaw.TRAIN_URL.split('/')[-1], MetaRaw.TRAIN_URL)
    test_path = tf.keras.utils.get_file(MetaRaw.TEST_URL.split('/')[-1], MetaRaw.TEST_URL)
    return train_path, test_path


def load_data()->Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
    train_path, test_path = maybe_download()
    train = pd.read_csv(train_path, names=MetaRaw.CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=MetaRaw.CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(MetaRaw.TARGET)
    test_x, test_y = test, test.pop(MetaRaw.TARGET)
    return (train_x, train_y), (test_x, test_y)


# transform
def parse_csv_row(csv_row)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    columns = tf.decode_csv(csv_row, record_defaults=MetaRaw.CSV_TYPES)
    features = dict(zip(MetaRaw.CSV_COLUMN_NAMES, columns))
    target = features.pop(MetaRaw.TARGET)
    return features, target


def process_features(features):
    pass


# load
def csv_input_fn(file_name_pattern, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=0,
                 num_epochs=None, batch_size=200)->tf.data.Dataset:
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    file_names = tf.matching_files(file_name_pattern)
    dataset = tf.data.TextLineDataset(file_names)
    dataset = dataset.skip(skip_header_lines)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    dataset = dataset.map(parse_csv_row)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


# #########################数据特征工程，生成特征列##########################
ds = csv_input_fn(file_name_pattern='')
MetaETL.FEATURE_NAMES = list(ds.output_classes[0].keys())
MetaETL.NUMERIC_FEATURE_NAMES = copy.deepcopy(MetaETL.FEATURE_NAMES)
MetaETL.TARGET = copy.deepcopy(MetaRaw.TARGET)


def get_feature_columns()->Dict[str, Any]:
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in MetaETL.NUMERIC_FEATURE_NAMES}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
    return feature_columns


# ##########################根据参数定义估计器#######################################

def create_estimator(run_config, hparams)->tf.estimator.Estimator:
    """不写死，通过参数传入，相当于sklearn中定义估计器"""
    my_feature_columns = list(get_feature_columns().values())
    estimator = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=hparams.hidden_units,
        # The model must choose between 3 classes.
        n_classes=hparams.n_classes,
        config=run_config
    )
    return estimator


# ############################定义服务导出的输入函数############################################
def csv_serving_input_fn():
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')
    receiver_tensor = {'csv_rows': rows_string_tensor}
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=MetaRaw.FEATURE_DEFAULT)
    features = dict(zip(MetaRaw.FEATURE_NAMES, columns))
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


# #############################定义模型的超参数及运行配置##########################################

def make_experiment_config():
    MetaMD = SimpleNamespace()
    MetaMD.TRAIN_SIZE = 1200
    MetaMD.NUM_EPOCHS = 100
    MetaMD.BATCH_SIZE = 20
    MetaMD.EVAL_AFTER_SEC = 15
    MetaMD.TOTAL_STEPS = (MetaMD.TRAIN_SIZE/MetaMD.BATCH_SIZE)*MetaMD.NUM_EPOCHS
    MetaMD.MODEL_NAME = 'my_iris'
    MetaMD.MODEL_DIR = path.Path('trained_models/{}'.format(MetaMD.MODEL_NAME)).makedirs_p()
    MetaMD.EXPORT_DIR = path.Path(MetaMD.MODEL_DIR + "/export/estimate").makedirs_p()

    MetaMD.HPARAMS=tf.contrib.training.HParams(
        num_epochs=MetaMD.NUM_EPOCHS,
        batch_size=MetaMD.BATCH_SIZE,
        hidden_units=[10, 10],
        max_steps=MetaMD.TOTAL_STEPS,
        n_classes=3)

    MetaMD.RUN_CONFIG = tf.estimator.RunConfig(tf_random_seed=19830610, model_dir=MetaMD.MODEL_DIR)
    return MetaMD


def config_experiment(MetaMD):
    EXP_CFG = SimpleNamespace()
    train_path, test_path = maybe_download()

    train_input_tr_fn = lambda: csv_input_fn(train_path, mode=tf.estimator.ModeKeys.TRAIN, skip_header_lines=1,
                                          batch_size=MetaMD.HPARAMS.batch_size, num_epochs=MetaMD.HPARAMS.num_epochs)
    tr_eval_input_tr_fn = lambda: csv_input_fn(test_path, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                         batch_size=MetaMD.HPARAMS.batch_size, num_epochs=MetaMD.HPARAMS.num_epochs)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_tr_fn, max_steps=MetaMD.HPARAMS.max_steps, hooks=None)

    eval_spec = tf.estimator.EvalSpec(input_fn=tr_eval_input_tr_fn,
        exporters=[tf.estimator.LatestExporter(name="estimate", serving_input_receiver_fn=csv_serving_input_fn, exports_to_keep=1,as_text=True)],
        steps=None,#hooks=[EarlyStoppingHook(15)],
        throttle_secs=MetaMD.EVAL_AFTER_SEC)

    train_input_eval_fn = lambda: csv_input_fn(train_path, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                               batch_size=MetaMD.HPARAMS.batch_size, num_epochs=MetaMD.HPARAMS.num_epochs)
    eval_eval_input_eval_fn = lambda: csv_input_fn(test_path, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                                   batch_size=MetaMD.HPARAMS.batch_size, num_epochs=MetaMD.HPARAMS.num_epochs)
    EXP_CFG.train_input_tr_fn = train_input_tr_fn
    EXP_CFG.tr_eval_input_tr_fn = tr_eval_input_tr_fn
    EXP_CFG.train_spec = train_spec
    EXP_CFG.eval_spec = eval_spec
    EXP_CFG.train_input_eval_fn = train_input_eval_fn
    EXP_CFG.eval_eval_input_eval_fn = eval_eval_input_eval_fn
    return EXP_CFG


def run_experiment(MetaMD, EXP_CFG, estimator):
    time_start = datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=EXP_CFG.train_spec, eval_spec=EXP_CFG.eval_spec)
    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
    #####################################################################################################
    import math

    train_results = estimator.evaluate(input_fn=EXP_CFG.train_input_eval_fn, steps=1)
    #train_rmse = round(math.sqrt(train_results["average_loss"]), 5)
    print()
    print("############################################################################################")
    #print("# Train RMSE: {} - {}".format(train_rmse, train_results))
    print("############################################################################################")

    test_results = estimator.evaluate(input_fn=EXP_CFG.eval_eval_input_eval_fn, steps=1)
    #test_rmse = round(math.sqrt(test_results["average_loss"]), 5)
    print()
    print("############################################################################################")
    #print("# Test RMSE: {} - {}".format(test_rmse, test_results))
    print("############################################################################################")
    predictions = estimator.predict(input_fn=EXP_CFG.eval_eval_input_eval_fn)
    for it in range(10):
        it = next(predictions)
        print(it)


def predict_input(MetaMD):
    import os
    saved_model_dir = MetaMD.EXPORT_DIR + "/" + os.listdir(path=MetaMD.EXPORT_DIR)[-1]

    print(saved_model_dir)

    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir,signature_def_key="predict")

    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)
    print()


def main(argv):
    MetaMD = make_experiment_config()
    EXP_CFG = config_experiment(MetaMD)
    estimator = create_estimator(MetaMD.RUN_CONFIG, MetaMD.HPARAMS)
    run_experiment(MetaMD, EXP_CFG, estimator)
    predict_input(MetaMD)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
