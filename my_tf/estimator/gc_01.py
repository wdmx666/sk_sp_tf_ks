import tensorflow as tf
import shutil,os,path
import math
from datetime import datetime
from types import SimpleNamespace
from tensorflow.python.feature_column import feature_column
from typing import ByteString, Dict, Tuple, Optional, Any,List


class MetaPoj(SimpleNamespace):
    RESUME_TRAINING = False
    PROCESS_FEATURES = True
    MULTI_THREADING = True


class MetaRaw(SimpleNamespace):
    root_p = 'E:/my_proj/ml_proj/sk_sp_tf_ks/my_tf/'
    TRAIN_DATA_FILES_PATTERN = root_p + 'data/reg/train-*.csv'
    VALID_DATA_FILES_PATTERN = root_p + 'data/reg/valid-*.csv'
    TEST_DATA_FILES_PATTERN = root_p + 'data/reg/test-*.csv'

    HEADER = ['key', 'x', 'y', 'alpha', 'beta', 'target']
    HEADER_DEFAULTS = [[0], [0.0], [0.0], ['NA'], ['NA'], [0.0]]
    NUMERIC_FEATURE_NAMES = ['x', 'y']
    CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01', 'ax02'], 'beta': ['bx01', 'bx02']}
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys())
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    TARGET_NAME = 'target'
    UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})
    print("Header: {}".format(HEADER))
    print("Numeric Features: {}".format(NUMERIC_FEATURE_NAMES))
    print("Categorical Features: {}".format(CATEGORICAL_FEATURE_NAMES))
    print("Target: {}".format(TARGET_NAME))
    print("Unused Features: {}".format(UNUSED_FEATURE_NAMES))


class MetaETL(SimpleNamespace):
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = ['x_2', 'y_2', 'xy', 'dist_xy']
    ALL_NUMERIC_FEATURE_NAMES = MetaRaw.NUMERIC_FEATURE_NAMES + CONSTRUCTED_NUMERIC_FEATURES_NAMES


def parse_csv_row(csv_row: ByteString)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """get a string tensor"""
    columns = tf.decode_csv(csv_row, record_defaults=MetaRaw.HEADER_DEFAULTS)
    features: Dict[str, tf.Tensor] = dict(zip(MetaRaw.HEADER, columns))
    for column in MetaRaw.UNUSED_FEATURE_NAMES:
        features.pop(column)
    target = features.pop(MetaRaw.TARGET_NAME)
    print('parse_csv_row--> ', features.keys(),features.values())
    return features, target


def process_features(features: Dict[str, tf.Tensor])->Dict[str, tf.Tensor]:
    features["x_2"] = tf.square(features['x'])
    features["y_2"] = tf.square(features['y'])
    features["xy"] = tf.multiply(features['x'], features['y'])  # features['x'] * features['y']
    features['dist_xy'] = tf.sqrt(tf.squared_difference(features['x'], features['y']))
    print('process_features--> ', features.keys(), features.values())
    return features


def csv_input_fn(file_name_pattern: str, mode: str=tf.estimator.ModeKeys.EVAL, batch_size: int=200,
                 num_epochs: Optional[int]=None, skip_header_lines: int=0)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    input_file_names: tf.Tensor = tf.matching_files(pattern=file_name_pattern)
    dataset = tf.data.TextLineDataset(input_file_names)
    dataset = dataset.skip(skip_header_lines)
    dataset = dataset.map(parse_csv_row)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)
    if MetaPoj.PROCESS_FEATURES:
        dataset = dataset.map(lambda features, target: (process_features(features), target))
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    feaures, target = iterator.get_next()
    return feaures, target


def get_feature_columns()->Dict[str, Any]:
    # 将各种列赋予类型，根据类型可以有响应处理
    numeric_columns: Dict[str, Any] = {feature_name: tf.feature_column.numeric_column(feature_name) for
                                       feature_name in MetaETL.ALL_NUMERIC_FEATURE_NAMES}

    categorical_column_with_vocabulary: Dict[str, Any] = {
        item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
        for item in MetaRaw.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.items()}

    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)
    # 之所用用字典，方便后面按照名称索引进一步处理，因为列还没有进一步转化
    feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column([feature_columns['alpha'], feature_columns['beta']], 4)
    return feature_columns


def get_final_feature_columns()->List[Any]:
    # 检查字段并对字段做按类别做出进一步处理
    FEATURE_COLUMNS = list(get_feature_columns().values())
    dense_columns = list(filter(lambda column: isinstance(column, feature_column._NumericColumn),FEATURE_COLUMNS))

    categorical_columns = list(
        filter(lambda column: isinstance(column, feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn), FEATURE_COLUMNS))

    # convert categorical columns to indicators,独热码化
    indicator_columns = list(map(lambda column: tf.feature_column.indicator_column(column), categorical_columns))
    return dense_columns + indicator_columns


def create_estimator(run_config, hparams)->tf.estimator.Estimator:

    estimator = tf.estimator.DNNRegressor(
        feature_columns=get_final_feature_columns(),
        hidden_units=hparams.hidden_units,
        dropout=hparams.dropout_prob,
        activation_fn=tf.nn.elu,
        optimizer=tf.train.AdamOptimizer(),
        config=run_config)

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator

# ##########################准备试验数据#############################


#  #####定义试验参数#######
class MetaMD(SimpleNamespace):
    EVAL_AFTER_SEC = 15
    NUM_EPOCHS = 10
    BATCH_SIZE = 500
    TRAIN_SIZE = 12000
    TOTAL_STEPS = (TRAIN_SIZE/BATCH_SIZE)*NUM_EPOCHS

    MODEL_NAME = 'reg-model-1_2'
    MODEL_DIR = path.Path('trained_models/{}'.format(MODEL_NAME)).makedirs_p()
    EXPORT_DIR = path.Path(MODEL_DIR + "/export/estimate").makedirs_p()

    HPARAMS = tf.contrib.training.HParams(hidden_units=[8, 4], dropout_prob=0.0)

    RUN_CONFIG = tf.estimator.RunConfig(
        save_checkpoints_steps=480, # to evaluate after each 20 epochs => (12000/500) * 20
        tf_random_seed=19830610,
        model_dir=MODEL_DIR)

    print("Model directory: {}".format(RUN_CONFIG.model_dir))
    print("Hyper-parameters: {}".format(HPARAMS))


#  #####定义服务函数，用于配置导出 #######
def csv_serving_input_fn()->tf.estimator.export.ServingInputReceiver:
    SERVING_HEADER = ['x', 'y', 'alpha', 'beta']
    SERVING_HEADER_DEFAULTS = [[0.0], [0.0], ['NA'], ['NA']]
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')
    receiver_tensor = {'csv_rows': rows_string_tensor}
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=SERVING_HEADER_DEFAULTS)
    features = dict(zip(SERVING_HEADER, columns))

    if MetaPoj.PROCESS_FEATURES:
        features = process_features(features)

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


# 定义训练和验证的输入函数
class ExperimentConfig(SimpleNamespace):
    train_input_tr_fn = lambda: csv_input_fn(
        file_name_pattern=MetaRaw.TRAIN_DATA_FILES_PATTERN, mode=tf.estimator.ModeKeys.TRAIN,
        num_epochs=MetaMD.NUM_EPOCHS, batch_size=MetaMD.BATCH_SIZE)

    eval_input_tr_fn = lambda: csv_input_fn(
        file_name_pattern=MetaRaw.VALID_DATA_FILES_PATTERN, mode=tf.estimator.ModeKeys.EVAL,
        num_epochs=1, batch_size=MetaMD.BATCH_SIZE)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_tr_fn, max_steps=MetaMD.TOTAL_STEPS)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_tr_fn,
        exporters=[tf.estimator.LatestExporter(name="estimate",serving_input_receiver_fn=csv_serving_input_fn,exports_to_keep=1,as_text=True)],
        steps=None, throttle_secs=MetaMD.EVAL_AFTER_SEC)  # evalute after each 15 training seconds!

    train_input_eval_fn = lambda: csv_input_fn(file_name_pattern=MetaRaw.TRAIN_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.EVAL, num_epochs=MetaMD.NUM_EPOCHS, batch_size=MetaMD.BATCH_SIZE)

    eval_input_eval_fn = lambda: csv_input_fn(file_name_pattern=MetaRaw.VALID_DATA_FILES_PATTERN,
        mode=tf.estimator.ModeKeys.EVAL, num_epochs=1, batch_size=MetaMD.BATCH_SIZE)

    test_input_eval_fn = lambda: csv_input_fn(file_name_pattern=MetaRaw.TEST_DATA_FILES_PATTERN,
                                              mode=tf.estimator.ModeKeys.EVAL, num_epochs=1,
                                              batch_size=MetaMD.BATCH_SIZE)


def run_experiment(EXP_CFG, estimator):
    if MetaPoj.RESUME_TRAINING:
        print("Removing previous artifacts...")
        shutil.rmtree(MetaMD.MODEL_DIR, ignore_errors=True)
    else:
        print("Resuming training...")

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
    print("############################################################################################")
    #print("# Train RMSE: {} - {}".format(train_rmse, train_results))
    print("############################################################################################")

    eval_results = estimator.evaluate(input_fn=EXP_CFG.eval_input_eval_fn, steps=1)
    #eval_rmse = round(math.sqrt(eval_results["average_loss"]), 5)
    print("############################################################################################")
    #print("# eval RMSE: {} - {}".format(eval_rmse, eval_results))
    print("############################################################################################")

    test_results = estimator.evaluate(input_fn=EXP_CFG.test_input_eval_fn, steps=1)
    #test_rmse = round(math.sqrt(test_results["average_loss"]), 5)
    print("############################################################################################")
    #print("# Test RMSE: {} - {}".format(test_rmse, test_results))
    print("############################################################################################")
    predictions = estimator.predict(input_fn=EXP_CFG.test_input_eval_fn)
    for it in range(6): print(next(predictions))


def export_model(estimator, model_dir, sub_dir=''):
    export_dir = path.Path(model_dir + sub_dir).makedirs_p()
    estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=csv_serving_input_fn, as_text=True)
    print(export_dir)
    return export_dir


def predict_input(export_dir):
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]
    print(saved_model_dir)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir,signature_def_key='predict')
    output = predictor_fn({'csv_rows': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
    print(output)


def predict_input2(model_dir):
    import os
    export_dir = model_dir +"/export/estimate/"
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]
    print(saved_model_dir)

    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='serving_default')
    output = predictor_fn({'inputs': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
    print(output)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='regression')
    output = predictor_fn({'inputs': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
    print(output)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='predict')
    output = predictor_fn({'csv_rows': ["0.5,1,ax01,bx02", "-0.5,-1,ax02,bx02"]})
    print(output)



def main(argv):
    estimator = create_estimator(MetaMD.RUN_CONFIG, MetaMD.HPARAMS)
    run_experiment(ExperimentConfig, estimator)
    export_dir = export_model(estimator, MetaMD.MODEL_DIR, sub_dir='/my_export')
    predict_input(export_dir)
    print("_-------------------------------------------------------______")
    predict_input2(MetaMD.MODEL_DIR)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    #main(None)

