import tensorflow as tf
from datetime import datetime
import copy
import path,os
from typing import Dict,List,Tuple,Optional,Any

from types import SimpleNamespace


class Parameter:
    def __init__(self, var):
        self.var = var

    @property
    def value(self):
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var, **kwargs):
        self.var = var
        return self

    def __repr__(self):
        return f"{self.value}"


class TotalSteps(Parameter):
    def __init__(self, train_size: Parameter, batch_size: Parameter, num_epochs: Parameter):
        super().__init__(None)
        self._train_size = train_size
        self._batch_size = batch_size
        self._num_epochs = num_epochs
        self._auto = True

    @property
    def value(self):
        if self._auto:
            return (self._train_size.value / self._batch_size.value) * self._num_epochs.value
        else:
            return self.var

    def set_value(self, var, auto=True):
        self._auto = auto
        self.var = var


#  #####################数据ETL##########################
class MetaRaw(SimpleNamespace):
    TRAIN_URL = Parameter("http://download.tensorflow.org/data/iris_training.csv")
    TEST_URL = Parameter("http://download.tensorflow.org/data/iris_test.csv")
    TRAIN_PATH = Parameter(tf.keras.utils.get_file(TRAIN_URL.value.split('/')[-1], TRAIN_URL))
    TEST_PATH = Parameter(tf.keras.utils.get_file(TEST_URL.value.split('/')[-1], TEST_URL))

    TARGET = Parameter('Species')
    FEATURE_NAMES = Parameter(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'])
    CATEGORICAL_FEATURE_NAMES = Parameter([])
    NUMERIC_FEATURE_NAMES = Parameter(FEATURE_NAMES)
    CSV_COLUMN_NAMES = Parameter(FEATURE_NAMES.value + [TARGET.value])
    SPECIES = Parameter(['Setosa', 'Versicolor', 'Virginica'])
    CSV_COLUMN_DEFAULTS = Parameter([[0.0], [0.0], [0.0], [0.0], [0]])
    FEATURE_DEFAULTS = Parameter([[0.0], [0.0], [0.0], [0.0]])


class MetaETL(SimpleNamespace):
    FEATURE_NAMES = MetaRaw.FEATURE_NAMES
    NUMERIC_FEATURE_NAMES = FEATURE_NAMES
    TARGET = MetaRaw.TARGET


# transform
def parse_csv_row(csv_row)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    columns = tf.decode_csv(csv_row, record_defaults=MetaRaw.CSV_COLUMN_DEFAULTS.value)
    features = dict(zip(MetaRaw.CSV_COLUMN_NAMES.value, columns))
    target = features.pop(MetaRaw.TARGET.value)
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


def get_feature_columns()->Dict[str, Any]:
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in MetaETL.NUMERIC_FEATURE_NAMES}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
    return feature_columns


# 得深入去了解Estimator要什么，而不是告诉程序员一个概念就可以
def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params.feature_columns)
    print("net-----> ", net.shape)
    for units in params.hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        print("net-----> ", net.shape)
    logits = tf.layers.dense(net, params.n_classes, activation=None)
    print("logits-----> ", logits.shape)
    predicted_classes = tf.argmax(logits, 1)
    # 在预测模式下要计算操作节点，主要是包括类别
    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {'class_ids': predicted_classes[:, tf.newaxis], 'probabilities': tf.nn.softmax(logits), 'logits': logits}

        """export_outputs指定了服务输出的signature_def，其中每组键值均代表一个signature_def,字典构成了一个命名空间，
        一个signature_def由inputs和outputs构成，同时输入输出是又是由字典构成命名空间，如上面的predictions字典，不同键
        代表不同输出项目，不指定的单个默认使用output字符串"""

        export_outputs = {'probabilities': tf.estimator.export.PredictOutput(tf.nn.softmax(logits)),
                          'class_ids': tf.estimator.export.PredictOutput(predicted_classes[:, tf.newaxis]),
                          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                              tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # 计算评价量，损失函数和其它评价函数都属于评价准则，只是用途有所区别
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)      # 损失
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    average_loss = tf.reduce_mean(loss)
    metrics = {'accuracy': accuracy, 'average_loss': tf.metrics.mean(average_loss)}

    tf.summary.scalar('accuracy', accuracy[1])
    # 验证模式下
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # 训练模式下，要运行优化节点，计算损失.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


# ############################定义服务导出的输入函数############################################
def csv_serving_input_fn():
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')
    receiver_tensor = {'csv_rows': rows_string_tensor}
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=MetaRaw.FEATURE_DEFAULTS)
    features = dict(zip(MetaRaw.FEATURE_NAMES, columns))
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


class MetaMD(SimpleNamespace):
    TRAIN_SIZE = 1200
    NUM_EPOCHS = 100
    BATCH_SIZE = 20
    #EVAL_AFTER_SEC = 1
    TOTAL_STEPS = (TRAIN_SIZE / BATCH_SIZE) * NUM_EPOCHS
    MODEL_NAME = 'my_custom_iris'
    MODEL_DIR = path.Path('trained_models/{}'.format(MODEL_NAME)).makedirs_p()
    EXPORT_DIR = path.Path(MODEL_DIR + "/export/estimate").makedirs_p()

    HPARAMS = tf.contrib.training.HParams(
        feature_columns=list(get_feature_columns().values()),
        hidden_units=[10, 10],
        n_classes=3,
        learning_rate=0.01,
        max_steps=TOTAL_STEPS)

    RUN_CONFIG = tf.estimator.RunConfig(tf_random_seed=19830610, model_dir=MODEL_DIR)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(HPARAMS)
    print("Model Directory:", MODEL_DIR)
    print("Dataset Size:", TRAIN_SIZE)
    print("Batch Size:", BATCH_SIZE)
    print("Steps per Epoch:", TRAIN_SIZE / BATCH_SIZE)
    print("Total Steps:", TOTAL_STEPS)
    #print("That is 1 evaluation step after each", EVAL_AFTER_SEC, " training seconds")


def create_estimator(model, run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model, params=hparams, config=run_config)
    print("Estimator Type: {}".format(type(estimator)))
    return estimator


class ExperimentConfig(SimpleNamespace):
    train_input_tr_fn = lambda: csv_input_fn(MetaRaw.TRAIN_PATH, mode=tf.estimator.ModeKeys.TRAIN, skip_header_lines=1,
                                          batch_size=MetaMD.BATCH_SIZE, num_epochs=MetaMD.NUM_EPOCHS)
    eval_input_tr_fn = lambda: csv_input_fn(MetaRaw.TEST_PATH, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                         batch_size=MetaMD.BATCH_SIZE, num_epochs=MetaMD.NUM_EPOCHS)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_tr_fn, max_steps=MetaMD.HPARAMS.max_steps, hooks=None)

    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_tr_fn, steps=None, #throttle_secs=MetaMD.EVAL_AFTER_SEC
        exporters=[tf.estimator.LatestExporter(name="estimate", serving_input_receiver_fn=csv_serving_input_fn,as_text=True)],
                                      )

    train_input_eval_fn = lambda: csv_input_fn(MetaRaw.TRAIN_PATH, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                               batch_size=MetaMD.BATCH_SIZE, num_epochs=MetaMD.NUM_EPOCHS)
    eval_input_eval_fn = lambda: csv_input_fn(MetaRaw.TEST_PATH, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                                   batch_size=MetaMD.BATCH_SIZE, num_epochs=MetaMD.NUM_EPOCHS)


def run_experiment(EXP_CFG, estimator):
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

    test_results = estimator.evaluate(input_fn=EXP_CFG.eval_input_eval_fn, steps=1)
    #test_rmse = round(math.sqrt(test_results["average_loss"]), 5)
    print()
    print("############################################################################################")
    #print("# Test RMSE: {} - {}".format(test_rmse, test_results))
    print("############################################################################################")

    predictions = estimator.predict(input_fn=EXP_CFG.eval_input_eval_fn)

    for it in range(10):
        it = next(predictions)
        print(it)


def export_model(estimator, model_dir, sub_dir=''):
    export_dir = path.Path(model_dir + sub_dir).makedirs_p()
    estimator.export_savedmodel(export_dir_base=export_dir, serving_input_receiver_fn=csv_serving_input_fn, as_text=True)
    print(export_dir)
    return export_dir


def predict_input(export_dir):
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]
    print(saved_model_dir)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='class_ids')
    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir)
    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)

def predict_input2(model_dir):
    import os
    export_dir = model_dir + "/export/estimate/"
    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]
    print(saved_model_dir)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='probabilities')
    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='serving_default')
    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)
    predictor_fn = tf.contrib.predictor.from_saved_model(export_dir=saved_model_dir, signature_def_key='class_ids')
    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)



def main(argv):
    estimator = create_estimator(my_model, MetaMD.RUN_CONFIG, MetaMD.HPARAMS)
    run_experiment(ExperimentConfig, estimator)
    export_dir = export_model(estimator, MetaMD.MODEL_DIR, sub_dir='/my_export')
    predict_input(export_dir)
    print("====================================================================================")
    predict_input2(MetaMD.MODEL_DIR)


if __name__ == "__main__":
    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main)
    main(None)


