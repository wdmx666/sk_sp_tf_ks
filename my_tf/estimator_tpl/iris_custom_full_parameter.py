import tensorflow as tf
from datetime import datetime
import path,os
from types import SimpleNamespace
from typing import Dict,List,Tuple,Optional,Any


class Parameter:
    def __init__(self, var):
        self.var = var
        self.auto = True

    def _value(self):
        if isinstance(self.var,self.__class__):
            return self.var.value
        return self.var

    @property
    def value(self):
        if self.auto:
            self.var = self._value()
        if not self.var:
            raise ValueError("值不能为空")
        return self.var

    def set_value(self, var=None, auto=False, **kwargs):
        self.auto = auto
        self.var = var
        print(self.__class__.__name__, 'set value->', self.var)
        return self


class ParaPath(Parameter):
    def __init__(self, para: Parameter):
        self.parameter = para
        super(ParaPath, self).__init__(self._value())

    def _value(self):
        return tf.keras.utils.get_file(self.parameter.value.split('/')[-1], self.parameter.value)


class ParaCsvColumnNames(Parameter):
    def __init__(self, feature_names, target):
        self.feature_names: Parameter = feature_names
        self.target: Parameter = target
        super(ParaCsvColumnNames, self).__init__(self._value())

    def _value(self):
        return self.feature_names.value+[self.target.value]


#  #####################数据ETL##########################
#  命名空间中生成了参数的引用对象，并且有函数引用该实例，本质上类似OOP那种方式，只是参数不在属于某个特殊对象，而是公开的。
class MetaETL(SimpleNamespace):
    """命名空间中创建了多个参数实例,这些设置方面的问题，不是在前面解决就是后面解决，是逃不掉的，只是方式和时机不一样
    这些内部的赋值操作，不能放在设置的接口中去否则，导致社会的接口没有通用性，与本应用挂钩紧密，也违背高内聚"""
    train_url = Parameter("http://download.tensorflow.org/data/iris_training.csv")
    train_path = ParaPath(train_url)
    test_url = Parameter("http://download.tensorflow.org/data/iris_test.csv")
    test_path = ParaPath(test_url)
    target = Parameter('species')
    feature_names = Parameter(['sepallength', 'sepalwidth', 'petallength', 'petalwidth'])
    categorical_feature_names = Parameter([])
    numeric_feature_names = Parameter(feature_names)
    csv_column_names = ParaCsvColumnNames(feature_names,target)
    species = Parameter(['setosa', 'versicolor', 'virginica'])
    csv_column_defaults = Parameter([[0.0], [0.0], [0.0], [0.0], [0]])
    feature_defaults = Parameter([[0.0], [0.0], [0.0], [0.0]])


# transform
def parse_csv_row(csv_row)->Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    columns = tf.decode_csv(csv_row, record_defaults=MetaETL.csv_column_defaults.value)
    features = dict(zip(MetaETL.csv_column_names.value, columns))
    target = features.pop(MetaETL.target.value)
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
        dataset = dataset.shuffle(buffer_size=int(2 * batch_size + 1))
    dataset = dataset.map(parse_csv_row)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    return dataset


def get_feature_columns()->Dict[str, Any]:
    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in MetaETL.numeric_feature_names.value}
    feature_columns = {}
    if numeric_columns is not None:
        feature_columns.update(numeric_columns)
    return feature_columns


# ############################定义服务导出的输入函数############################################
def csv_serving_input_fn():
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')
    receiver_tensor = {'csv_rows': rows_string_tensor}
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=MetaETL.feature_defaults.value)
    features = dict(zip(MetaETL.feature_names.value, columns))
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


class ParaTotalSteps(Parameter):
    def __init__(self, train_size, batch_size, num_epochs):
        self._train_size: Parameter = train_size
        self._batch_size: Parameter = batch_size
        self._num_epochs: Parameter = num_epochs
        super(ParaTotalSteps, self).__init__(self._value())

    def _value(self):
        return (self._train_size.value / self._batch_size.value) *self._num_epochs.value



class MetaMD(SimpleNamespace):
    train_size = Parameter(1200)
    num_epochs = Parameter(100)
    batch_size = Parameter(20)
    eval_after_sec = Parameter(1)
    max_steps = ParaTotalSteps(train_size, batch_size, num_epochs)
    model_name = Parameter('my_custom_iris')
    model_dir = Parameter(path.Path('trained_models/{}'.format(model_name.value)).makedirs_p())
    export_dir = Parameter(path.Path(model_dir.value + "/export/estimate").makedirs_p())

    @classmethod
    def hparams(cls):
        return tf.contrib.training.HParams(
            feature_columns=list(get_feature_columns().values()),
            hidden_units=[10, 10],
            n_classes=3,
            learning_rate=0.01,
            max_steps=cls.max_steps.value)

    @classmethod
    def run_config(cls):
        return tf.estimator.RunConfig(tf_random_seed=19830610, model_dir=cls.model_dir.value)

    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


# 得深入去了解Estimator要什么，而不是告诉程序员一个概念就可以
def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params.feature_columns)
    for units in params.hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params.n_classes, activation=None)
    predicted_classes = tf.argmax(logits, 1)
    # 在预测模式下要计算操作节点，主要是包括类别
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_ids': predicted_classes[:, tf.newaxis], 'probabilities': tf.nn.softmax(logits), 'logits': logits}
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


def create_estimator(model, run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model, params=hparams, config=run_config)
    print("Estimator Type: {}".format(type(estimator)))
    return estimator


# 这算中间状态量没必要公开设置口

# 这算中间状态量没必要公开设置口
def config_experiment(meta_etl, meta_md):
    cfg = SimpleNamespace()
    cfg.train_input_tr_fn = lambda: csv_input_fn(meta_etl.train_path.value, mode=tf.estimator.ModeKeys.TRAIN,
                                                 skip_header_lines=1, batch_size=meta_md.batch_size.value,
                                                 num_epochs=meta_md.num_epochs.value)
    cfg.eval_input_tr_fn = lambda: csv_input_fn(meta_etl.test_path.value, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                         batch_size=meta_md.batch_size.value, num_epochs=meta_md.num_epochs.value)
    cfg.train_spec = tf.estimator.TrainSpec(input_fn=cfg.train_input_tr_fn, max_steps=meta_md.max_steps.value,hooks=None)
    cfg.eval_spec = tf.estimator.EvalSpec(input_fn=cfg.eval_input_tr_fn, steps=None, #throttle_secs=MetaMD.EVAL_AFTER_SEC
        exporters=[tf.estimator.LatestExporter(name="estimate", serving_input_receiver_fn=csv_serving_input_fn,as_text=True)])
    cfg.train_input_eval_fn = lambda: csv_input_fn(meta_etl.train_path.value, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                                   batch_size=meta_md.batch_size.value, num_epochs=meta_md.num_epochs.value)
    cfg.eval_input_eval_fn = lambda: csv_input_fn(meta_etl.test_path.value, mode=tf.estimator.ModeKeys.EVAL, skip_header_lines=1,
                                                   batch_size=meta_md.batch_size.value, num_epochs=meta_md.num_epochs.value)
    return cfg



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




def run_pipe(argv):

    estimator = create_estimator(my_model, MetaMD.run_config(), MetaMD.hparams())
    ExperimentConfig = config_experiment(MetaETL, MetaMD)
    run_experiment(ExperimentConfig, estimator)
    export_dir = export_model(estimator, MetaMD.model_dir.value, sub_dir='/my_export')
    predict_input(export_dir)
    print("====================================================================================")


if __name__ == "__main__":
    #tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main)
    run_pipe(None)


