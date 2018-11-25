
import tensorflow as tf
import path
from types import SimpleNamespace

from my_tf.estimator.iris_premade import get_feature_columns, MetaRaw,csv_input_fn,maybe_download


# 得深入去了解Estimator要什么，而不是告诉程序员一个概念就可以
def my_model(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params.feature_columns)
    for units in params.hidden_units:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params.n_classes, activation=None)
    predicted_classes = tf.argmax(logits, 1)
    # 在预测模式下要计算操作节点，主要是包括类别
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'class_ids': predicted_classes[:, tf.newaxis],'probabilities': tf.nn.softmax(logits),'logits': logits}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 计算评价量，损失函数和其它评价函数都属于评价准则，只是用途有所区别
    # 损失
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    average_loss = tf.reduce_mean(loss)
    metrics = {'accuracy': accuracy,'average_loss': tf.metrics.mean(average_loss)}

    tf.summary.scalar('accuracy', accuracy[1])

    # 验证模式下
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # 训练模式下，要运行优化节点，计算损失.
    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=my_model, params=hparams, config=run_config)
    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")
    return estimator


def csv_serving_input_fn():
    rows_string_tensor = tf.placeholder(dtype=tf.string, shape=[None], name='csv_rows')
    receiver_tensor = {'csv_rows': rows_string_tensor}
    row_columns = tf.expand_dims(rows_string_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=MetaRaw.FEATURE_DEFAULT)
    features = dict(zip(MetaRaw.FEATURE_NAMES, columns))
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)


def make_experiment_config():
    MetaMD = SimpleNamespace()
    MetaMD.TRAIN_SIZE = 1500
    MetaMD.NUM_EPOCHS = 100
    MetaMD.BATCH_SIZE = 10
    MetaMD.EVAL_AFTER_SEC = 15
    MetaMD.TOTAL_STEPS = (MetaMD.TRAIN_SIZE/MetaMD.BATCH_SIZE)*MetaMD.NUM_EPOCHS
    MetaMD.MODEL_NAME = 'my_custom_iris'
    MetaMD.MODEL_DIR = path.Path('trained_models/{}'.format(MetaMD.MODEL_NAME)).makedirs_p()
    MetaMD.EXPORT_DIR = path.Path(MetaMD.MODEL_DIR + "/export/estimate").makedirs_p()
    print(MetaMD.MODEL_DIR, MetaMD.EXPORT_DIR)

    hparams = tf.contrib.training.HParams(
        feature_columns=list(get_feature_columns().values()),
        num_epochs=MetaMD.NUM_EPOCHS,
        batch_size=MetaMD.BATCH_SIZE,
        hidden_units=[10, 10],
        n_classes=3,
        max_steps=MetaMD.TOTAL_STEPS,
        learning_rate=0.05
    )

    run_config = tf.estimator.RunConfig(
        log_step_count_steps=5000,
        tf_random_seed=19830610,
        model_dir=MetaMD.MODEL_DIR
    )
    MetaMD.HPARAMS = hparams
    MetaMD.RUN_CONFIG = run_config
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++=")
    print(hparams)
    print("Model Directory:", run_config.model_dir)
    print("")
    print("Dataset Size:", MetaMD.TRAIN_SIZE)
    print("Batch Size:", MetaMD.BATCH_SIZE)
    print("Steps per Epoch:",MetaMD.TRAIN_SIZE/MetaMD.BATCH_SIZE)
    print("Total Steps:", MetaMD.TOTAL_STEPS)
    print("That is 1 evaluation step after each",MetaMD.EVAL_AFTER_SEC," training seconds")
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
        exporters=[tf.estimator.LatestExporter(name="estimate", serving_input_receiver_fn=csv_serving_input_fn,as_text=True)],
        steps=None,
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


def predict_input(MetaMD):
    import os
    saved_model_dir = MetaMD.EXPORT_DIR + "/" + os.listdir(path=MetaMD.EXPORT_DIR)[-1]

    print(saved_model_dir)

    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )


    output = predictor_fn({'csv_rows': ["0.5,1,2,4"]})
    print(output)
    print()

def main(argv):
    from my_tf.estimator.iris_premade import run_experiment
    MetaMD = make_experiment_config()
    EXP_CFG = config_experiment(MetaMD)
    estimator = create_estimator(MetaMD.RUN_CONFIG, MetaMD.HPARAMS)
    run_experiment(MetaMD, EXP_CFG, estimator)
    predict_input(MetaMD)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    #tf.app.run(main2)
    main(None)


