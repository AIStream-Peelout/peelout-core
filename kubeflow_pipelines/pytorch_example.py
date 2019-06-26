# https://www.kubeflow.org/docs/pipelines/sdk/dsl-overview/
import kfp
from kfp import dsl

@kfp.dsl.pipeline(
  name='PyTorch example',
  description='A simple PyTorch pipeline in Kubeflow'
)
def voice_make(output, project,
    evaluation='gs://ml-pipeline-playground/flower/eval100.csv',
    train='gs://ml-pipeline-playground/flower/train200.csv',
    schema='gs://ml-pipeline-playground/flower/schema.json',
    learning_rate=0.1,
    hidden_layer_size='100,50',
    steps=2000,
    target='label',
    workers=0,
    pss=0,
    preprocess_mode='local',
    predict_mode='local',
):
    output_template = str(output) + '/{{workflow.uid}}/{{pod.name}}/data'

    # set the flag to use GPU trainer
    use_gpu = False

    preprocess = dataflow_tf_transform_op(
        training_data_file_pattern=train,
        evaluation_data_file_pattern=evaluation,
        schema=schema,
        gcp_project=project,
        run_mode=preprocess_mode,
        preprocessing_module='',
        transformed_data_dir=output_template
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    training = kubeflow_tf_training_op(
        transformed_data_dir=preprocess.output,
        schema=schema,
        learning_rate=learning_rate,
        hidden_layer_size=hidden_layer_size,
        steps=steps,
        target=target,
        preprocessing_module='',
        training_output_dir=output_template
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    if use_gpu:
        training.image = 'gcr.io/ml-pipeline/ml-pipeline-kubeflow-tf-trainer-gpu:d4960d3379af4735fd04dc7167fab5fff82d0f22',
        training.set_gpu_limit(1)

    prediction = dataflow_tf_predict_op(
        data_file_pattern=evaluation,
        schema=schema,
        target_column=target,
        model=training.output,
        run_mode=predict_mode,
        gcp_project=project,
        predictions_dir=output_template
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))

    confusion_matrix = confusion_matrix_op(
        predictions=prediction.output,
        output_dir=output_template
    ).apply(gcp.use_gcp_secret('user-gcp-sa'))
