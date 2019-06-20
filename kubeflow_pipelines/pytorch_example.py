#https://www.kubeflow.org/docs/pipelines/sdk/dsl-overview/
import kfp
from kfp import dsl

@kfp.dsl.pipeline(
  name='PyTorch example',
  description='A simple PyTorch pipeline in Kubeflow'
)