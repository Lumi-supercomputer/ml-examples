[lumi-eap] `tf.distribute`

We are using here the image [amdih/tensorflow:rocm5.0-tf2.7-dev](https://www.amd.com/en/technologies/infinity-hub/tensorflow).

That image has tensorflow and horovod installed with no dependencies. So, the dependecies need to be installed.
Here we use a virtual environment and installed all of following except what was there from tensorflow and horovod:
```
absl-py==1.0.0
asn1crypto==0.24.0
astunparse==1.6.3
cachetools==5.0.0
certifi==2018.1.18
chardet==3.0.4
cloudpickle==2.0.0
cryptography==2.1.4
flatbuffers==2.0
gast==0.5.3
google-auth==2.6.4
google-auth-oauthlib==0.5.1
google-pasta==0.2.0
grpcio==1.44.0
h5py==3.6.0
horovod==0.23.0
idna==2.6
importlib-metadata==4.11.3
keras==2.7.0
Keras-Preprocessing==1.1.2
keyring==10.6.0
keyrings.alt==3.0
libclang==13.0.0
Markdown==3.3.6
numpy==1.22.3
oauthlib==3.2.0
opt-einsum==3.3.0
protobuf==3.20.0
psutil==5.9.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycrypto==2.6.1
PyGObject==3.26.1
python-apt==1.6.5+ubuntu0.7
python-hostlist==1.21
pyxdg==0.25
PyYAML==6.0
requests==2.18.4
requests-oauthlib==1.3.1
rsa==4.8
SecretStorage==2.3.1
six==1.16.0
ssh-import-id==5.7
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
tensorflow @ file:///tmp/tensorflow_pkg/tensorflow-2.7.0-cp39-cp39-linux_x86_64.whl
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.24.0
termcolor==1.1.0
typing_extensions==4.1.1
unattended-upgrades==0.1
urllib3==1.22
Werkzeug==2.1.1
wrapt==1.14.0
zipp==3.8.0
```
The venv should be sourced before running.

We use here `tf.distribute.MultiWorkerMirroredStrategy`, which in turns uses the `SlurmClusterResolver` to setup the distributed training.
It looks that `SlurmClusterResolver` hasn't been ported to ROCM. Also, in some case it fails to get the names of LUMI's compute nodes.
Some changes were necessary:
1. The environment variable `LUMI_VISIBLE_DEVICES` is used to [set the visible devices](https://github.com/Lumi-supercomputer/ml-examples/blob/3d4c33a0336d5ad4c60f28417a06719a5af6350a/tensorflow/tfdist/slurm_cluster_resolver.py#L149)
2. `python-hostlist` is used to [find the hostnames](https://github.com/Lumi-supercomputer/ml-examples/blob/3d4c33a0336d5ad4c60f28417a06719a5af6350a/tensorflow/tfdist/slurm_cluster_resolver.py#L72-L87)

The file [`slurm_cluster_resolver.py`](slurm_cluster_resolver.py) needs to [replace the original on the tensorflow installation](https://github.com/Lumi-supercomputer/ml-examples/blob/3d4c33a0336d5ad4c60f28417a06719a5af6350a/tensorflow/tfdist/run-singularity.sh#L16).

All the steps have been put together on the batch script [run-singularity.sh](run-singularity.sh).
