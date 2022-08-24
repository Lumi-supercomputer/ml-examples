These are the instructions to setup the dependencies needed for the TensorFlow installation that comes with [rocm/tensorflow:rocm5.0-tf2.7-dev](https://hub.docker.com/layers/tensorflow/rocm/tensorflow/rocm5.0-tf2.7-dev/images/sha256-664fbd3e38234f5b4419aa54b2b81664495ed0a9715465678f2bc14ea4b7ae16) on LUMI.

## Getting the image
```
mkdir $SCRATCH/tmp-singularity
SINGULARITY_TMPDIR=$SCRATCH/tmp-singularity singularity pull docker://rocm/tensorflow:rocm5.0-tf2.7-dev
```
The temporary directory `$SCRATCH/tmp-singularity` can be removed after the image has been created.

## Preparing the virtual environment
In the image, most of the dependencies of TensorFlow are missing. Here we create a virtual environment too install them.
Howerver, there's a problem with that: The image doesn't contain a package from the ubuntu repository that gives the
support for creating virtual environments. As a result we do a litle trick to install the dependencies that we need:

On the host (outside of the container):
```bash
module load cray-python
python -m venv tf2.7_rocm5.0_env --system-site-packages
. tf2.7_rocm5.0_env/bin/activate
pip install --upgrade pip
deactivate
```
 * Edit `tf2.7_rocm5.0_env/bin/pip` and remove the `/pfs/lustrep*` if present.
 * Edit `tf2.7_rocm5.0_env/bin/activate` and remove `/pfs/lustrep*` if present.

Now run the container and fix the `python` executable of the virtual environment:
```bash
singularity exec -B $SCRATCH:$SCRATCH $SCRATCH/tensorflow_rocm5.0-tf2.7-dev.sif bash
rm tf2.7_rocm5.0_env/bin/python
ln -s /usr/bin/python tf2.7_rocm5.0_env/bin/python
```

Now the dependencies can be installed:
```bash
. tf2.7_rocm5.0_env/bin/activate
pip install -r requirements.txt
```
Where `requirements.txt` is a file containing the following:
```bash
absl-py==1.2.0
asn1crypto==0.24.0
astunparse==1.6.3
cachetools==5.2.0
certifi==2018.1.18
chardet==3.0.4
charset-normalizer==2.1.1
cryptography==2.1.4
flatbuffers==2.0
gast==0.4.0
google-auth==2.11.0
google-auth-oauthlib==0.4.6
google-pasta==0.2.0
grpcio==1.47.0
h5py==3.7.0
# horovod==0.23.0
idna==2.6
importlib-metadata==4.12.0
keras==2.7.0
Keras-Preprocessing==1.1.2
keyring==10.6.0
keyrings.alt==3.0
libclang==13.0.0
Markdown==3.4.1
MarkupSafe==2.1.1
numpy==1.23.2
oauthlib==3.2.0
opt-einsum==3.3.0
protobuf==3.19.4
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycodestyle==2.9.1
pycrypto==2.6.1
PyGObject==3.26.1
python-apt==1.6.5+ubuntu0.7
pyxdg==0.25
requests==2.28.1
requests-oauthlib==1.3.1
rsa==4.9
SecretStorage==2.3.1
six==1.16.0
ssh-import-id==5.7
tensorboard==2.8.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.1
# tensorflow @ file:///tmp/tensorflow_pkg/tensorflow-2.7.0-cp39-cp39-linux_x86_64.whl
tensorflow-estimator==2.7.0
tensorflow-io-gcs-filesystem==0.24.0
termcolor==1.1.0
typing_extensions==4.3.0
unattended-upgrades==0.1
urllib3==1.22
Werkzeug==2.2.2
wrapt==1.14.1
zipp==3.8.1
wheel<1.0,>=0.32.0
psutil==5.9.1
PyYAML==6.0
cloudpickle==2.1.0
```
