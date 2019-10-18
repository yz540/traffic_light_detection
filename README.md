# 1. [Install Anaconda for python](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart): 

 - find the download page and copy the link

 - use curl to download the link, for instance:

```

curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh

```

 - Verify the Data Integrity of the Installer

```

sha256sum Anaconda3-2019.10-Linux-x86_64.sh

```

The output should be:

```

46d762284d252e51cd58a8ca6c8adc9da2eadc82c342927b2f66ed011d1d8b53  Anaconda3-2019.10-Linux-x86_64.sh

```

 - Run the Anaconda Script:

 - Activate Installation

```cd anaconda3/

```

 - Test Installation

```conda list```



 - create and activate anaconda env:

```

(base) yanz@yanz-VirtualBox:~/anaconda3$ conda env list

# conda environments:

#

base                  *  /home/yanz/anaconda3

train_env                /home/yanz/anaconda3/envs/train_env



(base) yanz@yanz-VirtualBox:~/anaconda3$ conda activate train_env

(train_env) yanz@yanz-VirtualBox:~/anaconda3$ 

```

# 2. [Label Images with predefined labels](https://github.com/tzutalin/labelImg)

The images manually annotated were labelled with LabelImg. Once the labelImg is launched, the labels should be labelled according to the labels are defined in the [label.pbtxt](https://github.com/yz540/traffic_light_detection/blob/master/label.pbtxt)

Install LabelImg:

```

git clone https://github.com/tzutalin/labelImg.git

cd labelImg

pip install --upgrade pyqt5

pip install --upgrade lxml

sudo pip3 install -r requirements/requirements-linux-python3.txt

make qt5py3

python3 labelImg.py

```



# 3. [Create Tensorflow record from our dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
In order to train the model using the object detection API the images needs to be fed as a TensorFlow Record. 
If you want to use existing tf records, you can skip the labelling and creating tf record step. But make sure you use their corresponding label file.

 - If tensorflow not yet installed:

```

pip install --upgrade pip

pip install tensorflow==1.14.0

python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"

```

 - If tensorflow object detection API not yet installed:

```

pip install --user Cython

pip install --user contextlib2

pip install --user pillow

pip install --user lxml

pip install --user jupyter

pip install --user matplotlib

```

 - Install the coco api

```

git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI

make

cp -r pycocotools <path_to_tensorflow>/models/research/

```

 - Manual protobuf-compiler

```

wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip

unzip protobuf.zip

# From tensorflow/models/research/

protoc object_detection/protos/*.proto --python_out=.

```

 - Add Libraries to PYTHONPATH


```

# From tensorflow/models/research/, if later in run time, the object detection module not found error occurs, you can run the following command again

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

```

 - Test the installation, you should see OK in the result.

```python object_detection/builders/model_builder_test.py```

 - Then create tf record using the following command, you can change the path to ```--data_dir=, --labels_dir=, --labels_map_path=, --output_path``` to your own local path.

```
python create_tf_record.py  --data_dir=sim_images --labels_dir=sim_labels --labels_map_path=label.pbtxt --output_path=sim_tf_records --split_train_test=0.25
```
The create_tf_record.py can be found [here](https://github.com/yz540/traffic_light_detection/blob/master/create_tf_record.py)
If you have errors run the ``` python create_tfrecord.py```, you might need to install some missing modules.

```
pip install tqdm
pip install sklearn
```

The result should be as follows:
```
Total samples: 143
WARNING:tensorflow:From create_tf_record.py:73: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

W1018 10:43:08.180608 140231217743680 deprecation_wrapper.py:119] From create_tf_record.py:73: The name tf.python_io.TFRecordWriter is deprecated. Please use tf.io.TFRecordWriter instead.

Converting:   0%|                                  | 0/107 [00:00<?, ? images/s]/home/yanz/trainmodel/models/research/object_detection/utils/dataset_util.py:79: FutureWarning: The behavior of this method will change in future versions. Use specific 'len(elem)' or 'elem is not None' test instead.
  if not xml:
Converting: 100%|███████████████████████| 107/107 [00:00<00:00, 255.49 images/s]
TF record file for training created with 107 samples: sim_tf_records_train.record
Converting: 100%|█████████████████████████| 36/36 [00:00<00:00, 266.20 images/s]
TF record file for validation created with 36 samples: sim_tf_records_eval.record

```

# 4. Train the traffic light classification model

The [object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) is the starting point.

 - Download a model from the [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

For instance, the ssdlite_mobilenet_v2_coco:

``` 
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

```


 - [Configure the pipeline.config file](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)

Make the following change to the pipeline.config:

```

	num_classes: 90 to 3

	max_detections_per_class: 10

	max_total_detections: 10

	fine_tune_checkpoint: "gs://udacity-capstone-253721-mlengine/model/model.ckpt"

	num_steps: 500000

	num_examples: 36

	label_map_path: "gs://udacity-capstone-253721-mlengine/data/label.pbtxt"	

	train_input_reader and eval_input_reader input_path to the record path, for instance, "gs://udacity-capstone-253721-mlengine/data/sim_tf_records_train.record"

```

The num_examples is the number of evaluation examples depends on the number of images times the percentage of your evaluation when you created your record.

The label_map_path and input_path should be changed according to your training mode: local or GCP. The above example is on GCP.


 - Train the model based on the chosen model.

## Train locally:

```

PIPELINE_CONFIG_PATH={path to pipeline config file}

MODEL_DIR={path to fine-tuned model directory}, 

NOTE: use a different directory from the existing model, otherwise the training will not start. For instance, 

mkdir /home/yanz/trainmodel/ssdlite_mobilenet_v2_coco_2018_05_09/new_train

MODEL_DIR=/home/yanz/trainmodel/ssdlite_mobilenet_v2_coco_2018_05_09/new_train



NUM_TRAIN_STEPS=50000

SAMPLE_1_OF_N_EVAL_EXAMPLES=1

python object_detection/model_main.py     --pipeline_config_path=${PIPELINE_CONFIG_PATH}     --model_dir=${MODEL_DIR}     --num_train_steps=${NUM_TRAIN_STEPS}     --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES     --alsologtostderr 

```



## [Train with GCP from local machine](https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction):

On local machine:

 - Install Google cloud SDK:

``` 

snap install google-cloud-sdk --classic

gcloud init

```

 - Packaging: The script are currently stored in your computer, not on the cloud. At running time you will send them to the cloud and run it there. This will be sent in the form of packages that you create by: 

```

# From tensorflow/models/research/

 bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools

 python setup.py sdist

 (cd slim && python setup.py sdist)

```

 - Upload your data and pre-trained model to your bucket: you can either use the command line with gsutil cp ... or the web GUI on your buckets page.

 - Modify and upload your [pipeline.config](https://github.com/yz540/traffic_light_detection/blob/master/pipeline.config): change the paths for the model and data to the corresponding location in your bucket in the form gs://PRE-TRAINED_MODEL_DIR and gs://DATA_DIR

 - Define or redefine the following environment variable in your terminal on local machine. You can find env.yaml [here](https://github.com/yz540/traffic_light_detection/blob/master/env.yaml): 

```

PROJECT_ID=$(gcloud config list project --format "value(core.project)") 

BUCKET_NAME=${PROJECT_ID}-mlengine 

NUM_TRAIN_STEPS=50000

SAMPLE_1_OF_N_EVAL_EXAMPLES=1

JOB_NAME=test

OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME

MODEL_DIR=udacity-capstone-253721-mlengine/sim

PATH_TO_LOCAL_YAML_FILE=/home/yanz/trainmodel/env.yaml

PIPELINE_CONFIG_PATH="udacity-capstone-253721-mlengine/model/pipeline.config"

```

On local machine:

 - Create YAML configuration file: this file describes the GPUs setup you will use on the cloud. You can just create a text file with the following content:

```

 trainingInput:

   runtimeVersion: "1.12"

   scaleTier: CUSTOM

   masterType: standard_gpu

   workerCount: 9

   workerType: standard_gpu

   parameterServerCount: 3

   parameterServerType: standard

```

 - Launch the training task:

```

cd tensorflow/models/research/ 

gcloud ml-engine jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S`      --runtime-version 1.12      --job-dir=gs://${MODEL_DIR}      --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz      --module-name object_detection.model_main      --region us-central1      --config ${PATH_TO_LOCAL_YAML_FILE}      --      --model_dir=gs://${MODEL_DIR}      --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}

```

Copy trained model to local:

```gsutil cp -r gs://udacity-capstone-253721-mlengine/sim /home/yanz/trainmodel/result```





# Export the model for tensorflow 1.4:



If not installed anaconda:

```

curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh 

sha256sum Anaconda3-2019.03-Linux-x86_64.sh  

bash Anaconda3-2019.03-Linux-x86_64.sh

```

 - Install corresponding env on local machine:

```

conda create -n tensorflow_1.4 

source ~/.bashrc 

conda create -n tensorflow_1.4 python=2.7 

conda activate tensorflow_1.4 

pip install tensorflow==1.4.0 

conda install pillow lxml matplotlib 

```

 - Install object detection for tensorflow 1.4. These steps are simillar to the ones above. The only difference is to find the corresponding version for tensorflow 1.4.

```

git clone https://github.com/tensorflow/models.git temp 

cd temp  

git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7 


cd .. 

mkdir exporter

cp -r temp/research/object_detection exporter/object_detection 

cp -r temp/research/slim exporter/slim 


cd temp/research

git clone https://github.com/cocodataset/cocoapi.git 

cd cocoapi/PythonAPI 

make 

cd protobuf-3.4.0/ 

conda env list 

conda activate tensorflow_1.4 

```

 - Install the corresponding version of Protocol buffer:
```

wget "https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-linux-x86_64.zip" 

unzip protoc-3.4.0-linux-x86_64.zip "/home/yz540/exporter/" 

mv protoc-3.4.0-linux-x86_64.zip exporter/ 

cd exporter/ 

protoc object_detection/protos/*.proto --python_out=.   


export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
```

 - Export the model in tf 1.4:
```

mkdir /home/yz540/ssd_mobilenet_v1_coco_2018_01_28/new/sim_new_export_model_v1.4/                                                                                                                                                                                                                  

EXPORT_DIR="/home/yz540/ssd_mobilenet_v1_coco_2018_01_28/new/sim_new_export_model_v1.4/" 

TRAINED_CKPT_PREFIX=/home/yz540/ssd_mobilenet_v1_coco_2018_01_28/new/sim_new/sim/model.ckpt-500013  

PIPELINE_CONFIG_PATH=/home/yz540/ssd_mobilenet_v1_coco_2018_01_28/pipeline.config

INPUT_TYPE=image_tensor 

python object_detection/export_inference_graph.py     --input_type=${INPUT_TYPE}     --pipeline_config_path=${PIPELINE_CONFIG_PATH}     --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX}     --output_directory=${EXPORT_DIR} 

```
