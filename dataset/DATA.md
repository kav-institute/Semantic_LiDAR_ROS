# Setup

[https://zenodo.org/records/14677379](https://zenodo.org/records/14677379)

Donwload the asset pack from Zenodo [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14677379.svg)](https://doi.org/10.5281/zenodo.14677379) and extrackt the files to the dataset folder.
Ensure the following structure:

```
dataset
└───models
└───recordings
└───vehicle
└───config.json
```

## Sample Recordings
We provide three sampe recordings.
You can select the recording by changing the path in the config.json
```json
{
    "METADATA_PATH": "/home/appuser/data/recordings/Test_Scene_0000/OS-2-128-992317000331-2048x10.json",
    "OSF_PATH": "/home/appuser/data/recordings/Test_Scene_0000/OS-2-128-992317000331-2048x10.osf",
    ...
    "STREAM_SENSOR": false # ensure this is set to false for recordings
}
```
> [!NOTE]
> You need the path in the docker container and the content of folder "dataset" is mapped to "/home/appuser/data/"

## Sensor Stream
Our demo is tailored to a Ouster OS2-128 Rev 7. We use the ouster-sdk==0.12.0. 
> [!NOTE]
> The release version of the Ouster SDK and OusterStudio had a major revision!
> 
> Ensure you use a suitable version for OusterStudio, Ouster SDK, and sensor firmware (we use: "ousteros-image-prod-aries-v2.5.2+20230714195410").
> 
> We use the LEGACY config for our Ouster recodings and record at 2048@10Hz.

To stream sensor data in our demo in real time you can change the config.json:
```json
{
    "METADATA_PATH": ".../metadata.json", # keep in mind that you have to initialize the sensor and create a meta data json!
    "SENSOR_IP": "fe80::4ce2:9d59:9f63:a9e0",  # set your ip v6!
    "SENSOR_PORT": 7502, # set your port
    "STREAM_SENSOR": false # ensure this is set to true for sensor streaming
}
```

## Model Zoo
You can set the used model by:
```json
{
    "MODEL_PATH": "/home/appuser/data/models/resnet34_ANMP/model_final.pth",
    "MODEL_CONFIG": "/home/appuser/data/models/resnet34_ANMP/config.json",
}
```

We support only models including attention, multi_scale_meta and surface normals!
