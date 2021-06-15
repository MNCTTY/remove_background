## Installation

The easiest way to deploy the service is via docker-compose, so you have to install Docker and docker-compose first. Here's a brief instruction for Ubuntu:

#### Docker installation

*	Install Docker:
```bash
$ sudo apt-get update
$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
$ sudo apt-key fingerprint 0EBFCD88
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
$ sudo usermod -aG docker $(whoami)
```
In order to run docker commands without sudo you might need to relogin.
*   Install docker-compose:
```
$ sudo curl -L "https://github.com/docker/compose/releases/download/1.25.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
$ sudo chmod +x /usr/local/bin/docker-compose
```

*   (Optional) If you're planning on using CUDA run these commands:
```
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo apt-key add -
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
$ sudo apt-get update
$ sudo apt-get install nvidia-container-runtime
```
Add the following content to the file **/etc/docker/daemon.json**:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
Restart the service:
```bash
$ sudo systemctl restart docker.service
``` 

## u2net

### Inference run sample code:

```
from u2net.prediction_class import Predict

image = '/home/ubuntu/removebg/test.jpg'
model_name = 'u2net/saved_models/u2net/u2net.pth'

prediction = Predict(model_name)
prediction.predict(image)
```

### Running as web-service:

```
$ docker-compose up -d --force-recreate
```

Go to http://localhost:5000, press upload button, choose an image to upload and click "Submit". After it is uploaded, the LFM starts its processing. After that service will render the uploaded image and the result.

### Contents of repository

**u2net** - this directory containts all code for the model and test images

**uploads** - users upload their images to this directory, also app puts result masks here
