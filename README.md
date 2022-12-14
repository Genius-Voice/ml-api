# From machine learning model to API

The main goal of this repo is to provide a simple example of how to serve your machine-learning model as an API

## How to install all requirements

### 1. Git clone project

```
git clone https://github.com/Genius-Voice/ml-api.git
```

### 2. Go to project dir

```
cd ml-api
```

### 3. Set up a virtual Python environment with [venv](https://docs.python.org/3/library/venv.html)

#### For MacOS users

```
pip install virtualenv
virtualenv venv
source venv/bin/activate
```

#### For Windows users

```
pip install virtualenv
virtualenv venv
venv\Scripts\activate
```

### 4. Install pip libraries

```
pip install -r requirements.txt
```

### 5. train model

```
Open train.ipynb and run this notebook
Wait until the model is finished training and the objects are saved..
```

### 6. DONE! 
```
You can now run the FastAPI app
```

# How to serve a model using fastAPI

### 1. Navigate to the project dir

```
cd fastapi
```

### 2. Run main.py

```
uvicorn main:app --port 8000 --reload
```

### 3. navigate to

```
http://127.0.0.1:8000/docs or http://localhost:8000/docs
```

### Example prediction for "the flight was awesome"

![](images/fastapi_response.png)


### How to bring FastAPI down

```
CTRL + C
```


## Run FastAPI in Docker container

### 1. Docker build image
Make sure the model is trained and saved in the models directory before proceeding.

```shell
docker build -t mlapp ./fastapi
``` 
wait...

###  2. Docker run container

#### For Windows Users
```shell
docker run -p 8000:8000 -v %cd%/models:/app/models -v %cd%/fastapi:/app mlapp
```

#### For MacOS Users
```shell
docker run -p 8000:8000 -v "$(pwd)"/models:/app/models -v "$(pwd)"/fastapi:/app mlapp
```

###  3. Test the model
- Open a browser
- Go to localhost:8000/docs

### Handy Docker commands

##### 1. Show running containers
```shell
docker ps
```

##### 2. Show Docker images
```shell
docker images
```

##### 3. stop container
```shell
docker stop [container-id]
```

##### 4. Remove image
```shell
docker rmi [image-id]
```

or

```shell
docker rmi -f [image-id]
```
