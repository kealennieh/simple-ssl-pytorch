## 1. Introduction
This is a simple project with self-supervised learning models, which is built on top of pytorch.

```
.
|
|---configs          # config file
|
|---datasets         # datast api
|
|---docs
|
|---models           # all the parts of models
|    |
|    |
|    |---backbone
|    |
|    |---loss
|    |
|    |---model
|    |
|
|---scripts          # shell scripts to execute
|
|---tools            # python tools to start training, testing, etc
|
|---utils            # useful functions
|
|---README.md
|
|---requirements.txt
|
```

## 2. Usage
```
pip install -r requirements.txt


bash ./scripts/start_train.sh

```


## 3. Model
- [x] SimCLR
- [ ] MoCo
- [ ] BYOL
- [ ] SimSiam


## 4. Other


## 5. Reference
Thanks for the following great project. Without the support of those projects, this repository wouldn't exist.
1. [pyssl](https://github.com/giakou4/pyssl)  a repository with many Self-Supervised Learning methods.
