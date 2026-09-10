# AE 5510: Security of Safety Critical Systems

## Physical Domain Attack on Object Detector Deployed on UAS

### Steps to initialize the project

#### Step 1: Create a virtual environment and activate

```bash
$ python3 -m venv myenv
$ source myenv/bin/activate
```

#### Step 2: Install the dependencies

```bash
$ pip install -r requirements.txt
```

#### Step 3: Download the project data

Make sure to download the project data from Google Drive. Please put your project data into the directory `project_data`

```bash
$ mkdir project_data
```
Make sure that your training, validation, and testing data folders are renamed to `train`, `validation`, and `test`. Inside each of these folders, you should have folders with names 'images', and 'labels'. You have to rename each such folder as `images`, and `labels`.