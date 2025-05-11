# GCP-LAB-Deploy-a-BigQuery-ML-Customer-Churn-Classifier-to-Vertex-AI-for-Online-Predictions
Deploy a BigQuery ML Customer Churn Classifier to Vertex AI for Online Predictions

#### Overview
In this lab, you will train, tune, evaluate, explain, and generate batch and online predictions with a BigQuery ML XGBoost model. You will use a Google Analytics 4 dataset from a real mobile application, Flood it! (Android app, iOS app), to determine the likelihood of users returning to the application. You will generate batch predictions with your BigQuery ML model as well as export and deploy it to Vertex AI for online predictions using the Vertex AI Python SDK.

BigQuery ML lets you train and do batch inference with machine learning models in BigQuery using standard SQL queries faster by eliminating the need to move data with fewer lines of code.

Vertex AI is Google Cloud's complimentary next generation, unified platform for machine learning development. By developing and deploying BigQuery ML machine learning solutions on Vertex AI, you can leverage a scalable online prediction service and MLOps tools for model retraining and monitoring to significantly enhance your development productivity, the ability to scale your workflow and decision making with your data, and accelerate time to value.

BigQuery ML Vertex AI Lab Architecture diagram Note: BQML is now BigQuery ML.

This lab is inspired by and extends Churn prediction for game developers using Google Analytics 4 (GA4) and BigQuery ML. Read the blog post and accompanying tutorial for additional depth on this use case and BigQuery ML.

In this lab, you will go one step further and focus on how Vertex AI extends BigQuery ML's capabilities through online prediction so you can incorporate both customer churn predictions into decision making UIs such as Looker dashboards but also online predictions directly into customer applications to power targeted interventions such as targeted incentives.


##### Objectives
In this lab, you learn how to:

. Explore and preprocess a Google Analytics 4 data sample in BigQuery for machine learning.
. Train a BigQuery ML XGBoost classifier to predict user churn on a mobile gaming application.
. Tune a BigQuery ML XGBoost classifier using BigQuery ML hyperparameter tuning features.
. Evaluate the performance of a BigQuery ML XGBoost classifier.
. Explain your XGBoost model with BigQuery ML Explainable AI global feature attributions.
. Generate batch predictions with your BigQuery ML XGBoost model.
. Export a BigQuery ML XGBoost model to a Google Cloud Storage bucket.
. Upload and deploy a BigQuery ML XGBoost model to a Vertex AI Prediction Endpoint for online predictions.


### Task 1. Create a Vertex AI Workbench instance

1. In the Google Cloud console, from the Navigation menu (Navigation menu), select Vertex AI.

2. Click Enable All Recommended APIs.

3. On the left-hand side, click Workbench.

4. At the top of the Workbench page, ensure you are in the Instances view.

5. Click add boxCreate New.

6. Configure the Instance:

Name: lab-workbench
Region: Set the region to Region
Zone: Set the zone to Zone
Advanced Options (Optional): If needed, click "Advanced Options" for further customization (e.g., machine type, disk size)

![alt text](images/Task1-1.png)

7. Click Create.

>Note: The instance will take a few minutes to create. A green checkmark will appear next to its name when it's ready.

8. Click Open JupyterLab next to the instance name to launch the JupyterLab interface. This will open a new tab in your browser.

![alt text](images/Task1-2.png)

9. Click the Terminal icon to open a terminal window.

![alt text](images/Task1-3.png)

Your terminal window will open in a new tab. You can now run commands in the terminal to interact with your Workbench instance.



### Task 2. Copy the notebook from a Cloud Storage bucket

1. In your notebook, Copy the below path.

```
gcloud storage cp gs://qwiklabs-gcp-04-8324c7646605-labconfig-bucket/lab_exercise-v1.0.0.ipynb .
```

![alt text](images/Task2-1.png)

2. Open the notebook file lab_exercise.ipynb.

3. In the Define constants section, update your REGION variable with the us-east1.

4. Continue the lab in the notebook, and run each cell by clicking the Run (run button icon) icon at the top of the screen. Alternatively, you can execute the code in a cell with SHIFT + ENTER.

Read the narrative and make sure you understand what's happening in each cell. As you progress through the lab notebook, return back to these instructions to complete the graded exercises.

### Deploy a BigQuery ML user churn propensity model to Vertex AI for online predictions

##### Learning objectives
. Explore and preprocess a Google Analytics 4 data sample in BigQuery for machine learning.
. Train a BigQuery ML (BQML) XGBoost classifier to predict user churn on a mobile gaming application.
. Tune a BQML XGBoost classifier using BQML hyperparameter tuning features.
. Evaluate the performance of a BQML XGBoost classifier.
. Explain your XGBoost model with BQML Explainable AI global feature attributions.
. Generate batch predictions with your BQML XGBoost model.
. Export a BQML XGBoost model to a Google Cloud Storage.
. Upload and deploy a BQML XGBoost model to a Vertex AI Prediction Endpoint for online predictions.

#### Introduction
In this lab, you will train, evaluate, explain, and generate batch and online predictions with a BigQuery ML (BQML) XGBoost model. You will use a Google Analytics 4 dataset from a real mobile application, Flood it! (Android app, iOS app), to determine the likelihood of users returning to the application. You will generate batch predictions with your BigQuery ML model as well as export and deploy it to Vertex AI for online predictions.

BigQuery ML lets you train and do batch inference with machine learning models in BigQuery using standard SQL queries faster by eliminating the need to move data with fewer lines of code. Vertex AI is Google Cloud's complimentary next generation, unified platform for machine learning development. By developing and deploying BQML machine learning solutions on Vertex AI, you can leverage a scalable online prediction service and MLOps tools for model retraining and monitoring to significantly enhance your development productivity, the ability to scale your workflow and decision making with your data, and accelerate time to value.

#### Use case: user churn propensity modeling in the mobile gaming industry

According to a 2019 study on 100K mobile games by the Mobile Gaming Industry Analysis, most mobile games only see a 25% retention rate for users after the first 24 hours, known and any game "below 30% retention generally needs improvement". For mobile game developers, improving user retention is critical to revenue stability and increasing profitability. In fact, Bain & Company research found that 5% growth in retention rate can result in a 25-95% increase in profits. With lower costs to retain existing customers, the business objective for game developers is clear: reduce churn and improve customer loyalty to drive long-term profitability.

Your task in this lab: use machine learning to predict user churn propensity after day 1, a crucial user onboarding window, and serve these online predictions to inform interventions such as targeted in-game rewards and notifications.


#### Setup

```
!pip3 install google-cloud-aiplatform --user
!pip3 install pyarrow==11.0.0 --user
!pip3 install --upgrade google-cloud-bigquery --user
!pip3 install --upgrade google-cloud-bigquery-storage --user
!pip3 install --upgrade google-cloud-storage --user
!pip install db-dtypes
```

![alt text](images/Task2-2.png)

Restart the kernel and ignore the compatibility errors.


##### Define constants

```
# Retrieve and set PROJECT_ID and REGION environment variables.
PROJECT_ID = !(gcloud config get-value core/project)
PROJECT_ID = PROJECT_ID[0]
```

Note: Replace the REGION with the associated region mentioned in the qwiklabs resource panel.

```
BQ_LOCATION = 'US'
REGION = 'us-east1'
```

##### Import libraries¶

```
from google.cloud import bigquery
from google.cloud import aiplatform as vertexai
import numpy as np
import pandas as pd
```


##### Create a GCS bucket for artifact storage

Create a globally unique Google Cloud Storage bucket for artifact storage. You will use this bucket to export your BQML model later in the lab and upload it to Vertex AI.

```
GCS_BUCKET = f"{PROJECT_ID}-bqmlga4"
```

```
!gsutil mb -l $REGION gs://$GCS_BUCKET
```

>Creating gs://qwiklabs-gcp-04-8324c7646605-bqmlga4/...


### Task 3. Create a BigQuery dataset

##### Create a BigQuery dataset


Next, create a BigQuery dataset from this notebook using the Python-based bq command line utility.

This dataset will group your feature views, model, and predictions table together. You can view it in the BigQuery console.


```
BQ_DATASET = f"{PROJECT_ID}:bqmlga4"
```

```
!bq mk --location={BQ_LOCATION} --dataset {BQ_DATASET}
```
>Dataset 'qwiklabs-gcp-04-8324c7646605:bqmlga4' successfully created.

##### Initialize the Vertex Python SDK client

Import the Vertex SDK for Python into your Python environment and initialize it.

```
vertexai.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{GCS_BUCKET}")
```

##### Exploratory Data Analysis (EDA) in BigQuery


This lab uses a public BigQuery dataset that contains raw event data from a real mobile gaming app called Flood it! (Android app, iOS app).

The data schema originates from Google Analytics for Firebase but is the same schema as Google Analytics 4.

Take a look at a sample of the raw event dataset using the query below:

```
%%bigquery --project $PROJECT_ID

SELECT 
    *
FROM
  `firebase-public-project.analytics_153293282.events_*`
    
TABLESAMPLE SYSTEM (1 PERCENT)
```
>Job ID 5570f643-5cd0-4a5e-a4aa-94b5fd89bd1f successfully executed: 100%

> Note: in the cell above, Jupyterlab runs cells starting with %%bigquery as SQL queries.

![alt text](images/Task4-1.png)

Google Analytics 4 uses an event based measurement model and each row in this dataset is an event. View the complete schema and details about each column. As you can see above, certain columns are nested records and contain detailed information such as:
. app_info
. device
. ecommerce
. event_params
. geo
. traffic_source
. user_properties
. items*
. web_info*

This dataset contains 5.7M events from 15K+ users.

```
%%bigquery --project $PROJECT_ID

SELECT 
    COUNT(DISTINCT user_pseudo_id) as count_distinct_users,
    COUNT(event_timestamp) as count_events
FROM
  `firebase-public-project.analytics_153293282.events_*`
```

> Job ID 65c06919-368f-440e-9403-56233ca78354 successfully executed: 100%

![alt text](images/Task4-2.png)

##### Dataset preparation in BigQuery

Now that you have a better sense for the dataset you will be working with, you will walk through transforming raw event data into a dataset suitable for machine learning using SQL commands in BigQuery. Specifically, you will:

Aggregate events so that each row represents a separate unique user ID.
Define the user churn label feature to train your model to prediction (e.g. 1 = churned, 0 = returned).
Create user demographic features.
Create user behavioral features from aggregated application events.

Defining churn for each user
There are many ways to define user churn, but for the purposes of this lab, you will predict 1-day churn as users who do not come back and use the app again after 24 hr of the user's first engagement. This is meant to capture churn after a user's "first impression" of the application or onboarding experience.

In other words, after 24 hr of a user's first engagement with the app:

if the user shows no event data thereafter, the user is considered churned.
if the user does have at least one event datapoint thereafter, then the user is considered returned.
You may also want to remove users who were unlikely to have ever returned anyway after spending just a few minutes with the app, which is sometimes referred to as "bouncing". For example, you will build your model on only on users who spent at least 10 minutes with the app (users who didn't bounce).

The query below defines a churned user with the following definition:

Churned = "any user who spent at least 10 minutes on the app, but after 24 hour from when they first engaged with the app, never used the app again"

You will use the raw event data, from their first touch (app installation) to their last touch, to identify churned and bounced users in the user_churn view query below:

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE VIEW bqmlga4.user_churn AS (
  WITH firstlasttouch AS (
    SELECT
      user_pseudo_id,
      MIN(event_timestamp) AS user_first_engagement,
      MAX(event_timestamp) AS user_last_engagement
    FROM
      `firebase-public-project.analytics_153293282.events_*`
    WHERE event_name="user_engagement"
    GROUP BY
      user_pseudo_id

  )
  
SELECT
    user_pseudo_id,
    user_first_engagement,
    user_last_engagement,
    EXTRACT(MONTH from TIMESTAMP_MICROS(user_first_engagement)) as month,
    EXTRACT(DAYOFYEAR from TIMESTAMP_MICROS(user_first_engagement)) as julianday,
    EXTRACT(DAYOFWEEK from TIMESTAMP_MICROS(user_first_engagement)) as dayofweek,

    #add 24 hr to user's first touch
    (user_first_engagement + 86400000000) AS ts_24hr_after_first_engagement,
    
    #churned = 1 if last_touch within 24 hr of app installation, else 0
    IF (user_last_engagement < (user_first_engagement + 86400000000),
    1,
    0 ) AS churned,
    
    #bounced = 1 if last_touch within 10 min, else 0
    IF (user_last_engagement <= (user_first_engagement + 600000000),
    1,
    0 ) AS bounced,
  FROM
    firstlasttouch
  GROUP BY
    user_pseudo_id,
    user_first_engagement,
    user_last_engagement
    );

SELECT 
  * 
FROM 
  bqmlga4.user_churn 
LIMIT 100;

```

>Job ID dc4d8547-f2fe-4cbc-9e6d-82e23aa6d3d6 successfully executed: 100%

![alt text](images/Task4-3.png)


Review how many of the 15k users bounced and returned below:

```
%%bigquery --project $PROJECT_ID

SELECT
    bounced,
    churned, 
    COUNT(churned) as count_users
FROM
    bqmlga4.user_churn
GROUP BY 
  bounced,
  churned
ORDER BY 
  bounced
```

>Job ID b574cdd6-505b-44c5-9918-956b7fd61498 successfully executed: 100%

![alt text](images/Task4-4.png)

For the training data, you will only end up using data where bounced = 0. Based on the 15k users, you can see that 5,557 ( about 41%) users bounced within the first ten minutes of their first engagement with the app. Of the remaining 8,031 users, 1,883 users ( about 23%) churned after 24 hours which you can validate with the query below:

```
%%bigquery --project $PROJECT_ID

SELECT
    COUNTIF(churned=1)/COUNT(churned) as churn_rate
FROM
    bqmlga4.user_churn
WHERE bounced = 0
```
>Job ID d2aeed32-0259-45d9-a775-a79da5f74402 successfully executed: 100%

![alt text](images/Task4-5.png)


##### Extract user demographic features
There is various user demographic information included in this dataset, including app_info, device, ecommerce, event_params, and geo. Demographic features can help the model predict whether users on certain devices or countries are more likely to churn.

Note that a user's demographics may occasionally change (e.g. moving countries). For simplicity, you will use the demographic information that Google Analytics 4 provides when the user first engaged with the app as indicated by MIN(event_timestamp) in the query below. This enables every unique user to be represented by a single row.

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE VIEW bqmlga4.user_demographics AS (

  WITH first_values AS (
      SELECT
          user_pseudo_id,
          geo.country as country,
          device.operating_system as operating_system,
          device.language as language,
          ROW_NUMBER() OVER (PARTITION BY user_pseudo_id ORDER BY event_timestamp DESC) AS row_num
      FROM `firebase-public-project.analytics_153293282.events_*`
      WHERE event_name="user_engagement"
      )
  SELECT * EXCEPT (row_num)
  FROM first_values
  WHERE row_num = 1
  );

SELECT
  *
FROM
  bqmlga4.user_demographics
LIMIT 10
```

>Job ID 35326a05-94fa-4b04-80c1-79ca74db8860 successfully executed: 100%

![alt text](images/Task4-6.png)


##### Aggregate user behavioral features
Behavioral data in the raw event data spans across multiple events -- and thus rows -- per user. The goal of this section is to aggregate and extract behavioral data for each user, resulting in one row of behavioral data per unique user.

As a first step, you can explore all the unique events that exist in this dataset, based on event_name:


```
%%bigquery --project $PROJECT_ID

SELECT
  event_name,
  COUNT(event_name) as event_count
FROM
    `firebase-public-project.analytics_153293282.events_*`
GROUP BY 
  event_name
ORDER BY
   event_count DESC
```

>Job ID 4ab78611-7e1a-4fda-8f70-edd339410509 successfully executed: 100%

![alt text](images/Task4-7.png)

For this lab, to predict whether a user will churn or return, you can start by counting the number of times a user engages in the following event types:

user_engagement
level_start_quickplay
level_end_quickplay
level_complete_quickplay
level_reset_quickplay
post_score
spend_virtual_currency
ad_reward
challenge_a_friend
completed_5_levels
use_extra_steps

In the SQL query below, you will aggregate the behavioral data by calculating the total number of times when each of the above event_names occurred in the data set per user.

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE VIEW bqmlga4.user_behavior AS (
WITH
  events_first24hr AS (
    # Select user data only from first 24 hr of using the app.
    SELECT
      e.*
    FROM
      `firebase-public-project.analytics_153293282.events_*` e
    JOIN
      bqmlga4.user_churn c
    ON
      e.user_pseudo_id = c.user_pseudo_id
    WHERE
      e.event_timestamp <= c.ts_24hr_after_first_engagement
    )
SELECT
  user_pseudo_id,
  SUM(IF(event_name = 'user_engagement', 1, 0)) AS cnt_user_engagement,
  SUM(IF(event_name = 'level_start_quickplay', 1, 0)) AS cnt_level_start_quickplay,
  SUM(IF(event_name = 'level_end_quickplay', 1, 0)) AS cnt_level_end_quickplay,
  SUM(IF(event_name = 'level_complete_quickplay', 1, 0)) AS cnt_level_complete_quickplay,
  SUM(IF(event_name = 'level_reset_quickplay', 1, 0)) AS cnt_level_reset_quickplay,
  SUM(IF(event_name = 'post_score', 1, 0)) AS cnt_post_score,
  SUM(IF(event_name = 'spend_virtual_currency', 1, 0)) AS cnt_spend_virtual_currency,
  SUM(IF(event_name = 'ad_reward', 1, 0)) AS cnt_ad_reward,
  SUM(IF(event_name = 'challenge_a_friend', 1, 0)) AS cnt_challenge_a_friend,
  SUM(IF(event_name = 'completed_5_levels', 1, 0)) AS cnt_completed_5_levels,
  SUM(IF(event_name = 'use_extra_steps', 1, 0)) AS cnt_use_extra_steps,
FROM
  events_first24hr
GROUP BY
  user_pseudo_id
  );

SELECT
  *
FROM
  bqmlga4.user_behavior
LIMIT 10
```

>Job ID c08bc68e-b91c-498f-9c0c-a151552f4eec successfully executed: 100%

![alt text](images/Task4-8.png)

##### Prepare your train/eval/test datasets for machine learning
In this section, you can now combine these three intermediary views (user_churn, user_demographics, and user_behavior) into the final training data view called ml_features. Here you can also specify bounced = 0, in order to limit the training data only to users who did not "bounce" within the first 10 minutes of using the app.

Note in the query below that a manual data_split column is created in your BigQuery ML table using BigQuery's hashing functions for repeatable sampling. It specifies a 80% train | 10% eval | 20% test split to evaluate your model's performance and generalization.

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE VIEW bqmlga4.ml_features AS (
    
  SELECT
    dem.user_pseudo_id,
    IFNULL(dem.country, "Unknown") AS country,
    IFNULL(dem.operating_system, "Unknown") AS operating_system,
    IFNULL(REPLACE(dem.language, "-", "X"), "Unknown") AS language,
    IFNULL(beh.cnt_user_engagement, 0) AS cnt_user_engagement,
    IFNULL(beh.cnt_level_start_quickplay, 0) AS cnt_level_start_quickplay,
    IFNULL(beh.cnt_level_end_quickplay, 0) AS cnt_level_end_quickplay,
    IFNULL(beh.cnt_level_complete_quickplay, 0) AS cnt_level_complete_quickplay,
    IFNULL(beh.cnt_level_reset_quickplay, 0) AS cnt_level_reset_quickplay,
    IFNULL(beh.cnt_post_score, 0) AS cnt_post_score,
    IFNULL(beh.cnt_spend_virtual_currency, 0) AS cnt_spend_virtual_currency,
    IFNULL(beh.cnt_ad_reward, 0) AS cnt_ad_reward,
    IFNULL(beh.cnt_challenge_a_friend, 0) AS cnt_challenge_a_friend,
    IFNULL(beh.cnt_completed_5_levels, 0) AS cnt_completed_5_levels,
    IFNULL(beh.cnt_use_extra_steps, 0) AS cnt_use_extra_steps,
    chu.user_first_engagement,
    chu.month,
    chu.julianday,
    chu.dayofweek,
    chu.churned,
    # https://towardsdatascience.com/ml-design-pattern-5-repeatable-sampling-c0ccb2889f39
    # BQML Hyperparameter tuning requires STRING 3 partition data_split column.
    # 80% 'TRAIN' | 10%'EVAL' | 10% 'TEST'    
    CASE
      WHEN ABS(MOD(FARM_FINGERPRINT(dem.user_pseudo_id), 10)) <= 7
        THEN 'TRAIN'
      WHEN ABS(MOD(FARM_FINGERPRINT(dem.user_pseudo_id), 10)) = 8
        THEN 'EVAL'
      WHEN ABS(MOD(FARM_FINGERPRINT(dem.user_pseudo_id), 10)) = 9
        THEN 'TEST'    
          ELSE '' END AS data_split
  FROM
    bqmlga4.user_churn chu
  LEFT OUTER JOIN
    bqmlga4.user_demographics dem
  ON 
    chu.user_pseudo_id = dem.user_pseudo_id
  LEFT OUTER JOIN 
    bqmlga4.user_behavior beh
  ON
    chu.user_pseudo_id = beh.user_pseudo_id
  WHERE chu.bounced = 0
  );

SELECT
  *
FROM
  bqmlga4.ml_features
LIMIT 10
```

>Job ID 9611247b-1217-4128-97ff-87a754be06d7 successfully executed: 100%

![alt text](images/Task4-9.png)


##### Validate feature splits
Run the query below to validate the number of examples in each data partition for the 80% train |10% eval |10% test split.

```
%%bigquery --project $PROJECT_ID

SELECT
  data_split,
  COUNT(*) AS n_examples
FROM bqmlga4.ml_features
GROUP BY data_split
```
>Job ID df511436-3c8e-4c42-9baa-a91b37cfcf8f successfully executed: 100%

![alt text](images/Task4-10.png)

###### Train and tune a BQML XGBoost propensity model to predict customer churn

The following code trains and tunes the hyperparameters for a XGBoost model. TO provide a minimal demonstration of BQML hyperparameter tuning in this lab, this model will take about 18 min to train and tune with its restricted search space and low number of trials. In practice, you would generally want at least 10 trials per hyperparameter to achieve improved results.

For more information on the default hyperparameters used, you can read the documentation: CREATE MODEL statement for Boosted Tree models using XGBoost

Model	BQML model_type	Advantages	Disadvantages
XGBoost	BOOSTED_TREE_CLASSIFIER (documentation)	High model performance with feature importances and explainability	Slower to train than BQML LOGISTIC_REG

Note: When you run the CREATE MODEL statement, BigQuery ML can automatically split your data into training and test so you can immediately evaluate your model's performance after training. This is a great option for fast model prototyping. In this lab, however, you split your data manually above using hashing for reproducible data splits that can be used comparing model evaluations across different runs.


```
MODEL_NAME="churn_xgb"
```

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE MODEL bqmlga4.churn_xgb

OPTIONS(
  MODEL_TYPE="BOOSTED_TREE_CLASSIFIER",
  # Declare label column.
  INPUT_LABEL_COLS=["churned"],
  # Specify custom data splitting using the `data_split` column.
  DATA_SPLIT_METHOD="CUSTOM",
  DATA_SPLIT_COL="data_split",
  # Enable Vertex Explainable AI aggregated feature attributions.
  ENABLE_GLOBAL_EXPLAIN=True,
  # Hyperparameter tuning arguments.
  num_trials=8,
  max_parallel_trials=4,
  HPARAM_TUNING_OBJECTIVES=["roc_auc"],
  EARLY_STOP=True,
  # Hyperpameter search space.
  LEARN_RATE=HPARAM_RANGE(0.01, 0.1),
  MAX_TREE_DEPTH=HPARAM_CANDIDATES([5,6])
) AS

SELECT
  * EXCEPT(user_pseudo_id)
FROM
  bqmlga4.ml_features
```

![alt text](images/Task4-11.png)


```
%%bigquery --project $PROJECT_ID

SELECT *
FROM
  ML.TRIAL_INFO(MODEL `bqmlga4.churn_xgb`);
```

![alt text](images/Task4-12.png)


##### Evaluate BQML XGBoost model performance
Once training is finished, you can run ML.EVALUATE to return model evaluation metrics. By default, all model trials will be returned so the below query just returns the model performance for optimal first trial.

```
%%bigquery --project $PROJECT_ID

SELECT
  *
FROM
  ML.EVALUATE(MODEL bqmlga4.churn_xgb)
WHERE trial_id=1;
```

![alt text](images/Task4-13.png)

ML.EVALUATE generates the precision, recall, accuracy, log_loss, f1_score and roc_auc using the default classification threshold of 0.5, which can be modified by using the optional THRESHOLD parameter.

Next, use the ML.CONFUSION_MATRIX function to return a confusion matrix for the input classification model and input data.

For more information on confusion matrices, you can read through a detailed explanation here.

```
%%bigquery --project $PROJECT_ID

SELECT
  expected_label,
  _0 AS predicted_0,
  _1 AS predicted_1
FROM
  ML.CONFUSION_MATRIX(MODEL bqmlga4.churn_xgb)
WHERE trial_id=1;
```
![alt text](images/Task4-14.png)


You can also plot the AUC-ROC curve by using ML.ROC_CURVE to return the metrics for different threshold values for the model.

```
%%bigquery df_roc --project $PROJECT_ID

SELECT * FROM ML.ROC_CURVE(MODEL bqmlga4.churn_xgb)
```

![alt text](images/Task4-15.png)

```
df_roc.plot(x="false_positive_rate", y="recall", title="AUC-ROC curve")
```

##### Inspect global feature attributions
To provide further context to your model performance, you can use the ML.GLOBAL_EXPLAIN function which leverages Vertex Explainable AI as a back-end. Vertex Explainable AI helps you understand your model's outputs for classification and regression tasks. Specifically, Vertex AI tells you how much each feature in the data contributed to your model's predicted result. You can then use this information to verify that the model is behaving as expected, identify and mitigate biases in your models, and get ideas for ways to improve your model and your training data.

```
%%bigquery --project $PROJECT_ID

SELECT
  *
FROM
  ML.GLOBAL_EXPLAIN(MODEL bqmlga4.churn_xgb)
ORDER BY
  attribution DESC;
```

![alt text](images/Task4-16.png)


##### Generate batch predictions
You can generate batch predictions for your BQML XGBoost model using ML.PREDICT.

```
%%bigquery --project $PROJECT_ID

SELECT
  *
FROM
  ML.PREDICT(MODEL bqmlga4.churn_xgb,
  (SELECT * FROM bqmlga4.ml_features WHERE data_split = "TEST"))
```

![alt text](images/Task4-17.png)

The following query returns the probability that the user will return after 24 hrs. The higher the probability and closer it is to 1, the more likely the user is predicted to churn, and the closer it is to 0, the more likely the user is predicted to return.

```
%%bigquery --project $PROJECT_ID

CREATE OR REPLACE TABLE bqmlga4.churn_predictions AS (
SELECT
  user_pseudo_id,
  churned,
  predicted_churned,
  predicted_churned_probs[OFFSET(0)].prob as probability_churned
FROM
  ML.PREDICT(MODEL bqmlga4.churn_xgb,
  (SELECT * FROM bqmlga4.ml_features))
);
```

![alt text](images/Task4-18.png)

![alt text](images/Task4-19.png)


###### Export a BQML model to Vertex AI for online predictions


See the official BigQuery ML Guide: Exporting a BigQuery ML model for online prediction for additional details.

Export BQML model to GCS
You will use the bq extract command in the bq command-line tool to export your BQML XGBoost model assets to Google Cloud Storage for persistence. See the documentation for additional model export options.

```
BQ_MODEL = f"{BQ_DATASET}.{MODEL_NAME}"
BQ_MODEL_EXPORT_DIR = f"gs://{GCS_BUCKET}/{MODEL_NAME}"
```

```
!bq --location=$BQ_LOCATION extract \
--destination_format ML_XGBOOST_BOOSTER \
--model $BQ_MODEL \
$BQ_MODEL_EXPORT_DIR
```

>Waiting on bqjob_r16bb951437accb41_00000196be377825_1 ... (2s) Current status: DONE 

Navigate to Google Cloud Storage in Google Cloud Console to "gs://{GCS_BUCKET}/{MODEL_NAME}". Validate that you see your exported model assets in the below format:

|--/{GCS_BUCKET}/{MODEL_NAME}/
   |--/assets/                       # Contains preprocessing code.  
      |--0_categorical_label.txt     # Contains country vocabulary.
      |--1_categorical_label.txt     # Contains operating_system vocabulary.
      |--2_categorical_label.txt     # Contains language vocabulary.
      |--model_metadata.json         # contains model feature and label mappings.
   |--main.py                        # Can be called for local training runs.
   |--model.bst                      # XGBoost saved model format.
   |--xgboost_predictor-0.1.tar.gz   # Compress XGBoost model with prediction function. 



![alt text](images/Task5-1.png)


##### Upload BQML model to Vertex AI from GCS
Vertex AI contains optimized pre-built training and prediction containers for popular ML frameworks such as TensorFlow, Pytorch, as well as XGBoost. You will upload your XGBoost from GCS to Vertex AI and provide the latest pre-built Vertex XGBoost prediction container to execute your model code to generate predictions in the cells below.


```
IMAGE_URI='us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-4:latest'
```

```
model = vertexai.Model.upload(
    display_name=MODEL_NAME,
    artifact_uri=BQ_MODEL_EXPORT_DIR,
    serving_container_image_uri=IMAGE_URI,
)
```

>Creating Model
Create Model backing LRO: projects/697047613721/locations/us-central1/models/5869319512905482240/operations/7945871020097798144
Model created. Resource name: projects/697047613721/locations/us-central1/models/5869319512905482240@1
To use this Model in another session:
model = aiplatform.Model('projects/697047613721/locations/us-central1/models/5869319512905482240@1')


##### Deploy a Vertex Endpoint for online predictions
Before you use your model to make predictions, you need to deploy it to an Endpoint object. When you deploy a model to an Endpoint, you associate physical (machine) resources with that model to enable it to serve online predictions. Online predictions have low latency requirements; providing resources to the model in advance reduces latency. You can do this by calling the deploy function on the Model resource. This will do two things:

Create an Endpoint resource for deploying the Model resource to.
Deploy the Model resource to the Endpoint resource.
The deploy() function takes the following parameters:

deployed_model_display_name: A human readable name for the deployed model.
traffic_split: Percent of traffic at the endpoint that goes to this model, which is specified as a dictionary of one or more key/value pairs. If only one model, then specify as { "0": 100 }, where "0" refers to this model being uploaded and 100 means 100% of the traffic.
machine_type: The type of machine to use for training.
accelerator_type: The hardware accelerator type.
accelerator_count: The number of accelerators to attach to a worker replica.
starting_replica_count: The number of compute instances to initially provision.
max_replica_count: The maximum number of compute instances to scale to. In this lab, only one instance is provisioned.
explanation_parameters: Metadata to configure the Explainable AI learning method.
explanation_metadata: Metadata that describes your TensorFlow model for Explainable AI such as features, input and output tensors.
Note: this can take about 3-5 minutes to provision prediction resources for your model.


```
endpoint = model.deploy(
    traffic_split={"0": 100},
    machine_type="e2-standard-2",
)
```

![alt text](images/Task5-2.png)


##### Query model for online predictions
XGBoost only takes numerical feature inputs. When you trained your BQML model above with CREATE MODEL statement, it automatically handled encoding of categorical features such as user country, operating system, and language into numeric representations. In order for our exported model to generate online predictions, you will use the categorical feature vocabulary files exported under the assets/ folder of your model directory and the Scikit-Learn preprocessing code below to map your test instances to numeric values.

```
CATEGORICAL_FEATURES = ['country',
                        'operating_system',
                        'language']
```

```
from sklearn.preprocessing import OrdinalEncoder
```

```
def _build_cat_feature_encoders(cat_feature_list, gcs_bucket, model_name, na_value='Unknown'):
    """Build categorical feature encoders for mapping text to integers for XGBoost inference. 
    Args:
      cat_feature_list (list): List of string feature names.
      gcs_bucket (str): A string path to your Google Cloud Storage bucket.
      model_name (str): A string model directory in GCS where your BQML model was exported to.
      na_value (str): default is 'Unknown'. String value to replace any vocab NaN values prior to encoding.
    Returns:
      feature_encoders (dict): A dictionary containing OrdinalEncoder objects for integerizing 
        categorical features that has the format [feature] = feature encoder.
    """
    
    feature_encoders = {}
    
    for idx, feature in enumerate(cat_feature_list):
        feature_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        feature_vocab_file = f"gs://{gcs_bucket}/{model_name}/assets/{idx}_categorical_label.txt"
        feature_vocab_df = pd.read_csv(feature_vocab_file, delimiter = "\t", header=None).fillna(na_value)
        feature_encoder.fit(feature_vocab_df.values)
        feature_encoders[feature] = feature_encoder
    
    return feature_encoders
```

```
def preprocess_xgboost(instances, cat_feature_list, feature_encoders):
    """Transform instances to numerical values for inference.
    Args:
      instances (list[dict]): A list of feature dictionaries with the format feature: value. 
      cat_feature_list (list): A list of string feature names.
      feature_encoders (dict): A dictionary with the format feature: feature_encoder.
    Returns:
      transformed_instances (list[list]): A list of lists containing numerical feature values needed
        for Vertex XGBoost inference.
    """
    transformed_instances = []
    
    for instance in instances:
        for feature in cat_feature_list:
            feature_int = feature_encoders[feature].transform([[instance[feature]]]).item()
            instance[feature] = feature_int
            instance_list = list(instance.values())
        transformed_instances.append(instance_list)

    return transformed_instances
```

```
# Build a dictionary of ordinal categorical feature encoders.
feature_encoders = _build_cat_feature_encoders(CATEGORICAL_FEATURES, GCS_BUCKET, MODEL_NAME)
```


```
%%bigquery test_df --project $PROJECT_ID 

SELECT* EXCEPT (user_pseudo_id, churned, data_split)
FROM bqmlga4.ml_features
WHERE data_split="TEST"
LIMIT 3;
```

```
# Convert dataframe records to feature dictionaries for preprocessing by feature name.
test_instances = test_df.astype(str).to_dict(orient='records')
```

```
# Apply preprocessing to transform categorical features and return numerical instances for prediction.
transformed_test_instances = preprocess_xgboost(test_instances, CATEGORICAL_FEATURES, feature_encoders)
```

```
# Generate predictions from model deployed to Vertex AI Endpoint.
predictions = endpoint.predict(instances=transformed_test_instances)
```

```
for idx, prediction in enumerate(predictions.predictions):
    # Class labels [1,0] retrieved from model_metadata.json in GCS model dir.
    # BQML binary classification default is 0.5 with above "Churn" and below "Not Churn".
    is_churned = "Churn" if prediction[0] >= 0.5 else "Not Churn"
    print(f"Prediction: Customer {idx} - {is_churned} {prediction}")
    print(test_df.iloc[idx].astype(str).to_json() + "\n")
```

###### Next steps
Congratulations! In this lab, you trained, tuned, explained, and deployed a BigQuery ML user churn model to generate high business impact batch and online churn predictions to target customers likely to churn with interventions such as in-game rewards and reminder notifications.

In this lab, you used user_psuedo_id as a user identifier. As next steps, you can extend this code further by having your application return a user_id to Google Analytics so you can join your model's predictions with additional first-party data such as purchase history and marketing engagement data. This enables you to integrate batch predictions into Looker dashboards to help product teams prioritize user experience improvements and marketing teams create targeted user interventions such as reminder emails to improve retention.

Through having your model in Vertex AI Prediction, you also have a scalable prediction service to call from your application to directly integrate online predictions in order to to tailor personalized user game experiences and allow for targeted habit-building notifications.

As you collect more data from your users, you may want to regularly evaluate your model on fresh data and re-train the model if you notice that the model quality is decaying. Vertex Pipelines can help you to automate, monitor, and govern your ML solutions by orchestrating your BQML workflow in a serverless manner, and storing your workflow's artifacts using Vertex ML Metadata. For another alternative for continuous BQML models, checkout the blog post Continuous model evaluation with BigQuery ML, Stored Procedures, and Cloud Scheduler.

