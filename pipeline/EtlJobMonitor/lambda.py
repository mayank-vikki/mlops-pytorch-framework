import boto3
import io
import json
import os
import logging
from botocore.exceptions import ClientError

glue = boto3.client('glue')
cw = boto3.client('events')
cp = boto3.client('codepipeline')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    logger.debug("## Environment Variables ##")
    logger.debug(os.environ)
    logger.debug("## Event ##")
    logger.debug(event)
    
    pipeline_name = os.environ['PIPELINE_NAME']
    model_name = os.environ['MODEL_NAME']
    
    result = None
    token = None
    
    try:
        # Get pipeline state and token
        response = cp.get_pipeline_state(name=pipeline_name)
        for stageState in response['stageStates']:
            if stageState['stageName'] == 'ETLApproval':
                for actionState in stageState['actionStates']:
                    if actionState['actionName'] == 'ApproveETL':
                        latestExecution = actionState['latestExecution']
                        executionId = stageState['latestExecution']['pipelineExecutionId']
                        if latestExecution['status'] != 'InProgress':
                            raise(Exception("ETL approval is not awaiting approval: {}".format(latestExecution['status'])))
                        token = latestExecution['token']
        
        # Check Glue job status
        job_name = "{}-pytorch-preprocess-{}".format(model_name, executionId)  # Updated prefix
        job_details = glue.get_job(JobName=job_name)
        logger.info(f"Job Details: {job_details}")
        response = glue.get_job_runs(JobName=job_name)
        if not response['JobRuns']:
                raise Exception(f"No job runs found for job {job_name}")
        job_run_id = response['JobRuns'][0]['Id']
        response = glue.get_job_run(JobName=job_name, RunId=job_run_id)
        status = response['JobRun']['JobRunState']
        logger.info(f"Job Status: {status}")
        
        if status == "SUCCEEDED":
            result = {
                'summary': 'Glue ETL Job completed',
                'status': 'Approved'
            }
        elif status in ["RUNNING", "STARTING"]:
            return "Glue ETL Job ({}) is in progress".format(executionId)
        else:
            result = {
                'summary': response['JobRun']['ErrorMessage'],
                'status': 'Rejected'
            }
            
    except Exception as e:
        logger.error(e)
        result = {
            'summary': str(e),
            'status': 'Rejected'
        }
        
    # Put approval result
    try:
        response = cp.put_approval_result(
            pipelineName=pipeline_name,
            stageName='ETLApproval',
            actionName='ApproveETL',
            result=result,
            token=token
        )
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
        
   # Disable the monitoring rule
    try: 
        response = cw.disable_rule(Name="etl-job-monitor-{}".format(model_name))
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)
    
    return "Done!"
