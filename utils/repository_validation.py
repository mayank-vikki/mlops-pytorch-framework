import os
import boto3
from botocore.exceptions import ClientError

s3 = boto3.resource('s3')
ecr = boto3.client('ecr')
code_commit = boto3.client('codecommit')

def check_bucket(bucket):
    """
    Description:
    -----------
    Checks to confirm that a S3 Bucket exists.

    Parameters:
    ----------
    bucket: String
            String specifying the name of the S3 Bucket.
    """
    try:
        s3.meta.client.head_bucket(Bucket=bucket)
        print("Bucket: {} [".format(bucket)+u'\u2714'+"]")
    except ClientError as e:
        print("Bucket: {} [X]".format(bucket))
        print("Error Reason: \n{}\n".format(e))
        if bucket == os.environ['DATA_BUCKET']:
            print("Please check 'Training Data S3 Bucket' is not available.")
        else:
            print("Please check ' ETL Data Bucket' is not available.")


def check_ecr(repo):
    """
    Description:
    -----------
    Checks to confirm that an Elastic Container Registry exists.

    Parameters:
    ----------
    repo: String
          String specifying the name of the Elastic Container Registry.
    """
    try:
        ecr.describe_repositories(repositoryNames=[repo])
        print("Elastic Container Repository: {} [".format(repo)+u'\u2714'+"]")
    except ClientError as e:
        print("Elastic Container Repository: {} [X]".format(repo))
        print("Error Reason: \n{}\n".format(e))
        print("Please check 'ECR Repository' is not available")

def check_codecommit(repo):
    """
    Description:
    -----------
    Checks to confirm that a CodeCommit Repository exists.

    Parameters:
    ----------
    repo: String
          String specifying the name of the CodeCommit Repository
    """
    try:
        code_commit.get_repository(repositoryName=repo)
        print("CodeCommit Repository: {} [".format(repo)+u'\u2714'+"]")
    except ClientError as e:
        print("CodeCommit Repository: {} [X]".format(repo))
        print("Error Reason: \n{}\n".format(e))
        print("Please check 'CodeCommit Repository' is not available")

def main():
    """
    Description:
    -----------
    Checks that the pipeline resources have been created correctly.
    """
    # Run alidation checks
    print("Validating Data Repositories Exist ...\n")
    check_bucket(os.environ['DATA_BUCKET'])
    check_bucket(os.environ['PIPELINE_BUCKET'])
    check_ecr('abalone-pytorch')
    check_codecommit('mlops-pytorch')


if __name__ == "__main__":
    main()