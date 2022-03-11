from abstract_classes.abstract_messaging_interface import AbstractMessagingInterface
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import json
import requests


def load_aws_resource_values():
    f = open('aws_resources.json')
    aws_resources = json.load(f)
    f.close()
    return aws_resources


class AWSMessagingInterface(AbstractMessagingInterface):

    def __init__(self):
        self.aws_resources = load_aws_resource_values()
        self.my_config = Config(
            region_name='eu-west-1',
            signature_version='v4',
            retries={
                'max_attempts': 10,
                'mode': 'standard'
            }
        )

    def send_data(self, type_of_data, data, queue='set_score_sqs_url'):
        sqs = boto3.client('sqs', config=self.my_config)
        queue_url = self.aws_resources['set_score_sqs_url']['value']
        data_types = ['Score', 'LevelID', 'PData']
        if type_of_data not in data_types:
            return [False, 'Invalid data type']
        try:
            sqs.send_message(
                QueueUrl=queue_url,
                DelaySeconds=10,
                MessageAttributes={
                    'type': {
                        'DataType': 'String',
                        'StringValue': type_of_data
                    },
                },
                MessageBody=json.dumps(data)
            )
        except ClientError as err:
            err_message = 'Error Message: {}'.format(err.response['Error']['Message'])
            if err.response['Error']['Code'] == 'InternalError':
                print('Error sending message to queue')
                print('Error Message: {}'.format(err.response['Error']['Message']))
                print('Request ID: {}'.format(err.response['ResponseMetadata']['RequestId']))
                print('Http code: {}'.format(err.response['ResponseMetadata']['HTTPStatusCode']))
                return [False, err_message]
            else:
                print('Unexpected error sending message to queue')
                print(err)
                return [False, err_message]
        return [True, 'Success, message sent']

    def authenticate(self, auth_body):
        api_key = self.aws_resources['authentication_api_key']['value']
        lambda_url = self.aws_resources['authentication_lambda_url']['value']
        header_values = {
            'x-api-key': api_key,
        }
        response = requests.post(lambda_url, headers=header_values, data=json.dumps(auth_body))
        print(response.text)
        if response.status_code == 200:
            return [True, json.loads(response.text)]
        else:
            return [False, response.reason]

    def get_data(self, data_type, data_request_body):
        api_key = self.aws_resources['get_data_url']['value']
        lambda_url = self.aws_resources['get_data_api_key']['value']
        header_values = {
            'x-api-key': api_key,
            'dataType': data_type
        }
        response = requests.post(lambda_url, headers=header_values, data=json.dumps(data_request_body))
        if response.status_code == '200':
            return [True, response.text]
        else:
            return [False, response.reason]
