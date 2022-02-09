from distutils.command.config import config
import boto3
import json
from botocore.config import Config

my_config = Config(
    region_name = 'eu-west-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)

def send_to_queue():

    sqs = boto3.client('sqs', config = my_config)
    queue_url = ''

    message = {
        'participantID': '1',
        'score': '3',
        'timestamp': '12232123',
        'levelID': '1D'
    }

    response = sqs.send_message(
        QueueUrl = queue_url,
        DelaySeconds = 10,
        MessageAttributes={
            'type' : {
                'DataType': 'String',
                'StringValue': 'Score'
            },
        },
        MessageBody=json.dumps(message)
    )
    print(response)

def main():
    send_to_queue()

if __name__ == "__main__":
    main()