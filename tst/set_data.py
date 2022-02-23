from distutils.command.config import config
import boto3
import json
from botocore.config import Config
import datetime

my_config = Config(
    region_name = 'eu-west-1',
    signature_version = 'v4',
    retries = {
        'max_attempts': 10,
        'mode': 'standard'
    }
)
sqs = boto3.client('sqs', config = my_config)
f = open('runnners/aws_resources.json')
aws_resources = json.load(f)
queue_url = aws_resources['set_data_sqs_url']['value']

def send_to_queue(message, attributes):

    response = sqs.send_message(
        QueueUrl = queue_url,
        DelaySeconds = 10,
        MessageAttributes = attributes,
        MessageBody = json.dumps(message)
    )
    print(response)

def main():
    #############################
    ########### Score ###########
    #############################
    message = {
        'participantID': '1',
        'score': '3',
        'timestamp': str(datetime.datetime.now()),
        'levelID': '1D'
    }
    attributes = {
            'type' : {
                'DataType': 'String',
                'StringValue': 'Score'
            },
        }
    send_to_queue(message, attributes)
    #############################
    ######## Entrainment ########
    #############################
    message = {
        'participantID': '1',
        'customEntrainment': {
            "visual": {
                'colour': 'red',
                'frequency': '500',
                },
            'audio': {
                'baseFrequency': 'red',
                'entrainmentFrequency': '500',
                },
            'neurofeedback': {
                'redChannel': '81',
                'greenChannel': '169'
                },
            },
        'timestamp': str(datetime.datetime.now(datetime.timezone.utc)),
        'session': '1'
    }
    attributes = {
        'type' : {
            'DataType': 'String',
            'StringValue': 'EntrainmentSettings'
        },
    }
    send_to_queue(message, attributes)
    
    #############################
    ########## LevelID ##########
    #############################
    message = {
        'participantID': '1',
        'levelID': '7',
        'timestamp': str(datetime.datetime.now()),
        'startTime': str(datetime.datetime.now()),
        'endTime': str(datetime.datetime.now()),
        'session': '1'
    }
    attributes = {
        'type' : {
            'DataType': 'String',
            'StringValue': 'LevelID'
        },
    }
    send_to_queue(message, attributes)

    #############################
    ########### PData ###########
    #############################

    message = {
        'participantID': '1',
        'sex': 'M',
        'age': 2,
        'experience': 0,
        'timestamp': str(datetime.datetime.now()),
    }
    attributes = {
        'type' : {
            'DataType': 'String',
            'StringValue': 'PData'
        },
    }
    send_to_queue(message, attributes)

if __name__ == "__main__":
    main()