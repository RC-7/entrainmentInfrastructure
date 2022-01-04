# import boto3
import json

def respond(body):
    return {
        'statusCode': '200',
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods" : "OPTIONS,POST",
            "Access-Control-Allow-Origin" : "*",
            "X-Requested-With" : "*"
        },
    }


def lambda_handler(event, context):
    print(event)
    return respond('yes')