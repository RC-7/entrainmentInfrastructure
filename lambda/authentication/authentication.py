import boto3
import json
import os

TABLE_NAME = os.getenv('tableName')

dynamodb = boto3.resource('dynamodb')
EXPERIMEN_TABLE = dynamodb.Table(TABLE_NAME)


def respond():
    return {
        'statusCode': '200',
        'body': '' ,
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods" : "OPTIONS,POST",
            "Access-Control-Allow-Origin" : "*",
            "X-Requested-With" : "*"
        },
    }

def lambda_handler(event, context):
    
    return respond()