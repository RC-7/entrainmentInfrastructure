import boto3
import json
import os
import hashlib
from base64 import b64encode
import datetime
import uuid
import smtplib

from boto3.dynamodb.conditions import Attr

TABLE_NAME = os.getenv('tableName')

dynamodb = boto3.resource('dynamodb')
EXPERIMENT_TABLE = dynamodb.Table(TABLE_NAME)

def any_and_not_all (iterable):
    return any(iterable) and not all(iterable)

def respond(code, body):
    return {
        'statusCode': code,
        'body': json.dumps(body),
        'headers': {
            'Content-Type': 'application/json',
            "Access-Control-Allow-Headers" : "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Methods" : "OPTIONS,POST",
            "Access-Control-Allow-Origin" : "*",
            "X-Requested-With" : "*"
        },
    }

def get_participant_id(hash):
    projection_attriibute = 'participantID'
    fe = buildFilterExpression(hash)

    # TODO pull into common util file
    response = EXPERIMENT_TABLE.scan(FilterExpression = eval(fe), ProjectionExpression = projection_attriibute)

    participant_count = response['Count']
    participant_ID = response['Items']

    while 'LastEvaluatedKey' in response:
        response = EXPERIMENT_TABLE.scan(FilterExpression = eval(fe), ProjectionExpression = projection_attriibute)
        participant_ID.append(response['Items'])
        participant_count += response['Count']

    if participant_count == 1:
        return [True, participant_ID[0]]
    elif participant_count == 0:
        return [False, 0]
    elif participant_count > 1:
        return [False, -1]

def create_participant_table_entry(participant_info):

    response = EXPERIMENT_TABLE.put_item(
        Item = participant_info
    )

def buildFilterExpression (hash):
    fe = "Attr('hashIdentifier').eq('" + hash +"')"
    return fe

def create_hash(secret_key, name):
    hash_calculator = hashlib.md5()
    hash_calculator.update(secret_key)
    hash_calculator.update(name)
    hash_value = hash_calculator.hexdigest()
    return hash_value

def check_id(secret_key, name):
    secret_key = bytes(secret_key, 'utf-8')
    name = bytes(name, 'utf-8')
    hash_value = create_hash(secret_key, name)
    return get_participant_id(hash_value)

def email_secret_key(key, email_participant):
    email_from = os.getenv('email_address')
    reply_to = email_from

    subject = '[Entrainment Experiment] Your secret key'
    body = 'Hi <br><br> Thank you again for participating in the study. Your secret key is: <b>{key}</b>. <br><br> \
        Please keep this safe as you will need it for the next session and it is not shared with anyone else'.format(key = key)

    username = email_from
    password = os.getenv('password')

    gmail_smtp_server = 'smtp.gmail.com'
    google_smtp_port = 587

    message = "From: {email_from}\nTo: {email_to}\nContent-type: text/html\nReply-To: {email_reply}\nSubject: {subject}\n\n{body}". \
        format(email_from = email_from, email_to = email_participant, email_reply = reply_to , subject = subject, body = body)
    
    try:
        smrp_session = smtplib.SMTP(gmail_smtp_server, google_smtp_port)
        smrp_session.ehlo()
        smrp_session.starttls()
        smrp_session.login(username, password)
        smrp_session.sendmail(email_from, email_participant, message)
        smrp_session.close()
        return True
    except Exception as ex:
        print (ex)
        return False

def create_participant(name, email):
    key_encoded = os.urandom(5)
    secret_key = b64encode(key_encoded)
    encoded_name = bytes(name, 'utf-8')
    participant_hash = create_hash(secret_key, encoded_name)
    
    participant_ID = str(uuid.uuid4())

    participant_info = {
        'participantID': participant_ID,
        'timestamp': str(datetime.datetime.now()),
        'hashIdentifier': participant_hash
    }
    create_participant_table_entry(participant_info)
    
    secret_key_string = str(secret_key.decode('utf-8'))

    # Return status
    # status = email_secret_key(secret_key_string, email)

    return [True, participant_ID]

def create_negative_response_body(message, resonse_body):
    resonse_body['participant_ID'] = -1
    resonse_body['message'] = message

def lambda_handler(event, context):
    requestValues =  json.loads(event['body'])
    response_message = {
        'participant_ID': '',
        'message': ''
    }
    response_code = ''
#  TODO add are you sure to GUI fe if secret key is not inputted
    # if 'secret_key' in requestValues.keys():
    if all (key in requestValues for key in ('secret_key','name')):
        name = requestValues['name']
        secret_key = requestValues['secret_key']
        [status, participantID] = check_id(secret_key, name)
        if status:
            response_message['participant_ID'] = participantID
            response_message['message'] = 'Participant ID found'
            response_code = '200'
        else:
            response_code = '204'
            if participantID == 0:
                create_negative_response_body('No participant ID found for name and secret key provided', response_message)
            if participantID == -1:
                create_negative_response_body('Multiple participant ID\'s found', response_message)
                
    if any_and_not_all (key in requestValues for key in ('secret_key','name')) and not \
        'email' in requestValues.keys():
            create_negative_response_body('Not enough info given to querry participant ID', response_message)
            response_code = '404'
    if all (key in requestValues for key in ('email','name')) and not \
        'secret_key' in requestValues.keys():
        name = requestValues['name']
        email = requestValues['email']
        [status, participantID] = create_participant(name, email)
        if status:
            response_message['participant_ID'] = participantID
            response_message['message'] = 'Participant ID created, check your email for your secret code'
            response_code = '200'
        else:
            create_negative_response_body('Error creating user', response_message)
            response_code = '404'

    if any_and_not_all (key in requestValues for key in ('email','name')) and not \
        'secret_key' in requestValues.keys():
        response_message['participant_ID'] = -1
        response_message['message'] = 'Not enough info given to create user'
        response_code = '404'

    return respond(response_code, response_message)