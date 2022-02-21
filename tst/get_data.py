import json
import requests

f = open('runnners/aws_resources.json')
aws_resources = json.load(f)
lambda_url = aws_resources['get_data_url']['value']
api_key = aws_resources['get_data_api_key']['value']

def get_data(data_type, body_values):
    header_values = {
        'x-api-key' : api_key,
        'dataType': data_type
    }
    response = requests.post(lambda_url, headers=header_values, data=json.dumps(body_values))

    print('---------------------')
    print('Valid response: ')
    print(response.text)
    print('---------------------')


def main():

    #############################
    ######## entrainment ########
    #############################

    body_values = {
        'projectionAttributes': 'participantID',
        'filterValues': 'test'

    }
    header_data_type = 'entrainmentSettings'

    get_data(header_data_type, body_values)

    #############################
    ####### generic query #######
    #############################


    body_values = {
        'projectionAttributes': 'participantID, score',
        'filterValues': {
            'participantID': {
                'condition': 'eq',
                'value': '1'
            },
            'timestamp': {
                'condition': 'between',
                'value': ['2022-02-21 11:48:10.586025', '2022-02-21 12:50:09.670406']            
            },
            'score': {
                'condition': 'exists',
                'value': ''
            }
        }

    }
    header_data_type = 'generic'

    get_data(header_data_type, body_values)

if __name__ == "__main__":
    main()