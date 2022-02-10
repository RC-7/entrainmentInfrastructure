import requests
import json

def valid_auth_request(lambda_url):
    header_values = {
        'x-api-key' : '',
        'Content-Type': 'application/json'
    }
    response = requests.post(lambda_url, headers=header_values)
    print(response.text)


def main():
    f = open('AWS_IaC/aws_resources.json')
    aws_resources = json.load(f)
    valid_auth_request(aws_resources['authentication_lambda_url'])

if __name__ == "__main__":
    main()