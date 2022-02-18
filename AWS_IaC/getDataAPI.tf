data "archive_file" "lambda_getData" {
  type = "zip"

  source_dir  = "${path.module}/../lambda_functions/get_data"
  output_path = "${path.module}/../lambda_functions/get_data/get_data.zip"
}

resource "aws_s3_bucket_object" "getData" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "get_data.zip"
  source = data.archive_file.lambda_getData.output_path

  etag = filemd5(data.archive_file.lambda_getData.output_path)
}


resource "aws_iam_role" "lambda_exec_get_data" {
  name = "get_data_lambda"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Sid    = ""
      Principal = {
        Service = "lambda.amazonaws.com"
      }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy_get_data" {
  name = "lambda_policy_get_data"
  role = aws_iam_role.lambda_exec_get_data.id

  policy = jsonencode({
    "Version" : "2012-10-17",
    "Statement" : [{
      "Effect" : "Allow",
      "Action" : [
        "dynamodb:BatchGetItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:Scan",
      ],
      "Resource" : "${aws_dynamodb_table.experimentTable.arn}"
      },
    ]
  })
}

resource "aws_lambda_function" "get_data" {
  function_name = "GetData"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.getData.key

  runtime = "python3.8"
  handler = "get_data.lambda_handler"

  source_code_hash = data.archive_file.lambda_getData.output_base64sha256

  role = aws_iam_role.lambda_policy_get_data.arn

  environment {
    variables = {
      tableName = "${var.table_name}"
    }
  }
}

resource "aws_api_gateway_rest_api" "get_data_api" {
  name        = "get_data_lambda"
  description = "Lambda for getting data from the table"
}

resource "aws_api_gateway_resource" "proxy_get_data" {
  rest_api_id = aws_api_gateway_rest_api.get_data_api.id
  parent_id   = aws_api_gateway_rest_api.get_data_api.root_resource_id
  path_part   = "{proxy+}"
}

resource "aws_api_gateway_method" "proxy_get_data" {
  rest_api_id      = aws_api_gateway_rest_api.get_data_api.id
  resource_id      = aws_api_gateway_resource.proxy_get_data.id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = "true"
}

resource "aws_api_gateway_integration" "lambda_get_data" {
  rest_api_id = aws_api_gateway_rest_api.get_data_api.id
  resource_id = aws_api_gateway_method.proxy_get_data.resource_id
  http_method = aws_api_gateway_method.proxy_get_data.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.get_data.invoke_arn
}

resource "aws_api_gateway_method" "proxy_root_get_data" {
  rest_api_id      = aws_api_gateway_rest_api.get_data_api.id
  resource_id      = aws_api_gateway_rest_api.get_data_api.root_resource_id
  http_method      = "ANY"
  authorization    = "NONE"
  api_key_required = "true"
}

resource "aws_api_gateway_integration" "lambda_root_getData" {
  rest_api_id = aws_api_gateway_rest_api.get_data_api.id
  resource_id = aws_api_gateway_method.proxy_root_get_data.resource_id
  http_method = aws_api_gateway_method.proxy_root_get_data.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.get_data.invoke_arn
}

resource "aws_api_gateway_deployment" "get_data" {
  depends_on = [
    aws_api_gateway_integration.lambda_get_data,
    aws_api_gateway_integration.lambda_root_getData,
  ]

  rest_api_id = aws_api_gateway_rest_api.get_data_api.id
  stage_name  = "experiment"
}

resource "aws_api_gateway_stage" "experiment_get_data" {
  deployment_id = aws_api_gateway_deployment.get_data.id
  rest_api_id   = aws_api_gateway_rest_api.get_data_api.id
  stage_name    = "experiment_get_data"
}

resource "aws_api_gateway_usage_plan" "get_data_usage_plan" {
  name = "get_data_usage_plan"

  api_stages {
    api_id = aws_api_gateway_rest_api.get_data_api.id
    stage  = aws_api_gateway_deployment.get_data.stage_name
  }
  throttle_settings = {
    burst_limit = $var.throttle_burst_limit
    rate_limit = $var.throttle_rate_limit
  }
  quota_settings ={
    limit = $var.quota_limit
    period = "${var.quota_preiod}"
  }
}

resource "aws_api_gateway_api_key" "get_data_api_key" {
  name        = "get Data api key"
  description = "API key to access the authentication lambda"
}

resource "aws_api_gateway_usage_plan_key" "get_data_usage_plan_key" {
  key_id        = aws_api_gateway_api_key.get_data_api_key.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.get_data_usage_plan.id
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.get_data.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_api_gateway_rest_api.get_data_api.execution_arn}/*/*"
}

# TODO Unify with entrainment resource
resource "aws_iam_role_policy_attachment" "lambda_policy_get_data" {
  role       = aws_iam_role.lambda_exec_get_data.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}