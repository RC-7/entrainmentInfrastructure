terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.48.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.2.0"
    }
  }

  required_version = "~> 1.0"
}

provider "aws" {
  region = var.aws_region
}

#########################################
############ Dynamo DB table ############
#########################################

resource "aws_dynamodb_table" "experimentTable" {
  name           = var.table_name
  billing_mode   = var.table_billing_mode
  hash_key       = var.table_hash_key
  range_key      = var.table_range_key
  read_capacity  = var.table_read_capacity
  write_capacity = var.table_write_capacity

  # Key attributes

  attribute {
    name = "participantID"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }

# TODO Add as GSI

}

#########################################
############ entrainmentLambda ############
#########################################


resource "random_pet" "lambda_bucket_name" {
  prefix = "learn-terraform-functions" #change
  length = 4
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = random_pet.lambda_bucket_name.id

  acl           = "private"
  force_destroy = true
}


data "archive_file" "lambda_entrainment_controller" {
  type = "zip"

  source_dir  = "${path.module}/../lambda/entrainment_controller"
  output_path = "${path.module}/../lambda/entrainment_controller/entrainment_controller.zip"
}

resource "aws_s3_bucket_object" "lambda_entrainment_controller" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "entrainment_controller.zip"
  source = data.archive_file.lambda_entrainment_controller.output_path

  etag = filemd5(data.archive_file.lambda_entrainment_controller.output_path)
}


resource "aws_iam_role" "lambda_exec_controller" {
  name = "serverless_lambda"

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

resource "aws_iam_role_policy" "lambda_policy_entrainment_controller" {
  name = "lambda_policy_entrainment_controller"
  role = aws_iam_role.lambda_exec_controller.id

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
      }
    ]
  })
}

resource "aws_lambda_function" "entrainment_controller" {
  function_name = "EntrainmentController"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.lambda_entrainment_controller.key

  runtime = "python3.8"
  handler = "entrainment_controller.lambda_handler"

  source_code_hash = data.archive_file.lambda_entrainment_controller.output_base64sha256

  role = aws_iam_role.lambda_exec_controller.arn

  environment {
    variables = {
      tableName           = "${var.table_name}"
      participantID       = ""
      experimentStartTime = ""
    }
  }
}


resource "aws_cloudwatch_log_group" "entrainment_controller" {
  name = "/aws/lambda/${aws_lambda_function.entrainment_controller.function_name}"

  retention_in_days = var.cloudwatch_retention
}



resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec_controller.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
resource "aws_apigatewayv2_api" "entrainment_controller_api" {
  name          = "entrainment_lambda"
  protocol_type = "HTTP"
}
# TODO Look at making gateway more secure with some tokens
resource "aws_apigatewayv2_stage" "entrainment_controller_api" {
  api_id = aws_apigatewayv2_api.entrainment_controller_api.id

  name        = "entrainment_lambda"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gw.arn

    format = jsonencode({
      requestId               = "$context.requestId"
      sourceIp                = "$context.identity.sourceIp"
      requestTime             = "$context.requestTime"
      protocol                = "$context.protocol"
      httpMethod              = "$context.httpMethod"
      resourcePath            = "$context.resourcePath"
      routeKey                = "$context.routeKey"
      status                  = "$context.status"
      responseLength          = "$context.responseLength"
      integrationErrorMessage = "$context.integrationErrorMessage"
      }
    )
  }
}

resource "aws_apigatewayv2_integration" "entrainment_controller_api" {
  api_id = aws_apigatewayv2_api.entrainment_controller_api.id

  integration_uri    = aws_lambda_function.entrainment_controller.invoke_arn
  integration_type   = "AWS_PROXY"
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "entrainment_controller_api" {
  api_id = aws_apigatewayv2_api.entrainment_controller_api.id

  route_key = "POST /getSettings"
  target    = "integrations/${aws_apigatewayv2_integration.entrainment_controller_api.id}"
}

resource "aws_cloudwatch_log_group" "api_gw" {
  name = "/aws/api_gw/${aws_apigatewayv2_api.entrainment_controller_api.name}"

  retention_in_days = var.cloudwatch_retention
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.entrainment_controller.function_name
  principal     = "apigateway.amazonaws.com"

  source_arn = "${aws_apigatewayv2_api.entrainment_controller_api.execution_arn}/*/*"
}
