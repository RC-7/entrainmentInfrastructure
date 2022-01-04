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

resource "random_pet" "lambda_bucket_name" {
  prefix = "learn-terraform-functions"
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

resource "aws_iam_role" "lambda_exec" {
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

resource "aws_lambda_function" "entrainment_controller" {
  function_name = "EntrainmentController"

  s3_bucket = aws_s3_bucket.lambda_bucket.id
  s3_key    = aws_s3_bucket_object.lambda_entrainment_controller.key

  runtime = "python3.8"
  handler = "entrainment_controller.lambda_handler"

  source_code_hash = data.archive_file.lambda_entrainment_controller.output_base64sha256

  role = aws_iam_role.lambda_exec.arn
}


resource "aws_cloudwatch_log_group" "entrainment_controller" {
  name = "/aws/lambda/${aws_lambda_function.entrainment_controller.function_name}"

  retention_in_days = 30
}



resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}