# Output value definitions

output "lambda_bucket_name" {
  description = "Name of the S3 bucket used to store function code."

  value = aws_s3_bucket.lambda_bucket.id
}

output "entrainment_controller_function_name" {
  description = "Name of the Lambda function."

  value = aws_lambda_function.entrainment_controller.function_name
}

output "base_url" {
  description = "Base URL for API Gateway stage."

  value = aws_apigatewayv2_stage.entrainment_controller_api.invoke_url
}
