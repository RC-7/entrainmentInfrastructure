#########################################
################## Logs #################
#########################################

resource "aws_cloudwatch_log_group" "authentication" {
  name = "/aws/lambda/${aws_lambda_function.auth.function_name}"

  retention_in_days = var.cloudwatch_retention
}

resource "aws_cloudwatch_log_group" "set_data" {
  name = "/aws/lambda/${aws_lambda_function.set_data.function_name}"

  retention_in_days = var.cloudwatch_retention
}
