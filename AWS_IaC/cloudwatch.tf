#########################################
################## Logs #################
#########################################

resource "aws_cloudwatch_log_group" "authentication" {
  name = "/aws/lambda/${aws_lambda_function.auth.function_name}"

  retention_in_days = var.cloudwatch_retention
}

resource "aws_cloudwatch_log_group" "set_score" {
  name = "/aws/lambda/${aws_lambda_function.set_score.function_name}"

  retention_in_days = var.cloudwatch_retention
}
