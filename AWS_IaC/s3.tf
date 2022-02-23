resource "random_pet" "lambda_bucket_name" {
  length = 4
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = random_pet.lambda_bucket_name.id

  acl           = "private"
  force_destroy = true
}

resource "random_pet" "participant_local_data_bucket_name" {
  length = 4
}

resource "aws_s3_bucket" "participant_data_bucket" {
  bucket = random_pet.participant_local_data_bucket_name.id

  acl           = "private"
  force_destroy = true
}