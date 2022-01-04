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

data "archive_file" "entrainment_controller" {
  type = "zip"

  source_dir  = "${path.module}/../lambda/entrainment_controller"
  output_path = "${path.module}/../lambda/entrainment_controller.zip"
}

resource "aws_s3_bucket_object" "entrainment_controller" {
  bucket = aws_s3_bucket.lambda_bucket.id

  key    = "entrainment_controller.zip"
  source = data.archive_file.entrainment_controller.output_path

  etag = filemd5(data.archive_file.entrainment_controller.output_path)
}
