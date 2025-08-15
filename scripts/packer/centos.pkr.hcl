packer {
  required_plugins {
    amazon = {
      version = ">= 1.2.8"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "github_sha" {
  type        = string
  description = "GitHub commit SHA to tag the AMI with"
  default     = env("GITHUB_SHA")
}

variable "github_repository" {
  type        = string
  description = "GitHub repository name to tag the AMI with"
  default     = env("GITHUB_REPOSITORY")
}

source "amazon-ebs" "centos" {
  ami_name      = "github-actions-centos-nvidia-ami-{{timestamp}}"
  # Use the lowest-cost instance type that can efficiently build and santity-check the driver.
  # It should be old enough to be low-cost, but new enough to be compatible with our desired driver version.
  instance_type = "g6.xlarge"
  region        = "us-east-2"
  source_ami_filter {
    filters = {
      name                = "CentOS Stream 9 x86_64*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["125523088429"] # CentOS CPE team ID.
  }
  ssh_username = "ec2-user"
  tags = {
    Name = "CentOS Stream 9 with Nvidia Drivers"
    BuiltBy = "Packer"
    GitHubCommitSHA = var.github_sha
    GitHubRepository = var.github_repository
  }
}

build {
  sources = ["source.amazon-ebs.centos"]
  provisioner "shell" {
    script = "./setup-centos.sh"
    execute_command = "sudo bash {{.Path}}"
  }
}
