import argparse
import os

# TODO add help docs

build_commands = {
    "experiment_setup_only": "docker build --target experiment_setup_only -t experiment_setup ..",
    "experiment_complete": "docker build --target experiment_complete --build-arg tensorflow_version=latest -t "
                           "experiment_runner .. "
}

delete_commands = {
    "experiment_setup_only": "docker image rm experiment_setup .",
    "experiment_complete": "docker image rm experiment_runner ."
}

supported_frameworks = ['tf', 'docker']

docker_actions = ['build', 'run']
terraform_actions = ['validate', 'fmt', 'apply', 'plan', 'state', 'destroy', 'init']
all_action = docker_actions + terraform_actions

docker_targets = ['experiment_setup_only', 'experiment_complete']

my_parser = argparse.ArgumentParser(prog='Experiment runner',
                                    usage='%(prog)s [options] path',
                                    description='Assiste in running the experiment')

my_parser.add_argument('--framework', action='store', choices=supported_frameworks, required=True)
my_parser.add_argument('--action', action='store', choices=all_action, required=True)
my_parser.add_argument('--target', action='store', choices=docker_targets, required=False)
args = my_parser.parse_args()

if args.framework == 'docker':
    if args.action == 'build':
        os.system(build_commands[args.target])
    if args.action == 'delete':
        os.system(delete_commands[args.target])
    # if args.action == 'clean':
    #     os.system("docker rm -f $(docker ps -a -q)")

if args.framework == 'tf':
    if args.action in docker_actions:
        message = "{action} can only be used with Docker relate commands".format(action=args.action)
        my_parser.error(message)
        # TODO Add check to see what images are around
    if args.action != 'apply':
        command_to_run = "docker run --rm experiment_setup init && terraform -chdir=../AWS_IaC  {action}"\
            .format(action=args.action)
        os.system(command_to_run)
    else:
        AWS_ACCESS_KEY_ID = os.popen("aws --profile default configure get aws_access_key_id").read().rstrip()
        AWS_SECRET_ACCESS_KEY = os.popen("aws --profile default configure get aws_secret_access_key").read().rstrip()

        apply_command = "docker run --rm -e AWS_ACCESS_KEY_ID={access_key} -e AWS_SECRET_ACCESS_KEY={secret_key} " \
                        "experiment_setup init && terraform -chdir=../AWS_IaC  {action} &&  terraform " \
                        "-chdir=../AWS_IaC output -json > aws_resources.json". \
            format(access_key=AWS_ACCESS_KEY_ID, secret_key=AWS_SECRET_ACCESS_KEY, action=args.action)
        os.system(apply_command)
        os.system('cp aws_resources.json ../testing_interface')
