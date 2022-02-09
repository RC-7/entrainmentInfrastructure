import argparse
import os

# TODO add help docs

run_commands = {
    "experiment_setup_only": "docker build --target experiment_setup_only -t experiment_setup .",
    "experiment_complete": "docker build --target experiment_complete --build-arg tensorflow_version=latest -t experiment_runner ."
}

delete_commands = {
    "experiment_setup_only": "docker image rm experiment_setup .",
    "experiment_complete": "docker image rm experiment_runner ."
}

supported_frameworks = ['tf', 'docker']

docker_actions = ['build', 'run']
terraform_actions = ['validate', 'fmt', 'apply', 'plan', 'state', 'destroy', 'init']
all_action =[docker_actions, terraform_actions]

docker_targets = ['experiment_setup_only', 'experiment_complete']

my_parser = argparse.ArgumentParser(prog='Experiment runner',
                                    usage='%(prog)s [options] path',
                                    description='Assiste in running the experiment')

my_parser.add_argument('--framework', action='store', choices= supported_frameworks, required=True)
my_parser.add_argument('--action', action='store', choices= all_action, required=True)
my_parser.add_argument('--target', action='store', choices= docker_targets, required=False)
args = my_parser.parse_args()

print(args)
if args.framework == 'docker':
    if args.action.lower() == 'build':
        os.system(run_commands[args.action])
    if  args.action.lower() == 'run':
        os.system(delete_commands[args.action])

if args.framework == 'tf':
    if args.action in docker_actions:
        message = "{action} can only be used with Docker relate commands".format(action = args.action)
        my_parser.error(message)
    if args.action == 'validate':
        # TODO Add check to see what images are around
        command_to_run = "docker run experiment_setup validate"
        os.system(command_to_run)
