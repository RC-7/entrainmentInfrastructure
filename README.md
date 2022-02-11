# entrainmentInfrastructure
A repository to pull up the needed infrastructure for the entrainment Oculus game.

Runner's allow easier actions for terraform and docker:

--------------------------------------------------------------
| To deploy terraform changes | python runner.py --f tf --a apply|


Note: Email password will be stored in AWS Secrets Manager. This step needs to be completed via the console. Currently, it is an env variable that needs to be manually set for the lamba. This has been done for the priving of AWS Screts Manager.

TODO once experiment is ready:
- Enable email from auth lambdda
- Enable event source mapping for save score Lambda