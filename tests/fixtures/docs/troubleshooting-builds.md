# Troubleshooting Build Failures

When a build fails in Konflux, there are several common causes and debugging steps you can follow to identify and resolve the issue.

## Viewing Build Logs

To view the logs of a failed build:

1. Navigate to the Applications page in the Konflux UI.
2. Select your application and click on the component that failed.
3. Go to the Activity tab and select the failed PipelineRun.
4. Click on the task that failed to view its detailed logs.

The log output will show the exact error message from the build task. Common errors include missing dependencies, Containerfile syntax issues, and registry authentication problems.

## Common Build Failures

### Authentication Errors

If your build fails with a "401 Unauthorized" or "403 Forbidden" error when pushing images, check the following:

- Verify that the robot account credentials in your image push secret are correct.
- Ensure the secret is linked to the pipeline service account.
- Check that the registry URL matches the expected destination.

### Out of Memory Errors

Builds that run out of memory will be killed by the OOM killer. You can increase the memory limit by adding resource requests to the build task in your PipelineRun definition.

### Network Timeouts

Hermetic builds restrict network access during the build step. If you need to download dependencies, use a prefetch task that runs before the hermetic build step.

## Rerunning a Failed Build

You can rerun a failed build by pushing a new commit to the source repository or by manually triggering a new PipelineRun from the Konflux UI using the "Rerun" button on the failed PipelineRun page.
