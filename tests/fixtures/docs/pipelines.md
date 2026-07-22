# Build Pipelines

Konflux uses Tekton pipelines to build container images from your source code. Each component has its own build pipeline that runs as a PipelineRun in the cluster.

## How Build Pipelines Work

When a new commit is pushed to a component's source repository, Pipelines as Code triggers a new PipelineRun. The pipeline executes a series of tasks to build, scan, and verify your container image.

The default build pipeline includes tasks for:

- Cloning the source repository.
- Building the container image using Buildah.
- Scanning the image for known vulnerabilities with Clair.
- Generating a signed SLSA provenance attestation with Tekton Chains.
- Pushing the image to the configured container registry.

## Customizing Your Pipeline

You can modify the pipeline by editing the PipelineRun YAML file that Konflux placed in your repository's `.tekton` directory. Common customizations include:

- Adding pre-build tasks such as running unit tests or linters.
- Configuring build arguments and environment variables.
- Using a different base image for the build.
- Enabling hermetic builds that restrict network access during the build process.
- Adding prefetch tasks to download dependencies before the build.

After making changes, commit the updated PipelineRun file and Konflux will use the new pipeline definition for subsequent builds.

## Pipeline Results

Each PipelineRun produces several results that downstream processes consume:

- `IMAGE_URL`: The fully qualified reference to the built container image.
- `IMAGE_DIGEST`: The SHA-256 digest of the image manifest.
- `CHAINS-GIT_URL`: The source repository URL recorded in the provenance.
- `CHAINS-GIT_COMMIT`: The exact commit SHA that was built.
