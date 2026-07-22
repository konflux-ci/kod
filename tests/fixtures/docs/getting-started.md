# Getting Started with Konflux

Konflux is a cloud-native platform for building, testing, and deploying applications with a strong focus on software supply chain security. It provides a comprehensive solution that fortifies your software supply chain against emerging threats.

## Creating Your First Application

To get started with Konflux, you need to create an application and add components to it. An application in Konflux represents a logical grouping of components that are developed, tested, and released together.

Steps to create your first application:

1. Log in to the Konflux UI at your organization's Konflux instance.
2. Navigate to the Applications page and click "Create Application."
3. Provide a name for your application and select the workspace where it will reside.
4. Add one or more components by pointing Konflux to your source code repositories.

Once your application is created, Konflux will automatically set up build pipelines for each component.

## Onboarding Components

Each component represents a single container image built from a source repository. When you onboard a component, Konflux will:

- Detect the programming language and build system used in your repository.
- Create a Tekton PipelineRun definition in your repository via a pull request.
- Set up webhooks to trigger builds automatically on new commits.
- Configure integration tests to validate your builds before they are released.

You can customize the build pipeline by editing the PipelineRun YAML file in your repository. Konflux uses Pipelines as Code to manage build definitions directly in your source code.
