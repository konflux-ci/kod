# Managing Workspaces

Workspaces in Konflux provide isolated environments where teams can develop, test, and release their applications independently.

## What is a Workspace

A workspace maps to a Kubernetes namespace on the cluster. Each workspace has its own set of applications, components, secrets, and service accounts. Team members are granted access to specific workspaces through role-based access control.

When you create a new workspace, Konflux provisions the namespace and sets up the required resources including pipeline service accounts, image pull secrets, and default integration test scenarios.

## Creating a Workspace

To create a new workspace:

1. Navigate to the Workspaces page in the Konflux UI.
2. Click "Create Workspace" and provide a unique name.
3. Select the team members who should have access.
4. Choose the default build pipeline configuration for the workspace.

Workspace names must be unique within your Konflux instance and can only contain lowercase letters, numbers, and hyphens.

## Environment Configuration

Each workspace can have multiple environments configured for different stages of the release process. Common environment configurations include:

- **Development**: Used for iterating on application changes with relaxed policies.
- **Staging**: A pre-production environment that mirrors production settings.
- **Production**: The final deployment target with strict Enterprise Contract policies.

Environments are represented by ReleasePlanAdmission resources that define which policies to enforce and where to deploy the released artifacts.

## Sharing Resources Between Workspaces

By default, workspaces are fully isolated. However, you can share certain resources between workspaces by creating cross-namespace role bindings. This is useful when multiple teams need to access shared container images or common build task bundles.
