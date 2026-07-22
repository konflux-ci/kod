# Enterprise Contract

The Enterprise Contract (EC) is a policy enforcement tool in Konflux that verifies whether container images meet a set of release policies before they can be deployed to production.

## How Enterprise Contract Works

Enterprise Contract evaluates container image attestations against a configurable set of Rego policies. These policies check aspects such as:

- Provenance: Was the image built by a trusted pipeline?
- Signatures: Is the image signed by Tekton Chains?
- Test results: Did all required integration tests pass?
- CVE scanning: Are there any critical vulnerabilities in the image?
- Source code: Is the source code accessible and from an approved repository?

## Configuring Policies

Policies are defined in an EnterpriseContractPolicy custom resource. Each policy specifies a list of policy sources and optional configuration. You can customize which rules are enforced for your application by creating or modifying the policy resource.

The default policy set includes rules for SLSA compliance, signature verification, and test requirements. You can exclude specific rules if they do not apply to your use case.

## Integration with Release Pipelines

During a release, the release pipeline runs an Enterprise Contract validation task. This task evaluates the snapshot being released against the policy defined in the ReleasePlanAdmission resource. If any policy violations are found, the release is blocked and the violations are reported in the PipelineRun results.

## Viewing Policy Results

Policy evaluation results are available in the integration test PipelineRun logs. Each policy rule produces a pass, fail, or warning result with a message explaining the outcome.
