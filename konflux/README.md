# KOD Konflux onboarding

Onboards KOD to a Konflux instance so the image builds on push/PR and rebuilds
daily with fresh documentation. The steps are the same regardless of the cluster
or environment (staging, production, an internal instance, …); only the tenant
namespace, cluster, and generated image registry differ.

Throughout, substitute your own values:

| Placeholder | Meaning |
|-------------|---------|
| `<TENANT_NS>` | Your Konflux tenant namespace |
| `<REPO>` | The KOD git repository Konflux watches (e.g. `github.com/konflux-ci/kod`) |

## Design

The FAISS index (`data/index/`) is **gitignored** and produced by `kod pipeline`
(extract → transform → embed → index). The `Containerfile` only `COPY`s a
pre-built index, so a stock git-checkout build has nothing to copy. Rather than
move the ETL into the Containerfile (which would break local `podman build`), the
ETL runs as a dedicated in-repo Tekton task (`.tekton/tasks/kod-etl.yaml`).

The generated `docker-build` pipeline shares the source between tasks via a
**`workspace` PVC** (git-clone checks out into `$(workspaces.output.path)/source`;
buildah builds from `$(workspaces.source.path)/source`). The ETL task simply
mounts that same workspace and runs `kod -c config.production.yaml pipeline` in
the checkout, writing `data/index/` in place. `buildah` then runs next and its
`COPY data/index/` finds the freshly generated index. No trusted-artifact
plumbing is needed. To keep the small workspace PVC from filling up, the ETL puts
the uv venv, uv cache, and FastEmbed model on an `emptyDir` (`/var/workdir`); only
`data/` lands on the PVC (bumped 1Gi → 5Gi). Local builds are unaffected — devs
still run `kod pipeline` + `kod build-image`.

## Files

| File | Applied to | Purpose |
|------|-----------|---------|
| `application-and-component.yaml` | `<TENANT_NS>` (cluster) | Application + Component (`configure-pac`) |
| `daily-rebuild.yaml` | `<TENANT_NS>` (cluster) | SA + RoleBinding + daily CronJob |
| `../.tekton/tasks/kod-etl.yaml` | KOD repo | ETL task (generates + injects the index) |
| `../.tekton/tasks/kod-smoke-test.yaml` | KOD repo | Starts the image, asserts MCP tools are served |

> Placement note: the Application/Component/CronJob CRs here are bootstrap
> manifests kept for convenience. Confirm whether your instance expects tenant
> resources to be managed declaratively in its GitOps / tenant-config repo before
> treating this directory as the source of truth.

## Prerequisites

The ETL task (`.tekton/tasks/kod-etl.yaml`) mounts a ConfigMap named
`trusted-ca` (key `ca-bundle.crt`) and appends it to the model-download trust
store. This mount is currently **required**: a tenant without that exact
ConfigMap fails the `kod-etl` pod before it starts, even on a public-CA network.
Ensure it exists in your tenant namespace — on OpenShift, a cluster-CA-injected
ConfigMap works:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: trusted-ca
  labels:
    config.openshift.io/inject-trusted-cabundle: "true"
data: {}
```

(A follow-up will make this mount optional so public-CA tenants need no setup.)

## Onboarding steps

1. **Bootstrap the Component** (opens the PaC PR):
   ```
   oc apply -f konflux/application-and-component.yaml -n <TENANT_NS>
   ```
   Konflux opens a PR on `<REPO>` adding `.tekton/kod-push.yaml`,
   `.tekton/kod-pull-request.yaml`, and pulling in the `docker-build` pipeline
   (current bundle digests).

   > The Component in `application-and-component.yaml` hard-codes
   > `spec.source.git.url: https://github.com/konflux-ci/kod` (the upstream
   > repo). If you are onboarding a **fork**, edit that URL to your fork before
   > applying — otherwise Konflux configures the upstream repo, not yours.

2. **Customize that PR** (already applied to both `.tekton/kod-push.yaml` and
   `.tekton/kod-pull-request.yaml` in this repo; do the edits on the generated
   branch so the bundle SHAs stay current — do not hand-copy an older pipeline):
   - `hermetic` stays `"false"` and `prefetch-input` `""` (pipeline defaults):
     the ETL and `uv sync` need network, and the `Containerfile` build itself
     downloads the FastEmbed model from Hugging Face.
   - Add `spec.timeouts.pipeline: 2h0m0s` (the default 1h is too tight for the
     slow embedding step).
   - Register the two in-repo tasks via the PaC annotation on both PipelineRuns:
     ```
     pipelinesascode.tekton.dev/task: '[.tekton/tasks/kod-etl.yaml, .tekton/tasks/kod-smoke-test.yaml]'
     ```
   - Insert the ETL task **after** `prefetch-dependencies`, mounting the shared
     `workspace` PVC:
     ```yaml
     - name: kod-etl
       runAfter: [prefetch-dependencies]
       taskRef: {name: kod-etl}
       params:
         - {name: config, value: config.production.yaml}
       workspaces:
         - {name: source, workspace: workspace}
     ```
   - Repoint `build-container`'s `runAfter` from `prefetch-dependencies` to
     `kod-etl` (it already reads the same `workspace`, so nothing else changes —
     the ETL's `data/index/` is sitting in buildah's build context).
   - Add the smoke test **after** `build-image-index`:
     ```yaml
     - name: smoke-test
       runAfter: [build-image-index]
       when: [{input: "$(params.skip-checks)", operator: in, values: ["false"]}]
       taskRef: {name: kod-smoke-test}
       params:
         - {name: IMAGE, value: "$(tasks.build-image-index.results.IMAGE_URL)@$(tasks.build-image-index.results.IMAGE_DIGEST)"}
     ```
   - Bump the `workspace` `volumeClaimTemplate` storage `1Gi` → `5Gi` (the ETL
     writes the doc clones + index onto the PVC).
   - PR-variant params are as generated: `output-image` `…:on-pr-{{revision}}`,
     `image-expires-after: 5d`, `cancel-in-progress: "true"`.

   Two environment-specific gotchas the ETL task already handles, worth knowing
   if you adapt it:
   - **TLS to Hugging Face.** Both the ETL task and the `Containerfile` build
     download the FastEmbed model from `huggingface.co`. If your build network
     terminates TLS with an internal CA, the download fails under Python's
     bundled `certifi`. The ETL task mounts the cluster `trusted-ca` ConfigMap
     and appends it to the venv's `certifi` bundle (httpx, used by
     `huggingface_hub`, ignores `SSL_CERT_FILE`). Note: the `Containerfile`'s
     own model download does **not** yet get this CA fix, so on a
     TLS-intercepting network the container build can still fail — a follow-up
     will reuse the ETL's model cache instead of downloading twice.
   - **Embed step memory.** FastEmbed's default batch size drives O(seq²)
     attention memory and can OOM the embed step. The task sets
     `KOD_EMBED_BATCH_SIZE` (and `KOD_EMBED_THREADS` to cap the ONNX Runtime
     thread pool) to keep it within the container's memory limit.

3. **Enable daily rebuilds:**
   ```
   oc apply -f konflux/daily-rebuild.yaml -n <TENANT_NS>
   ```
   The CronJob nudges the Component once a day
   (`build.appstudio.openshift.io/request=trigger-pac-build`) so the ETL re-runs
   against the latest docs.

   RBAC note: many Konflux tenants do **not** let you create namespaced `Role`
   objects, but do let you create `RoleBinding`s to the platform's curated
   ClusterRoles. `daily-rebuild.yaml` therefore binds the CronJob's ServiceAccount
   to the `konflux-maintainer-user-actions` ClusterRole (which grants `patch` on
   `components`) rather than defining a custom Role. If your instance uses
   different ClusterRole names, adjust the `roleRef`.

## Verification

- After step 1: the PaC PR appears on `<REPO>`.
- After step 2 merges: open a test PR → the on-PR PipelineRun runs
  clone → prefetch → **kod-etl** → build → build-image-index → **smoke-test**,
  and pushes an expiring `on-pr-<sha>` image. The `kod-etl` log shows
  extract/embed counts; the buildah log shows `COPY data/index/` succeeding.
- Merge to `main` → the on-push run publishes the Component's image (the registry
  path is generated by the image-controller for your tenant, e.g.
  `quay.io/redhat-user-workloads*/<TENANT_NS>/kod:<sha>`).
- After step 3: `oc create job --from=cronjob/kod-daily-rebuild kod-rebuild-test -n <TENANT_NS>`
  → the Component annotation flips and a new PipelineRun starts.
