# Multica on OpenShift — Deployment Guide

## Prerequisites

- OpenShift cluster (ROSA on AWS)
- `oc` CLI logged in
- Quay.io access (or internal registry) for pushing the daemon image
- Vertex AI credentials for Claude Code
- GitHub PAT with repo access

## Quick Start

### 1. Replace placeholders

Edit `openshift-manifests.yaml` and replace all `REPLACE_WITH_*` values:

```bash
# Find all placeholders
grep -n "REPLACE_WITH" openshift-manifests.yaml
```

| Placeholder | Example Value |
|-------------|---------------|
| `REPLACE_WITH_CLUSTER_DOMAIN` | `rosa-abc123.p1.openshiftapps.com` |
| `REPLACE_WITH_YOUR_ORG` | `your-quay-org` |
| `REPLACE_WITH_openssl_rand_-hex_32` | Run: `openssl rand -hex 32` |
| `REPLACE_WITH_STRONG_PASSWORD` | Run: `openssl rand -base64 24` |
| `REPLACE_WITH_YOUR_GCP_PROJECT` | Your GCP project ID for Vertex AI |
| `REPLACE_WITH_DAEMON_TOKEN` | Generate from Multica UI after deploy |
| `REPLACE_WITH_GITHUB_PAT` | GitHub token with repo scope |

### 2. Build and push the daemon image

```bash
# Build
docker build -f Dockerfile.daemon -t quay.io/YOUR_ORG/multica-daemon:latest .

# Push
docker push quay.io/YOUR_ORG/multica-daemon:latest
```

### 3. Deploy to OpenShift

```bash
# Create namespace and deploy everything
oc apply -f openshift-manifests.yaml

# Wait for pods
oc -n sdg-hub-agents get pods -w
```

### 4. Generate daemon token

1. Open the Multica UI at `https://multica.apps.YOUR_CLUSTER_DOMAIN`
2. Log in (use verification code `888888` in dev mode)
3. Go to **Settings → Runtimes → Add Runtime**
4. Copy the daemon token
5. Update the secret:
   ```bash
   oc -n sdg-hub-agents patch secret multica-secrets \
     --type merge -p '{"stringData":{"MULTICA_TOKEN":"mdt_YOUR_TOKEN"}}'
   ```
6. Restart the daemon pod:
   ```bash
   oc -n sdg-hub-agents rollout restart deployment/multica-daemon
   ```

### 5. Configure agents and skills

Once the daemon is running and connected, configure agents and skills
via the Multica UI or CLI (installed locally):

```bash
multica config set server_url https://multica-api.apps.YOUR_CLUSTER_DOMAIN
multica config set app_url https://multica.apps.YOUR_CLUSTER_DOMAIN
multica login
```

Then create agents, import skills, and set up autopilots as described
in the harness design spec.

## Architecture

```
OpenShift Namespace: sdg-hub-agents
├── multica-postgres    (Deployment + PVC + Service)
│   └── pgvector/pgvector:pg17
├── multica-backend     (Deployment + PVC + Service + Route)
│   └── ghcr.io/multica-ai/multica-backend
├── multica-frontend    (Deployment + Service + Route)
│   └── ghcr.io/multica-ai/multica-web
└── multica-daemon      (Deployment + PVC)
    └── Custom image with Claude Code CLI
        └── Spawns Claude Code processes per task
        └── Workspaces persisted on PVC
```

## Security Notes

- The `APP_ENV=development` and `MULTICA_DEV_VERIFICATION_CODE` settings
  are for initial setup only. Switch to email auth (Resend) or OAuth
  for production use.
- The daemon runs with `--permission-mode bypassPermissions` which gives
  Claude Code full filesystem access. This is by design for autonomous
  operation.
- GitHub tokens and Vertex AI credentials are stored as OpenShift Secrets.
- Consider using OpenShift's built-in Vault integration for secret rotation.
