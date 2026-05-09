FROM registry.redhat.io/ubi9/nodejs-22:latest

USER root

RUN dnf install -y git openssh-clients python3 python3-pip && dnf clean all

RUN npm install -g @anthropic-ai/claude-code@latest

RUN LATEST=$(curl -sI https://github.com/multica-ai/multica/releases/latest | grep -i '^location:' | sed 's/.*tag\///' | tr -d '\r\n') \
    && VERSION="${LATEST#v}" \
    && ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') \
    && curl -fsSL "https://github.com/multica-ai/multica/releases/download/${LATEST}/multica-cli-${VERSION}-linux-${ARCH}.tar.gz" \
       -o /tmp/multica.tar.gz \
    && tar -xzf /tmp/multica.tar.gz -C /tmp \
    && mv /tmp/multica /usr/local/bin/multica \
    && chmod +x /usr/local/bin/multica \
    && rm -rf /tmp/multica*

RUN useradd -m -s /bin/bash agent
USER agent
WORKDIR /home/agent

RUN mkdir -p /home/agent/multica_workspaces /home/agent/.claude

COPY --chown=agent:agent entrypoint.sh /home/agent/entrypoint.sh

ENTRYPOINT ["/home/agent/entrypoint.sh"]
