FROM registry.redhat.io/ubi9/nodejs-22:latest

USER root

RUN dnf install -y git openssh-clients python3 python3-pip \
    alsa-lib atk cups-libs gtk3 libXcomposite libXdamage libXrandr \
    mesa-libgbm nss pango libdrm libxkbcommon \
    && dnf clean all

RUN ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') \
    && curl -fsSL "https://github.com/cli/cli/releases/latest/download/gh_$(curl -sI https://github.com/cli/cli/releases/latest | grep -i '^location:' | sed 's/.*tag\/v//' | tr -d '\r\n')_linux_${ARCH}.tar.gz" \
       -o /tmp/gh.tar.gz \
    && tar -xzf /tmp/gh.tar.gz -C /tmp \
    && mv /tmp/gh_*/bin/gh /usr/local/bin/gh \
    && chmod +x /usr/local/bin/gh \
    && rm -rf /tmp/gh*

RUN npm install -g @anthropic-ai/claude-code@latest @playwright/mcp@latest

RUN npx playwright install chromium

RUN LATEST=$(curl -sI https://github.com/multica-ai/multica/releases/latest | grep -i '^location:' | sed 's/.*tag\///' | tr -d '\r\n') \
    && VERSION="${LATEST#v}" \
    && ARCH=$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/') \
    && curl -fsSL "https://github.com/multica-ai/multica/releases/download/${LATEST}/multica-cli-${VERSION}-linux-${ARCH}.tar.gz" \
       -o /tmp/multica.tar.gz \
    && tar -xzf /tmp/multica.tar.gz -C /tmp \
    && mv /tmp/multica /usr/local/bin/multica \
    && chmod +x /usr/local/bin/multica \
    && rm -rf /tmp/multica*

RUN mkdir -p /home/default/multica_workspaces /home/default/.claude \
    && chmod -R g+rwX /home/default

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER 1001
WORKDIR /home/default

ENTRYPOINT ["entrypoint.sh"]
