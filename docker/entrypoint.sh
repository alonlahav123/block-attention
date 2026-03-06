#!/usr/bin/env bash
set -euo pipefail

mkdir -p /var/run/sshd
ssh-keygen -A >/dev/null 2>&1

if [[ -n "${CONTAINER_PASSWORD:-}" ]]; then
    echo "root:${CONTAINER_PASSWORD}" | chpasswd
fi

if [[ "${ENABLE_SSH:-1}" == "1" ]]; then
    if grep -q "^#*PermitRootLogin" /etc/ssh/sshd_config; then
        sed -ri "s/^#?PermitRootLogin.*/PermitRootLogin yes/" /etc/ssh/sshd_config
    else
        echo "PermitRootLogin yes" >> /etc/ssh/sshd_config
    fi

    if grep -q "^#*PasswordAuthentication" /etc/ssh/sshd_config; then
        sed -ri "s/^#?PasswordAuthentication.*/PasswordAuthentication yes/" /etc/ssh/sshd_config
    else
        echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
    fi

    /usr/sbin/sshd
fi

exec "$@"
