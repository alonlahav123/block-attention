#!/usr/bin/env bash
set -euo pipefail

if [[ "$(id -u)" == "0" && -n "${LOCAL_GID:-}" ]]; then
    if ! getent group "${LOCAL_GID}" >/dev/null 2>&1; then
        groupadd -o -g "${LOCAL_GID}" blockattention >/dev/null 2>&1 || true
    fi
fi

if [[ "$(id -u)" == "0" && -n "${LOCAL_UID:-}" ]]; then
    if ! getent passwd "${LOCAL_UID}" >/dev/null 2>&1; then
        useradd -o -m -u "${LOCAL_UID}" -g "${LOCAL_GID:-${LOCAL_UID}}" -s /bin/bash blockattention >/dev/null 2>&1 || true
    fi
    HOME_DIR=$(getent passwd "${LOCAL_UID}" | cut -d: -f6)
    mkdir -p "${HOME_DIR}"
    chown "${LOCAL_UID}:${LOCAL_GID:-${LOCAL_UID}}" "${HOME_DIR}"
    export HOME="${HOME_DIR}"
fi

if [[ "$(id -u)" == "0" && "${ENABLE_SSH:-0}" == "1" ]]; then
    mkdir -p /var/run/sshd
    ssh-keygen -A >/dev/null 2>&1

    if [[ -n "${CONTAINER_PASSWORD:-}" ]]; then
        echo "root:${CONTAINER_PASSWORD}" | chpasswd
    fi

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

if [[ "$(id -u)" == "0" && -n "${LOCAL_UID:-}" ]]; then
    exec gosu "${LOCAL_UID}:${LOCAL_GID:-${LOCAL_UID}}" "$@"
fi

exec "$@"
