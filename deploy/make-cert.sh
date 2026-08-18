#!/usr/bin/env bash
# Self-signed cert for local HTTPS, so a phone on the LAN can use its camera.
#
# getUserMedia needs a secure context. localhost is exempt, an IP address is
# not, so reaching this server from a phone requires TLS even on a home network.
#
#   ./deploy/make-cert.sh 192.168.1.146
#
# Not needed on Runpod: the proxy already serves HTTPS.
set -euo pipefail

IP="${1:-}"
if [ -z "$IP" ]; then
  IP=$(ip -4 addr show scope global | grep -oP 'inet \K[\d.]+' | head -1)
  echo "no IP given, using $IP"
fi

OUT="${2:-deploy/certs}"
mkdir -p "$OUT"

# iOS rejects certificates valid for more than 825 days, and ignores CN
# entirely - the address must appear in subjectAltName or Safari refuses.
openssl req -x509 -newkey rsa:2048 -nodes -days 800 \
  -keyout "$OUT/key.pem" -out "$OUT/cert.pem" \
  -subj "/CN=deep-live-cam" \
  -addext "subjectAltName=IP:${IP},IP:127.0.0.1,DNS:localhost" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth" 2>/dev/null

chmod 600 "$OUT/key.pem"
echo "wrote $OUT/cert.pem and $OUT/key.pem for IP:$IP"
echo
echo "run the server with:"
echo "  -e DLC_TLS_CERT=/app/deploy/certs/cert.pem"
echo "  -e DLC_TLS_KEY=/app/deploy/certs/key.pem"
