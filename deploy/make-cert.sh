#!/usr/bin/env bash
# Self-signed cert for local HTTPS, so a phone on the LAN can use its camera.
#
# getUserMedia needs a secure context. localhost is exempt, an IP address is
# not, so reaching this server from a phone requires TLS even on a home network.
#
#   ./deploy/make-cert.sh 192.168.1.146
#   ./deploy/make-cert.sh 192.168.1.146 172.20.10.2      # home Wi-Fi and hotspot
#
# Several addresses on purpose. A laptop that moves between a home network and a
# phone's hotspot gets a different IP on each, and a certificate naming only one
# of them fails on the other - the address has to be in subjectAltName, so this
# cannot be worked around from the browser. Naming them all means the phone
# trusts the certificate once instead of once per network. With no arguments it
# picks up every global IPv4 address this machine currently has.
#
# Not needed on Runpod: the proxy already serves HTTPS.
set -euo pipefail

if [ "$#" -gt 0 ]; then
  IPS=("$@")
else
  mapfile -t IPS < <(ip -4 addr show scope global | grep -oP 'inet \K[\d.]+')
  echo "no IP given, using every global address: ${IPS[*]}"
fi

OUT="deploy/certs"
mkdir -p "$OUT"

SAN="IP:127.0.0.1,DNS:localhost"
for ip in "${IPS[@]}"; do
  SAN="IP:${ip},${SAN}"
done

# iOS rejects certificates valid for more than 825 days, and ignores CN
# entirely - the address must appear in subjectAltName or Safari refuses.
openssl req -x509 -newkey rsa:2048 -nodes -days 800 \
  -keyout "$OUT/key.pem" -out "$OUT/cert.pem" \
  -subj "/CN=deep-live-cam" \
  -addext "subjectAltName=${SAN}" \
  -addext "basicConstraints=critical,CA:FALSE" \
  -addext "keyUsage=critical,digitalSignature,keyEncipherment" \
  -addext "extendedKeyUsage=serverAuth" 2>/dev/null

chmod 600 "$OUT/key.pem"
echo "wrote $OUT/cert.pem and $OUT/key.pem for ${SAN}"
echo
echo "run the server with:"
echo "  -e DLC_TLS_CERT=/app/deploy/certs/cert.pem"
echo "  -e DLC_TLS_KEY=/app/deploy/certs/key.pem"
