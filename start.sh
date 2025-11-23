#!/bin/sh
set -e

if [ -n "$DB_ROOT_CERT" ]; then
  mkdir -p /root/.postgresql
  echo "$DB_ROOT_CERT" > /root/.postgresql/root.crt
  chmod 600 /root/.postgresql/root.crt
  echo "✅ Wrote /root/.postgresql/root.crt"
else
  echo "⚠️ No DB_ROOT_CERT environment variable found!"
fi

exec uvicorn embedding_service_bin:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1
