web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --threads 2 --worker-class sync --max-requests 50 --max-requests-jitter 10

