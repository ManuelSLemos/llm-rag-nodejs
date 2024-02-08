bash:
	docker run --rm \
	-w /app \
	-v ${PWD}:/app \
	-it node:20 bash