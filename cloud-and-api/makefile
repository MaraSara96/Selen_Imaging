IMAGE = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/selenimaging/$(GAR_IMAGE):prod

# Default target: Run everything sequentially
all: build push deploy

# Docker Build
build:
	docker build --platform linux/amd64 -t $(IMAGE) .

# Docker Push
push: build
	docker push $(IMAGE)

# Deploy to Cloud Run
deploy: push
	gcloud run deploy \
	  --image $(IMAGE) \
	  --region $(GCP_REGION) \
	  --env-vars-file .env.yaml
