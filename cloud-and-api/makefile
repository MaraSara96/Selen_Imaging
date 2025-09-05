# Variablen
GCP_REGION ?= europe-west1
GCP_PROJECT ?= le-wagon-testprojekt
GAR_IMAGE ?= selen-imaging_torstenweindl

IMAGE = $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/selenimaging/$(GAR_IMAGE):prod

# Default Target: alles nacheinander ausführen
all: build push deploy

# Docker Build
build:
	docker build --platform linux/amd64 -t $(IMAGE) .

# Docker Push
push: build
	docker push $(IMAGE)

# Deploy zu Cloud Run
deploy: push
	gcloud run deploy \
	  --image $(IMAGE) \
	  --region $(GCP_REGION) \
	  --env-vars-file .env.yaml
