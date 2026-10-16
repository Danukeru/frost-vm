# Reproducible build/test environment for frost-vm.
# Pin the base image by digest for byte-reproducible CI:
#   docker build --pull -t frost-vm .
#   docker run --rm frost-vm
FROM golang:1.24-bookworm@sha256:1a6d4452c65dea36aac2e2d606b01b4a029ec90cc1ae53890540ce6173ea77ac

WORKDIR /src

# Cache module downloads first (no external deps today, but future-proof).
COPY go.mod ./
RUN go mod download

COPY . .

# AVX kernels are runtime-dispatched (CPUID); the scalar fallback keeps this
# image testable on any amd64 host and on arm64 via the stub build.
ENV GOFLAGS=-trimpath
# The vm/ package runs real TFHE oblivious cycles (machine programs ~11 min,
# the load-extension pin ~40 min); the default 10-minute go test timeout is
# far too short for the whole module. Env-gated multi-hour end-to-end runs
# (FROST_RUN_FIB) are not part of the image build.
RUN go vet ./... && go build ./... && go test -count=1 -timeout 90m ./...

CMD ["go", "test", "-count=1", "-v", "-timeout", "90m", "./..."]
