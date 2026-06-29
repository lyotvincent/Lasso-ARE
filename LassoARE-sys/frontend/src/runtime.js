export function runtimeLabel(runtime) {
  if (runtime?.profile === "cuda" && runtime?.degraded) {
    return "CUDA · Scanpy fallback";
  }
  if (runtime?.profile === "cuda") {
    return "CUDA · PyTorch + RAPIDS";
  }
  return "CPU · PyTorch + Scanpy";
}

export function sampleActionLabel(sample) {
  if (sample?.action === "load") {
    return "Load sample";
  }
  if (sample?.action === "download") {
    return "Download & load sample";
  }
  return "Sample not configured";
}
