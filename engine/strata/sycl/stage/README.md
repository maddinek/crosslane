# Stage coordinator and transports

The host coordinator owns the model plan, context and layer order. Each backend
owns its device memory and queue. Transfers use explicit host buffers; no HIP
event, device pointer or SYCL USM allocation crosses the backend ABI.

The default split uses a SYCL prefix and HIP tail. Warm `hip-v1` dense ownership
uses an empty SYCL dense range and keeps Intel as an expert-only lane. Plans
bind geometry, dependencies, memory budgets and pack identity before execution.
