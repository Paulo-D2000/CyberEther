# AGENTS.md — CyberEther fork (ldpc-dev)

Fork of luigifcruz/CyberEther (MIT). Branch `ldpc-dev` mission: add an
LDPC decoder block (CCSDS AR4JA r=1/2 k=1024), CPU module first, then a
Vulkan compute module. Reference algorithm: NVlabs/sionna-rk
`plugins/ldpc_cuda` (Apache-2.0, 5G NR flooded min-sum, int8 LLRs) —
algorithm transfers, edge tables must be regenerated for AR4JA.

## Policy
- Git identity: Paulo-D2000 <highvoltagegenerator@hotmail.com>
- AI disclosure: not required (personal fork)
- Upstream license: MIT — contributions must be MIT-compatible;
  sionna-derived code structure must keep Apache-2.0 attribution

## Build (verified on this machine, Ubuntu 22.04)
export PATH="$HOME/.local/bin:$PATH"   # meson/ninja installed via pip --user
meson setup -Dbuildtype=debugoptimized build
meson compile -C build

Dependencies verified present: glslang-tools 12.0.0, libvulkan-dev 1.3.239,
mesa-vulkan-drivers 23.0.4, libx11-dev, libwayland-dev, zenity.
GPU verified: Intel HD Graphics 620 (KBL GT2), Vulkan 1.3.238, Mesa ANV.
Tests require Catch2 (`docs/contributing.md`); check `dpkg -l catch2`
before relying on `module_test_lst` builds.

## Commit style (verified from git log, follow exactly)
Conventional commits, lowercase, imperative, no trailing period:
`feat(io): add TCP receiver and transmitter blocks`
- Types in use: feat, fix, refactor, docs, test, perf, chore, build, ci
- Scopes in use: io, dsp, sdr, scheduler, platform, runtime, memory,
  agent, compositor, settings, remote, build, etc.
- Short subjects; bodies rare. Branches: feat/<name>, fix/<name>.
- Upstream PRs: branch off main, PR title repeats the feat subject.
- Our work: LDPC block PR = `feat(dsp): add LDPC decoder block` on
  branch `feat/ldpc-block`. Vulkan runtime question = upstream issue
  BEFORE any Vulkan code (contributing.md: issue-first rule).

## Code rules (from docs/contributing.md — they win over any global rules)
- Google C++ Style Guide, 88-char lines, 4-space indent, no exceptions
- All functions return `Result` (SUCCESS/WARNING/ERROR/FATAL/INCOMPLETE/
  RECREATE/RELOAD/SKIP/YIELD/TIMEOUT as applicable)
- Use the JST logger, never printf (JST_DEBUG env var selects level)
- Create an upstream issue before starting any change intended for upstream

## Block/module architecture (from docs/blocks-and-modules.md + agc example)
- Block = flowgraph node: `Config` struct with JST_BLOCK_TYPE/DOMAIN/PARAMS/
  DESCRIPTION macros + `Impl` with define()/create() hooks.
- Module = per-device kernel: `Config` with JST_MODULE_TYPE/PARAMS + `Impl`
  inheriting Module::Impl, DynamicConfig<Config>, a runtime context
  (NativeCpuRuntimeContext | NativeCudaRuntimeContext), Scheduler::Context.
  Hooks: validate/define/create/destroy/reconfigure + computeSubmit().
- File layout (copy agc): include/jetstream/domains/dsp/<name>/{block.hh,
  module.hh}; src/domains/dsp/<name>/{block_impl.cc, module_impl.hh,
  module_impl.cc, module_impl_native_cpu.cc, module_tests.cc,
  block_tests.cc, meson.build}. Wire in: src/domains/dsp/meson.build
  subdir('<name>'). Registration is static via JST_REGISTER_BLOCK /
  JST_REGISTER_MODULE macros — dispatch key = config type string +
  DeviceType + RuntimeType + provider ("generic").
- computeSubmit MUST NOT block (single compute thread per instance).
  Sources answer hasPendingCompute() immediately.
- Module is STATELESS-eligible only if pure function of config+inputs
  (LDPC decode qualifies; it publishes outputs only).
- Prefer RECREATE over clever in-place reconfigure.
- One-to-one block/module: validate in the module, not both.

## Critical verified facts (do not re-derive, but re-verify if older than ~1 month)
- ONLY CPU and CUDA compute runtimes exist (runtime_context_native_cpu.hh,
  runtime_context_native_cuda.hh). NO NativeVulkanRuntimeContext upstream;
  README "Vulkan compute" is aspirational.
- Backend::Vulkan (include/jetstream/backend/devices/vulkan/base.hh)
  exposes getDevice(), getComputeQueue(), getDescriptorPool(),
  getDefaultCommandPool/Buffer/Fence() → a module can build its own Vulkan
  compute pipeline through these handles. This is the Phase 2 path.
- Vulkan tensor storage exists: include/jetstream/memory/devices/vulkan/buffer.hh.
- `[NEW BLOCK HOOK]` markers described in contributing.md are mostly gone;
  the only hook left is [NEW DEPENDENCY HOOK] in
  src/compositor/default/views/modal/settings/legal.hh.
- Compute shaders in-tree live in resources/shaders/ (GLSL .comp, e.g.
  waterfall_filter.comp) — compiled via glslang in the build.

## What NOT to do
- Don't register DeviceType::VULKAN modules yet — no runtime dispatches them.
- Don't block in computeSubmit; drain ring buffers inline instead.
- Don't duplicate module validation in the block layer.
- Don't print with printf; use JST_DEBUG/JST_INFO/... logger.
