# Install OpenAI triton_kernels from https://github.com/triton-lang/triton/tree/main/python/triton_kernels

# Set TRITON_KERNELS_SRC_DIR for use with local development with vLLM. We expect TRITON_KERNELS_SRC_DIR to
# be directly set to the triton_kernels python directory.
if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  message(STATUS "[triton_kernels] Fetch from $ENV{TRITON_KERNELS_SRC_DIR}")
  FetchContent_Declare(
          triton_kernels
          SOURCE_DIR $ENV{TRITON_KERNELS_SRC_DIR}
  )

else()
  if (VLLM_TARGET_DEVICE STREQUAL "rocm")
    set(TRITON_GIT "https://github.com/ROCm/triton.git")
    # Pinned to the head of release/internal/3.6.x, which is what PR 965 below
    # applies to. The previous pin is one commit behind it.
    set(TRITON_KERNELS_TAG "74e4569a70a34d7466d3707bf9d8a7762167068e")
  else()
    set(TRITON_GIT "https://github.com/triton-lang/triton.git")
    set(TRITON_KERNELS_TAG "v3.5.1")
  endif()
  message (STATUS "[triton_kernels] Fetch from ${TRITON_GIT}:${TRITON_KERNELS_TAG}")
  FetchContent_Declare(
          triton_kernels
          # TODO (varun) : Fetch just the triton_kernels directory from Triton
          GIT_REPOSITORY ${TRITON_GIT}
          GIT_TAG ${TRITON_KERNELS_TAG}
          GIT_PROGRESS TRUE
          SOURCE_SUBDIR python/triton_kernels/triton_kernels
  )
endif()

# Fetch content
FetchContent_MakeAvailable(triton_kernels)

if (NOT triton_kernels_SOURCE_DIR)
  message (FATAL_ERROR "[triton_kernels] Cannot resolve triton_kernels_SOURCE_DIR")
endif()

# 3.6.x sizes the CDNA4 MoE config for async copy being off, but Triton 3.8 turns it on.
# Carry the fix until PR 965 lands, then drop this and bump TRITON_KERNELS_TAG.
if (NOT DEFINED ENV{TRITON_KERNELS_SRC_DIR} AND VLLM_TARGET_DEVICE STREQUAL "rocm")
  set(_ogs_opt_flags
    "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py")
  file(READ "${_ogs_opt_flags}" _ogs_opt_flags_contents)
  # A re-configure reuses the populated tree, so only patch what is still unpatched.
  if (NOT _ogs_opt_flags_contents MATCHES "_is_async_copy_enabled_on_gfx950")
    message (STATUS "[triton_kernels] Applying ROCm/triton PR 965")
    execute_process(
      COMMAND patch --batch --forward -p1
        "--input=${CMAKE_CURRENT_LIST_DIR}/../patches/rocm_triton_kernels_965.patch"
      WORKING_DIRECTORY "${triton_kernels_SOURCE_DIR}"
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
endif()

if (DEFINED ENV{TRITON_KERNELS_SRC_DIR})
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/")
else()
  set(TRITON_KERNELS_PYTHON_DIR "${triton_kernels_SOURCE_DIR}/python/triton_kernels/triton_kernels/")
endif()

message (STATUS "[triton_kernels] triton_kernels is available at ${TRITON_KERNELS_PYTHON_DIR}")

add_custom_target(triton_kernels)

# Ensure the vllm/third_party directory exists before installation
install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/third_party/triton_kernels\")")

## Copy .py files to install directory.
install(DIRECTORY
        ${TRITON_KERNELS_PYTHON_DIR}
        DESTINATION
        vllm/third_party/triton_kernels/
        COMPONENT triton_kernels
        FILES_MATCHING PATTERN "*.py")
