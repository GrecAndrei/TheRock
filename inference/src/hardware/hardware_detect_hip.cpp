// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Queries HIP at runtime to find the active GPU's GCN arch string.
// Isolated in a .cpp so the C descriptor table stays header-free.

#include <hip/hip_runtime.h>
#include <cstring>

extern "C" const char *therock_detect_gfx_arch(void) {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return nullptr;

  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    return nullptr;

  // gcnArchName is e.g. "gfx1032:sramecc-:xnack-" — strip everything after ':'
  static char arch[32];
  const char *colon = strchr(prop.gcnArchName, ':');
  size_t len = colon ? (size_t)(colon - prop.gcnArchName)
                     : strlen(prop.gcnArchName);
  if (len >= sizeof(arch))
    return nullptr;
  memcpy(arch, prop.gcnArchName, len);
  arch[len] = '\0';
  return arch;
}
