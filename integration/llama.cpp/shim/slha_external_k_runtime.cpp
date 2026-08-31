// Keep the existing external-K implementation and lifecycle extensions in one
// translation unit. The suffix extension needs the private CCOS handle but does
// not expose it through any public header or C ABI.
#include "slha_external_k.cpp"
#include "slha_external_k_suffix.inc"
