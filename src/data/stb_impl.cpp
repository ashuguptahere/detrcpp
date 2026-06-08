// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Single translation unit that compiles the vendored stb image headers (public
// domain). Every other file includes the headers for declarations only.

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_resize2.h"
#include "stb_image_write.h"
