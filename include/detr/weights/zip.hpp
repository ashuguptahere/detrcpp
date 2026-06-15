// Copyright 2026 detrcpp authors. Apache-2.0.
//
// Minimal ZIP read/write over miniz — the container format `torch.save` uses for
// modern `.pth` checkpoints (a zip of `data.pkl` + `data/<key>` storage blobs).
// Torch-free, so the same code serves the LibTorch build, the ONNX exporter, and
// the tests. Entries are read whole into memory (checkpoints are not huge); writes
// produce "stored" (uncompressed) entries, matching torch.save.

#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "detr/core/result.hpp"

namespace detr::weights {

// Reads a ZIP archive from a file into memory and extracts entries by exact name.
class ZipReader {
 public:
  ZipReader() = default;
  ~ZipReader();
  ZipReader(ZipReader&& other) noexcept;
  ZipReader& operator=(ZipReader&& other) noexcept;
  ZipReader(const ZipReader&) = delete;
  ZipReader& operator=(const ZipReader&) = delete;

  static core::Result<ZipReader> Open(const std::filesystem::path& path);

  const std::vector<std::string>& Names() const { return names_; }
  bool Has(std::string_view name) const;
  core::Result<std::vector<std::byte>> Read(std::string_view name) const;

 private:
  void Reset() noexcept;
  void* zip_ = nullptr;          // mz_zip_archive* (owns the reader state)
  std::vector<std::byte> buf_;   // owns the archive bytes (read from memory)
  std::vector<std::string> names_;
};

// Writes "stored" (uncompressed) ZIP entries to a file. Call Finish() (or let the
// destructor finalize) to flush the central directory.
class ZipWriter {
 public:
  ZipWriter() = default;
  ~ZipWriter();
  ZipWriter(ZipWriter&& other) noexcept;
  ZipWriter& operator=(ZipWriter&& other) noexcept;
  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;

  static core::Result<ZipWriter> Create(const std::filesystem::path& path);
  core::Result<void> Add(std::string_view name, std::span<const std::byte> data);
  core::Result<void> Finish();

 private:
  void Reset() noexcept;
  void* zip_ = nullptr;  // mz_zip_archive*
  bool finished_ = false;
};

}  // namespace detr::weights
