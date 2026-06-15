// Copyright 2026 detrcpp authors. Apache-2.0.

#include "detr/weights/zip.hpp"

#include <miniz.h>

#include <cstring>
#include <fstream>
#include <utility>

namespace detr::weights {

namespace {
using core::Err;
using core::ErrorCode;
using core::Result;

mz_zip_archive* Z(void* p) { return static_cast<mz_zip_archive*>(p); }
}  // namespace

// ---- ZipReader ----

void ZipReader::Reset() noexcept {
  if (zip_ != nullptr) {
    mz_zip_reader_end(Z(zip_));
    delete Z(zip_);
    zip_ = nullptr;
  }
  buf_.clear();
  names_.clear();
}

ZipReader::~ZipReader() { Reset(); }

ZipReader::ZipReader(ZipReader&& other) noexcept
    : zip_(other.zip_), buf_(std::move(other.buf_)), names_(std::move(other.names_)) {
  other.zip_ = nullptr;
}

ZipReader& ZipReader::operator=(ZipReader&& other) noexcept {
  if (this != &other) {
    Reset();
    zip_ = other.zip_;
    buf_ = std::move(other.buf_);
    names_ = std::move(other.names_);
    other.zip_ = nullptr;
  }
  return *this;
}

Result<ZipReader> ZipReader::Open(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return Err(ErrorCode::Io, "cannot open '" + path.string() + "'");
  const auto size = static_cast<std::streamsize>(f.tellg());
  if (size < 0) return Err(ErrorCode::Io, "cannot size '" + path.string() + "'");
  f.seekg(0);
  ZipReader r;
  r.buf_.resize(static_cast<std::size_t>(size));
  if (size > 0 && !f.read(reinterpret_cast<char*>(r.buf_.data()), size)) {
    return Err(ErrorCode::Io, "cannot read '" + path.string() + "'");
  }
  r.zip_ = new mz_zip_archive;
  std::memset(r.zip_, 0, sizeof(mz_zip_archive));
  if (!mz_zip_reader_init_mem(Z(r.zip_), r.buf_.data(), r.buf_.size(), 0)) {
    delete Z(r.zip_);
    r.zip_ = nullptr;
    return Err(ErrorCode::ParseError, "'" + path.string() + "' is not a valid zip archive");
  }
  const mz_uint n = mz_zip_reader_get_num_files(Z(r.zip_));
  r.names_.reserve(n);
  for (mz_uint i = 0; i < n; ++i) {
    char name[512];
    const mz_uint len = mz_zip_reader_get_filename(Z(r.zip_), i, name, sizeof(name));
    r.names_.emplace_back(name, len > 0 ? len - 1 : 0);  // miniz length includes the NUL
  }
  return r;
}

bool ZipReader::Has(std::string_view name) const {
  if (zip_ == nullptr) return false;
  std::string s(name);
  return mz_zip_reader_locate_file(Z(zip_), s.c_str(), nullptr, 0) >= 0;
}

Result<std::vector<std::byte>> ZipReader::Read(std::string_view name) const {
  if (zip_ == nullptr) return Err(ErrorCode::Internal, "zip reader not open");
  std::string s(name);
  const int idx = mz_zip_reader_locate_file(Z(zip_), s.c_str(), nullptr, 0);
  if (idx < 0) return Err(ErrorCode::NotFound, "zip entry '" + s + "' not found");
  std::size_t out_size = 0;
  void* p = mz_zip_reader_extract_to_heap(Z(zip_), static_cast<mz_uint>(idx), &out_size, 0);
  if (p == nullptr) return Err(ErrorCode::ParseError, "failed to extract zip entry '" + s + "'");
  std::vector<std::byte> out(out_size);
  std::memcpy(out.data(), p, out_size);
  mz_free(p);
  return out;
}

// ---- ZipWriter ----

void ZipWriter::Reset() noexcept {
  if (zip_ != nullptr) {
    if (!finished_) {
      mz_zip_writer_finalize_archive(Z(zip_));
    }
    mz_zip_writer_end(Z(zip_));
    delete Z(zip_);
    zip_ = nullptr;
  }
}

ZipWriter::~ZipWriter() { Reset(); }

ZipWriter::ZipWriter(ZipWriter&& other) noexcept : zip_(other.zip_), finished_(other.finished_) {
  other.zip_ = nullptr;
}

ZipWriter& ZipWriter::operator=(ZipWriter&& other) noexcept {
  if (this != &other) {
    Reset();
    zip_ = other.zip_;
    finished_ = other.finished_;
    other.zip_ = nullptr;
  }
  return *this;
}

Result<ZipWriter> ZipWriter::Create(const std::filesystem::path& path) {
  ZipWriter w;
  w.zip_ = new mz_zip_archive;
  std::memset(w.zip_, 0, sizeof(mz_zip_archive));
  if (!mz_zip_writer_init_file(Z(w.zip_), path.string().c_str(), 0)) {
    delete Z(w.zip_);
    w.zip_ = nullptr;
    return Err(ErrorCode::Io, "cannot create zip '" + path.string() + "'");
  }
  return w;
}

Result<void> ZipWriter::Add(std::string_view name, std::span<const std::byte> data) {
  if (zip_ == nullptr) return Err(ErrorCode::Internal, "zip writer not open");
  std::string s(name);
  // MZ_NO_COMPRESSION → a "stored" entry, matching torch.save.
  if (!mz_zip_writer_add_mem(Z(zip_), s.c_str(), data.data(), data.size(), MZ_NO_COMPRESSION)) {
    return Err(ErrorCode::Io, "failed to add zip entry '" + s + "'");
  }
  return {};
}

Result<void> ZipWriter::Finish() {
  if (zip_ == nullptr) return Err(ErrorCode::Internal, "zip writer not open");
  if (!finished_) {
    if (!mz_zip_writer_finalize_archive(Z(zip_))) {
      return Err(ErrorCode::Io, "failed to finalize zip archive");
    }
    finished_ = true;
  }
  return {};
}

}  // namespace detr::weights
